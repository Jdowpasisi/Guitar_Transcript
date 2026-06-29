"""
P7: Celery Task — Full P1–P6 Pipeline
=======================================
This is the core worker task. It:
  1. Optionally runs Demucs stem separation (P2)
  2. Runs Basic Pitch for note detection (P5 model)
  3. Runs ChordCNN for chord recognition (P4 model)
  4. Runs VoicingLSTM to assign (string, fret) to each note (P6 model)
  5. Renders ASCII tab
  6. Cleans up the staging file
  7. Returns the full TranscriptionResult dict to Redis

Progress updates are published via self.update_state() so /status/{id}
can show the current step while the job is running.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from celery import Task
from celery.utils.log import get_task_logger

from .celery_app import celery_app

logger = get_task_logger(__name__)

# ─── Optional model imports ────────────────────────────────────────────────────
# Wrapped in try/except so the API container can import this module even when
# heavy ML libraries are not installed (e.g. during CI or unit testing).

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ─── Guitar constants ──────────────────────────────────────────────────────────

OPEN_STRINGS_MIDI = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
STRING_NAMES       = ["E2", "A2", "D3", "G3", "B3", "E4"]
STANDARD_PITCH_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

def midi_to_pitch_name(midi: int) -> str:
    octave = (midi // 12) - 1
    note   = STANDARD_PITCH_NAMES[midi % 12]
    return f"{note}{octave}"


# ─── Progress helper ───────────────────────────────────────────────────────────

def _progress(task: Task, step: str, percent: int) -> None:
    task.update_state(
        state="STARTED",
        meta={"step": step, "percent": percent},
    )
    logger.info("[%s] %s (%d%%)", task.request.id, step, percent)


# ─── Sub-pipeline functions ────────────────────────────────────────────────────

def _load_audio(audio_path: str, sr: int = 22050) -> Tuple[Any, float]:
    """Load and normalise audio. Returns (waveform_array, duration_sec)."""
    if not _HAS_LIBROSA:
        raise RuntimeError("librosa not installed in worker environment.")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    # peak normalise
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak
    duration = float(len(y) / sr)
    return y, duration


def _run_stem_separation(audio_path: str) -> Optional[str]:
    """
    Run Demucs (P2) and return path to the 'other' (guitar) stem.
    Returns None if Demucs is not available — caller falls back to raw audio.
    """
    try:
        from demucs.api import Separator
        sep = Separator(model="htdemucs")
        wav_tensor, sr = sep.load_audio(audio_path)
        # Separate returns dict: {"drums": ..., "bass": ..., "vocals": ..., "other": ...}
        stems = sep.separate_tensor(wav_tensor, sr)
        guitar_stem = stems["other"]  # shape: (2, T) stereo

        stem_path = str(Path(audio_path).with_suffix("")) + "_guitar_stem.wav"
        sep.save_audio(guitar_stem, stem_path, samplerate=sr)
        return stem_path
    except Exception as exc:
        logger.warning("Stem separation skipped: %s", exc)
        return None


def _run_basic_pitch(audio_path: str) -> List[Dict]:
    """
    Run Basic Pitch (ONNX) and return list of note dicts.
    Falls back to an empty list if not available.
    """
    try:
        import os as _os
        import basic_pitch
        from basic_pitch.inference import predict

        # Explicitly locate the ONNX model file.
        # basic-pitch 0.4.0 defaults to the TF SavedModel directory (nmp/)
        # which fails on TF 2.16+ with AttributeError('add_slot').
        # We must pass the nmp.onnx path directly — same fix as benchmark.py.
        bp_models_dir = _os.path.join(
            _os.path.dirname(basic_pitch.__file__),
            "saved_models", "icassp_2022",
        )
        onnx_path = _os.path.join(bp_models_dir, "nmp.onnx")

        if not _os.path.exists(onnx_path):
            logger.warning("Basic Pitch ONNX model not found at %s", onnx_path)
            return []

        model_output, midi_data, note_events = predict(
            audio_path,
            onnx_path,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,       # ms
            minimum_frequency=82.41,      # E2
            maximum_frequency=2000,       # well above top fret
        )
        notes = []
        for note in note_events:
            # note_events: list of (start_time, end_time, pitch, amplitude, ...)
            if len(note) >= 4:
                start, end, pitch, amp = note[0], note[1], note[2], note[3]
            else:
                start, end, pitch = note[0], note[1], note[2]
                amp = 0.8
            notes.append({
                "onset":  float(start),
                "offset": float(end),
                "pitch":  int(pitch),
                "confidence": float(min(amp, 1.0)),
            })
        return sorted(notes, key=lambda n: n["onset"])
    except Exception as exc:
        logger.warning("Basic Pitch failed: %s — returning empty note list", exc)
        return []


def _run_chord_cnn(audio_path: str, sr: int = 22050) -> List[Dict]:
    """
    Run ChordCNN (P4) on 1-second sliding windows.
    Returns list of {start, end, label, confidence} dicts.
    """
    chord_events: List[Dict] = []

    if not (_HAS_TORCH and _HAS_LIBROSA and _HAS_NUMPY):
        logger.warning("ChordCNN skipped — missing dependencies.")
        return chord_events

    # Locate model checkpoint
    project_root = Path(os.getenv("GUITARAI_ROOT", "/app"))
    model_path   = project_root / "models" / "chord_cnn.pth"
    label_map_path = project_root / "data" / "processed" / "chord_dataset" / "label_map.json"

    if not model_path.exists():
        logger.warning("chord_cnn.pth not found at %s — skipping chord detection", model_path)
        return chord_events

    try:
        import json
        import torch
        import torch.nn.functional as F

        # Import model architecture
        sys.path.insert(0, str(project_root))
        from src.ml.models import load_model

        model  = load_model(str(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()

        with open(label_map_path) as f:
            label_map = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}

        # CQT feature extraction
        y, _  = librosa.load(audio_path, sr=sr, mono=True)
        y     = librosa.effects.preemphasis(y, coef=0.97)
        C     = librosa.cqt(y, sr=sr, hop_length=256, fmin=82.41, n_bins=84, bins_per_octave=12)
        C_mag = np.abs(C)
        c_max = C_mag.max()
        if c_max > 1e-6:
            C_mag = C_mag / c_max

        # Sliding window: 1s windows, 0.5s stride
        window_frames = 87
        stride_frames = int(0.5 * sr / 256)
        n_frames      = C_mag.shape[1]
        hop_sec       = 256 / sr  # seconds per frame

        windows, starts = [], []
        t = 0
        while t + window_frames <= n_frames:
            windows.append(C_mag[:, t : t + window_frames])
            starts.append(t * hop_sec)
            t += stride_frames

        if not windows:
            return chord_events

        X = torch.tensor(np.stack(windows)[:, np.newaxis], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X)
            probs  = F.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1).cpu().numpy()
            confs  = probs.max(dim=-1).values.cpu().numpy()

        # Merge consecutive identical chords
        chord_events = []
        prev_label, prev_start, prev_conf_sum, prev_count = None, None, 0.0, 0
        window_sec = 1.0

        for i, (pred, conf, start) in enumerate(zip(preds, confs, starts)):
            label = idx_to_label.get(int(pred), "N")
            if label == prev_label:
                prev_conf_sum += float(conf)
                prev_count    += 1
            else:
                if prev_label is not None:
                    chord_events.append({
                        "start":      prev_start,
                        "end":        start,
                        "label":      prev_label,
                        "confidence": round(prev_conf_sum / prev_count, 4),
                    })
                prev_label, prev_start = label, start
                prev_conf_sum, prev_count = float(conf), 1

        if prev_label is not None:
            chord_events.append({
                "start":      prev_start,
                "end":        prev_start + window_sec,
                "label":      prev_label,
                "confidence": round(prev_conf_sum / prev_count, 4),
            })

    except Exception as exc:
        logger.error("ChordCNN error: %s\n%s", exc, traceback.format_exc())

    return chord_events


def _greedy_voicing(midi_pitch: int, prev_fret: int) -> Tuple[int, int]:
    """
    Greedy (string, fret) assignment — fallback when LSTM is unavailable.
    Picks the string/fret closest in fret-distance to the previous position.
    """
    best_string, best_fret, best_dist = 0, 0, float("inf")
    for s, open_midi in enumerate(OPEN_STRINGS_MIDI):
        fret = midi_pitch - open_midi
        if 0 <= fret <= 22:
            dist = abs(fret - prev_fret)
            if dist < best_dist:
                best_dist, best_string, best_fret = dist, s, fret
    return best_string, best_fret


def _run_voicing_lstm(notes: List[Dict], project_root: Path) -> List[Dict]:
    """
    Run the VoicingLSTM (P6) to assign (string, fret) to each note.
    Falls back to greedy heuristic if model is unavailable.
    """
    if not notes:
        return notes

    voiced = []
    use_greedy = True

    if _HAS_TORCH:
        model_path = project_root / "models" / "voicing_lstm.pth"
        if model_path.exists():
            try:
                import torch
                import torch.nn.functional as F

                sys.path.insert(0, str(project_root))
                from src.ml.models import VoicingLSTM

                checkpoint = torch.load(str(model_path), map_location="cpu")
                model = VoicingLSTM()
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                # Build input tensors — single sequence (B=1)
                midi_pitches  = torch.tensor([[n["pitch"] for n in notes]], dtype=torch.long)
                prev_positions = torch.zeros_like(midi_pitches)   # BOS = 0 everywhere
                onsets        = [n["onset"] for n in notes]
                delta_t_vals  = [0.0] + [onsets[i] - onsets[i-1] for i in range(1, len(onsets))]
                delta_t       = torch.tensor([delta_t_vals], dtype=torch.float32)
                lengths       = torch.tensor([len(notes)], dtype=torch.long)

                with torch.no_grad():
                    logits = model(midi_pitches, prev_positions, delta_t, lengths)
                    preds  = logits[0].argmax(dim=-1).cpu().numpy()

                for note, pred in zip(notes, preds):
                    string = int(pred) // 23
                    fret   = int(pred) % 23
                    voiced.append({
                        **note,
                        "string":         string,
                        "fret":           fret,
                        "string_name":    STRING_NAMES[string],
                        "pitch_name":     midi_to_pitch_name(note["pitch"]),
                        "voicing_source": "lstm",
                    })
                use_greedy = False
                logger.info("VoicingLSTM assigned voicings for %d notes", len(notes))
            except Exception as exc:
                logger.warning("VoicingLSTM failed: %s — falling back to greedy", exc)

    if use_greedy:
        prev_fret = 0
        for note in notes:
            string, fret = _greedy_voicing(note["pitch"], prev_fret)
            prev_fret = fret
            voiced.append({
                **note,
                "string":         string,
                "fret":           fret,
                "string_name":    STRING_NAMES[string],
                "pitch_name":     midi_to_pitch_name(note["pitch"]),
                "voicing_source": "greedy",
            })

    return voiced


def _render_ascii_tab(notes: List[Dict], duration_sec: float) -> str:
    """
    Render a 6-line ASCII guitar tab string.
    Time is quantised to 0.1s columns. Each column is ~2 chars wide.
    """
    if not notes:
        return (
            "e|---|\n"
            "B|---|\n"
            "G|---|\n"
            "D|---|\n"
            "A|---|\n"
            "E|---|"
        )

    resolution = 0.1  # seconds per column
    n_cols = max(int(duration_sec / resolution) + 2, 10)

    # 6 rows, high E first (string 5 → row 0, string 0 → row 5)
    grid = [["-"] * n_cols for _ in range(6)]

    for note in notes:
        col     = int(note["onset"] / resolution)
        string  = note.get("string", 0)
        fret    = note.get("fret", 0)
        row     = 5 - string    # flip: high E at top
        if 0 <= col < n_cols and 0 <= row < 6:
            grid[row][col] = str(fret)

    string_labels = ["e", "B", "G", "D", "A", "E"]
    lines = []
    for i, (label, row) in enumerate(zip(string_labels, grid)):
        lines.append(f"{label}|{''.join(row)}|")

    return "\n".join(lines)


# ─── Main Celery task ──────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.tasks.run_pipeline",
    max_retries=0,
    acks_late=True,
)
def run_pipeline(self: Task, audio_path: str, job_id: str) -> Dict[str, Any]:
    """
    Full P1–P6 transcription pipeline.

    Args:
        audio_path: Absolute path to the uploaded audio file.
        job_id:     UUID matching the Celery task ID.

    Returns:
        Dict matching TranscriptionResult schema, stored in Redis by Celery.
    """
    t_start  = time.monotonic()
    project_root = Path(os.getenv("GUITARAI_ROOT", "/app"))

    try:
        # ── Step 1: Load audio ────────────────────────────────────────────────
        _progress(self, "Loading audio", 5)
        y, duration_sec = _load_audio(audio_path)
        logger.info("Audio loaded: %.1f seconds", duration_sec)

        # ── Step 2: Stem separation (Demucs) ─────────────────────────────────
        _progress(self, "Running stem separation (Demucs)", 15)
        guitar_path = _run_stem_separation(audio_path)
        stem_applied = guitar_path is not None
        working_path = guitar_path or audio_path
        models_used  = []
        if stem_applied:
            models_used.append("Demucs htdemucs")
            logger.info("Using guitar stem: %s", guitar_path)
        else:
            logger.info("Stem separation skipped; using raw audio.")

        # ── Step 3: Note transcription (Basic Pitch) ──────────────────────────
        _progress(self, "Transcribing notes (Basic Pitch)", 35)
        raw_notes = _run_basic_pitch(working_path)
        models_used.append("Basic Pitch ONNX")
        logger.info("Basic Pitch found %d notes", len(raw_notes))

        # ── Step 4: Chord detection (ChordCNN) ───────────────────────────────
        _progress(self, "Detecting chords (ChordCNN)", 55)
        chord_events = _run_chord_cnn(working_path)
        models_used.append("ChordCNN")
        logger.info("ChordCNN found %d chord segments", len(chord_events))

        # ── Step 5: Voicing assignment (VoicingLSTM) ─────────────────────────
        _progress(self, "Assigning voicings (VoicingLSTM)", 75)
        voiced_notes = _run_voicing_lstm(raw_notes, project_root)
        if voiced_notes and voiced_notes[0].get("voicing_source") == "lstm":
            models_used.append("VoicingLSTM")
        else:
            models_used.append("Greedy heuristic")

        # ── Step 6: Render ASCII tab ──────────────────────────────────────────
        _progress(self, "Rendering tablature", 90)
        tab_str = _render_ascii_tab(voiced_notes, duration_sec)

        # ── Step 7: Package result ────────────────────────────────────────────
        processing_time = time.monotonic() - t_start

        result = {
            "job_id": job_id,
            "chords": chord_events,
            "tab":    tab_str,
            "notes":  voiced_notes,
            "pipeline": {
                "stem_separation":     stem_applied,
                "models_used":         models_used,
                "audio_duration_sec":  round(duration_sec, 2),
                "processing_time_sec": round(processing_time, 2),
                "note_count":          len(voiced_notes),
                "chord_count":         len(chord_events),
            },
        }

        logger.info(
            "Pipeline complete in %.1fs: %d notes, %d chords",
            processing_time, len(voiced_notes), len(chord_events),
        )
        return result

    except Exception as exc:
        logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise  # Celery marks task as FAILURE and stores exc in backend

    finally:
        # Always clean up the staging upload (not the stem — kept for debugging)
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass
