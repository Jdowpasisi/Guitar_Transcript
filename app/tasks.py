"""
P7/P13: Celery Tasks — Full Pipeline (Audio + Vision + Fusion)
==============================================================
P13 adds three new execution paths on top of the P7 audio-only pipeline:

  1. run_pipeline_with_video(audio_path, video_path, job_id)
     - Runs P9→P11 vision pipeline in parallel with audio pipeline
     - Routes through FusionModel (P12) when video features are available
     - Falls back to LSTM / greedy when vision fails

  2. run_pipeline_from_url(url, job_id)
     - Downloads video via yt-dlp
     - Splits into audio + video
     - Then runs run_pipeline_with_video logic

  3. run_pipeline(audio_path, job_id)  [unchanged from P7]
     - Audio-only path: Basic Pitch + ChordCNN + VoicingLSTM

Progress updates are published via self.update_state() so /status/{id}
can show the current step while the job is running.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from celery import Task
from celery.utils.log import get_task_logger

from .celery_app import celery_app

logger = get_task_logger(__name__)

# ─── Optional model imports ────────────────────────────────────────────────────

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
            maximum_frequency=2000,
        )
        notes = []
        for note in note_events:
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

        sys.path.insert(0, str(project_root))
        from src.ml.models import load_model

        model  = load_model(str(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()

        with open(label_map_path) as f:
            label_map = json.load(f)
        idx_to_label = {v: k for k, v in label_map.items()}

        y, _  = librosa.load(audio_path, sr=sr, mono=True)
        y     = librosa.effects.preemphasis(y, coef=0.97)
        C     = librosa.cqt(y, sr=sr, hop_length=256, fmin=82.41, n_bins=84, bins_per_octave=12)
        C_mag = np.abs(C)
        c_max = C_mag.max()
        if c_max > 1e-6:
            C_mag = C_mag / c_max

        window_frames = 87
        stride_frames = int(0.5 * sr / 256)
        n_frames      = C_mag.shape[1]
        hop_sec       = 256 / sr

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

                sys.path.insert(0, str(project_root))
                from src.ml.models import VoicingLSTM

                checkpoint = torch.load(str(model_path), map_location="cpu")
                model = VoicingLSTM()
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                midi_pitches  = torch.tensor([[n["pitch"] for n in notes]], dtype=torch.long)
                prev_positions = torch.zeros_like(midi_pitches)
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


# ─── P13: Vision Pipeline ──────────────────────────────────────────────────────

def _run_vision_pipeline(video_path: str, project_root: Path) -> Optional[Dict]:
    """
    Run the P9→P11 vision pipeline on a video file.

    Returns a dict with video features (fret/string positions per timestamp)
    that the FusionModel can consume, or None if the vision pipeline fails.

    The pipeline:
      1. P9: Extract frames + audio from video (FFmpeg)
      2. P10: Detect guitar neck bounding box with YOLOv8n
      3. P9: Warp fretboard using detected neck bbox
      4. P11: Track fingertips with MediaPipe HandLandmarker
      5. Parse finger_tracking.csv into structured video features
    """
    try:
        import tempfile
        sys.path.insert(0, str(project_root))

        video_stem = Path(video_path).stem
        output_dir = project_root / "outputs" / "frames" / f"p13_{video_stem}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract frames using P9
        from src.vision.extract_frames import extract_frames
        frames_dir = output_dir / "frames"
        audio_path = output_dir / "audio.wav"
        extract_frames(video_path, str(output_dir), fps=5)

        if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
            logger.warning("Vision pipeline: No frames extracted from video")
            return None

        # Step 2 & 3: Auto-detect neck with P10 NeckDetector, get homography
        homography_path = output_dir / "homography.npy"
        corners = None

        try:
            from src.vision.guitar_vision import GuitarVisionPipeline
            vision = GuitarVisionPipeline(project_root=str(project_root))

            # Get first frame to detect neck
            first_frames = sorted(frames_dir.glob("*.png"))[:3]
            for frame_path in first_frames:
                import cv2
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    result = vision.process_frame(frame)
                    if result.get("neck_detected"):
                        # Save homography for P11
                        import numpy as np_local
                        H = result.get("homography")
                        if H is not None:
                            np_local.save(str(homography_path), H)
                            corners = result.get("corners")
                            break
        except Exception as e:
            logger.warning("Vision: Neck detector failed: %s — trying headless corners", e)

        # If no homography from neck detector, use a synthetic default
        if not homography_path.exists():
            try:
                import numpy as np_local
                import cv2
                # Read first frame for dimensions
                first_frames = sorted(frames_dir.glob("*.png"))
                if first_frames:
                    frame = cv2.imread(str(first_frames[0]))
                    h, w = frame.shape[:2]
                    # Approximate corners: 10% margins
                    src_pts = np_local.float32([
                        [w*0.1, h*0.2], [w*0.9, h*0.2],
                        [w*0.9, h*0.8], [w*0.1, h*0.8]
                    ])
                    dst_pts = np_local.float32([
                        [0, 0], [600, 0], [600, 200], [0, 200]
                    ])
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                    if H is not None:
                        np_local.save(str(homography_path), H)
            except Exception as e2:
                logger.warning("Vision: Could not create fallback homography: %s", e2)

        if not homography_path.exists():
            logger.warning("Vision pipeline: No homography available")
            return None

        # Step 4: Run P11 finger tracker
        try:
            from src.vision.finger_tracker import process_frame_directory
            tracking_result = process_frame_directory(
                str(frames_dir),
                str(homography_path),
                str(output_dir),
                make_video=False,
            )
        except Exception as e:
            logger.warning("Vision: Finger tracker failed: %s", e)
            return None

        # Step 5: Parse the CSV
        csv_path = output_dir / "finger_tracking.csv"
        if not csv_path.exists():
            logger.warning("Vision: No finger tracking CSV produced")
            return None

        video_features = _parse_finger_tracking_csv(str(csv_path))
        logger.info(
            "Vision pipeline complete: %d finger detections from video",
            len(video_features)
        )
        return video_features

    except Exception as exc:
        logger.error("Vision pipeline error: %s\n%s", exc, traceback.format_exc())
        return None


def _parse_finger_tracking_csv(csv_path: str) -> List[Dict]:
    """
    Parse the P11 finger_tracking.csv into a list of
    {timestamp, finger_id, string, fret, confidence} dicts.
    """
    import csv
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "timestamp":  float(row.get("timestamp", 0)),
                        "finger_id":  int(row.get("finger_id", 0)),
                        "string":     int(row.get("string", 0)),
                        "fret":       int(row.get("fret", 0)),
                        "confidence": float(row.get("confidence", 0.5)),
                    })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.warning("Could not parse finger tracking CSV: %s", e)
    return rows


# ─── P13: Fusion Model voicing ─────────────────────────────────────────────────

def _run_fusion_model(notes: List[Dict], video_features: List[Dict], project_root: Path) -> List[Dict]:
    """
    Run the FusionModel (P12) to assign (string, fret) using both audio and video.

    For each note, finds the nearest video finger detection by timestamp,
    builds the 56-dim audio + 7-dim video feature vectors, and runs inference.

    Falls back to LSTM if the FusionModel checkpoint is not available.
    """
    if not notes:
        return notes

    if not _HAS_TORCH or not _HAS_NUMPY:
        logger.warning("FusionModel: PyTorch/NumPy not available — falling back to LSTM")
        return _run_voicing_lstm(notes, project_root)

    model_path = project_root / "models" / "fusion_model.pth"
    if not model_path.exists():
        logger.warning("FusionModel checkpoint not found at %s — falling back to LSTM", model_path)
        return _run_voicing_lstm(notes, project_root)

    try:
        import torch

        sys.path.insert(0, str(project_root))
        from src.ml.fusion_model import FusionModel
        from src.ml.fusion_dataset import _build_audio_features, _build_video_features

        checkpoint = torch.load(str(model_path), map_location="cpu")
        # Handle both direct state_dict and wrapped checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model = FusionModel()
        model.load_state_dict(state_dict)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        # Build per-note feature vectors
        audio_feats_list = []
        video_feats_list = []

        for i, note in enumerate(notes):
            # Audio features (56-dim) from the note
            midi_pitch = note["pitch"]
            delta_t = notes[i]["onset"] - notes[i-1]["onset"] if i > 0 else 0.0
            audio_feat = _build_audio_feature_vector(note, delta_t)
            audio_feats_list.append(audio_feat)

            # Video features (7-dim): find nearest timestamp
            video_feat = _find_nearest_video_feature(note["onset"], video_features)
            video_feats_list.append(video_feat)

        # Stack into tensors: (1, T, D)
        audio_tensor = torch.tensor(
            np.array(audio_feats_list)[np.newaxis], dtype=torch.float32
        ).to(device)
        video_tensor = torch.tensor(
            np.array(video_feats_list)[np.newaxis], dtype=torch.float32
        ).to(device)
        lengths = torch.tensor([len(notes)], dtype=torch.long)

        with torch.no_grad():
            logits = model(audio_tensor, video_tensor, lengths)
            preds = logits[0].argmax(dim=-1).cpu().numpy()

        voiced = []
        for note, pred in zip(notes, preds):
            string = int(pred) // 23
            fret   = int(pred) % 23
            voiced.append({
                **note,
                "string":         string,
                "fret":           fret,
                "string_name":    STRING_NAMES[string],
                "pitch_name":     midi_to_pitch_name(note["pitch"]),
                "voicing_source": "fusion",
            })
        logger.info("FusionModel assigned voicings for %d notes", len(notes))
        return voiced

    except Exception as exc:
        logger.warning(
            "FusionModel failed: %s — falling back to LSTM\n%s",
            exc, traceback.format_exc()
        )
        return _run_voicing_lstm(notes, project_root)


def _build_audio_feature_vector(note: Dict, delta_t: float) -> List[float]:
    """
    Build a 56-dim audio feature vector for one note.
    Matches the FusionDataset's audio feature schema.
    """
    midi_pitch = note.get("pitch", 60)
    confidence = note.get("confidence", 0.8)
    duration   = note.get("offset", note.get("onset", 0) + 0.5) - note.get("onset", 0)
    pitch_class = midi_pitch % 12

    # [midi_pitch/127, confidence, delta_t/5, duration/5, pitch_class/11, ...]
    # + 51-dim chord probs (uniform fallback)
    base = [
        midi_pitch / 127.0,
        float(confidence),
        min(delta_t, 5.0) / 5.0,
        min(duration, 5.0) / 5.0,
        pitch_class / 11.0,
    ]
    chord_probs = [1.0 / 51.0] * 51   # uniform prior (no ChordCNN output here)
    return base + chord_probs  # 5 + 51 = 56


def _find_nearest_video_feature(onset_time: float, video_features: List[Dict]) -> List[float]:
    """
    Build a 7-dim video feature vector for a note by finding the nearest
    video detection by timestamp. Returns "no video" vector if no match.
    """
    if not video_features:
        # video_available = 0 → model uses audio-only path
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Find closest detection
    best = min(video_features, key=lambda d: abs(d["timestamp"] - onset_time))
    time_gap = abs(best["timestamp"] - onset_time)

    # If the nearest detection is more than 0.5s away, treat as no video
    if time_gap > 0.5:
        return [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0]  # video_available=0

    return [
        best["fret"] / 22.0,         # fret_number (normalized)
        best["string"] / 5.0,        # string_number (normalized)
        best["finger_id"] / 4.0,     # finger_id (normalized)
        best["confidence"],          # detection_confidence
        0.8,                         # frame_quality (assumed good)
        0.6,                         # num_fingers_detected (normalized)
        1.0,                         # video_available = 1
    ]


# ─── P13: yt-dlp YouTube downloader ───────────────────────────────────────────

def _download_youtube(url: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Download a video from YouTube (or any yt-dlp supported site).
    Returns (audio_path, video_path). Both may be None on failure.

    Downloads best quality video (with audio embedded) in mp4 format.
    Also extracts a separate audio file for the audio pipeline.
    """
    try:
        import yt_dlp

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        video_template = str(output_dir_path / "%(id)s.%(ext)s")
        audio_template = str(output_dir_path / "%(id)s_audio.%(ext)s")

        # Download video (best mp4 up to 720p for speed)
        ydl_video_opts = {
            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": video_template,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }

        downloaded_video = None
        video_id = None
        with yt_dlp.YoutubeDL(ydl_video_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "video")
            # Find the downloaded file
            for ext in ["mp4", "webm", "mkv"]:
                candidate = output_dir_path / f"{video_id}.{ext}"
                if candidate.exists():
                    downloaded_video = str(candidate)
                    break

        if not downloaded_video:
            logger.warning("yt-dlp: Could not locate downloaded video file")
            return None, None

        # Extract audio as wav for the audio pipeline
        audio_path = str(output_dir_path / f"{video_id}_audio.wav")
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", downloaded_video,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                    audio_path,
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0 or not Path(audio_path).exists():
                # Fallback: use video file directly as audio source
                audio_path = downloaded_video
        except Exception as e:
            logger.warning("yt-dlp: ffmpeg audio extraction failed: %s", e)
            audio_path = downloaded_video

        logger.info("yt-dlp downloaded: video=%s, audio=%s", downloaded_video, audio_path)
        return audio_path, downloaded_video

    except Exception as exc:
        logger.error("yt-dlp download failed: %s\n%s", exc, traceback.format_exc())
        return None, None


# ─── ASCII tab renderer ────────────────────────────────────────────────────────

def _render_ascii_tab(notes: List[Dict], duration_sec: float) -> str:
    """
    Render a 6-line ASCII guitar tab string.
    Time is quantised to 0.1s columns.
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

    grid = [["-"] * n_cols for _ in range(6)]

    for note in notes:
        col     = int(note["onset"] / resolution)
        string  = note.get("string", 0)
        fret    = note.get("fret", 0)
        row     = 5 - string
        if 0 <= col < n_cols and 0 <= row < 6:
            grid[row][col] = str(fret)

    string_labels = ["e", "B", "G", "D", "A", "E"]
    lines = []
    for label, row in zip(string_labels, grid):
        lines.append(f"{label}|{''.join(row)}|")

    return "\n".join(lines)


# ─── Main Celery tasks ─────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.tasks.run_pipeline",
    max_retries=0,
    acks_late=True,
)
def run_pipeline(self: Task, audio_path: str, job_id: str) -> Dict[str, Any]:
    """
    P7 audio-only transcription pipeline (unchanged from P7).

    Steps: Load → Demucs → Basic Pitch → ChordCNN → VoicingLSTM → ASCII tab
    """
    t_start  = time.monotonic()
    project_root = Path(os.getenv("GUITARAI_ROOT", "/app"))

    try:
        _progress(self, "Loading audio", 5)
        y, duration_sec = _load_audio(audio_path)
        logger.info("Audio loaded: %.1f seconds", duration_sec)

        _progress(self, "Running stem separation (Demucs)", 15)
        guitar_path = _run_stem_separation(audio_path)
        stem_applied = guitar_path is not None
        working_path = guitar_path or audio_path
        models_used  = []
        if stem_applied:
            models_used.append("Demucs htdemucs")

        _progress(self, "Transcribing notes (Basic Pitch)", 35)
        raw_notes = _run_basic_pitch(working_path)
        models_used.append("Basic Pitch ONNX")
        logger.info("Basic Pitch found %d notes", len(raw_notes))

        _progress(self, "Detecting chords (ChordCNN)", 55)
        chord_events = _run_chord_cnn(working_path)
        models_used.append("ChordCNN")
        logger.info("ChordCNN found %d chord segments", len(chord_events))

        _progress(self, "Assigning voicings (VoicingLSTM)", 75)
        voiced_notes = _run_voicing_lstm(raw_notes, project_root)
        if voiced_notes and voiced_notes[0].get("voicing_source") == "lstm":
            models_used.append("VoicingLSTM")
        else:
            models_used.append("Greedy heuristic")

        _progress(self, "Rendering tablature", 90)
        tab_str = _render_ascii_tab(voiced_notes, duration_sec)

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
                "has_video":           False,
                "fusion_used":         False,
                "video_source":        None,
            },
        }

        logger.info(
            "Pipeline complete in %.1fs: %d notes, %d chords",
            processing_time, len(voiced_notes), len(chord_events),
        )
        return result

    except Exception as exc:
        logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise

    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass


@celery_app.task(
    bind=True,
    name="app.tasks.run_pipeline_with_video",
    max_retries=0,
    acks_late=True,
)
def run_pipeline_with_video(
    self: Task,
    audio_path: str,
    video_path: str,
    job_id: str,
    video_source: str = "upload",
) -> Dict[str, Any]:
    """
    P13 multimodal pipeline: audio + video → FusionModel transcription.

    Steps (parallel where possible):
      Audio branch:  Load → Demucs → Basic Pitch → ChordCNN
      Vision branch: P9 frames → P10 neck detect → P11 finger tracking
      Fusion:        FusionModel (P12) combines both → voicings
      Output:        ASCII tab
    """
    t_start  = time.monotonic()
    project_root = Path(os.getenv("GUITARAI_ROOT", "/app"))

    try:
        _progress(self, "Loading audio", 5)
        y, duration_sec = _load_audio(audio_path)
        logger.info("Audio loaded: %.1f seconds", duration_sec)

        # Run audio preprocessing + vision pipeline in parallel
        _progress(self, "Running audio + vision pipelines in parallel", 15)

        audio_result = {}
        vision_result_holder = {}

        def _audio_branch():
            guitar_path = _run_stem_separation(audio_path)
            working = guitar_path or audio_path
            raw_notes = _run_basic_pitch(working)
            chord_events = _run_chord_cnn(working)
            return {
                "stem_applied": guitar_path is not None,
                "working_path": working,
                "raw_notes": raw_notes,
                "chord_events": chord_events,
            }

        def _vision_branch():
            return _run_vision_pipeline(video_path, project_root)

        with ThreadPoolExecutor(max_workers=2) as executor:
            audio_future  = executor.submit(_audio_branch)
            vision_future = executor.submit(_vision_branch)

            # Update progress while waiting
            _progress(self, "Demucs stem separation + video frame extraction", 25)
            audio_data    = audio_future.result()
            _progress(self, "Basic Pitch + finger tracking", 45)
            video_features = vision_future.result()

        stem_applied  = audio_data["stem_applied"]
        raw_notes     = audio_data["raw_notes"]
        chord_events  = audio_data["chord_events"]
        models_used   = []
        if stem_applied:
            models_used.append("Demucs htdemucs")
        models_used.append("Basic Pitch ONNX")
        models_used.append("ChordCNN")

        logger.info(
            "Audio branch: %d notes, %d chords | Vision branch: %s detections",
            len(raw_notes),
            len(chord_events),
            len(video_features) if video_features else "None",
        )

        # Fusion or fallback
        _progress(self, "Fusing audio + video (FusionModel)", 70)
        fusion_used = False

        if video_features and len(video_features) > 0:
            voiced_notes = _run_fusion_model(raw_notes, video_features, project_root)
            if voiced_notes and voiced_notes[0].get("voicing_source") == "fusion":
                models_used.append("FusionModel (P12)")
                fusion_used = True
            elif voiced_notes and voiced_notes[0].get("voicing_source") == "lstm":
                models_used.append("VoicingLSTM (fallback)")
            else:
                models_used.append("Greedy heuristic (fallback)")
        else:
            logger.info("No video features — routing through LSTM voicing")
            voiced_notes = _run_voicing_lstm(raw_notes, project_root)
            if voiced_notes and voiced_notes[0].get("voicing_source") == "lstm":
                models_used.append("VoicingLSTM")
            else:
                models_used.append("Greedy heuristic")

        _progress(self, "Rendering tablature", 90)
        tab_str = _render_ascii_tab(voiced_notes, duration_sec)

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
                "has_video":           True,
                "fusion_used":         fusion_used,
                "video_source":        video_source,
            },
        }

        logger.info(
            "Multimodal pipeline complete in %.1fs: %d notes, %d chords, fusion=%s",
            processing_time, len(voiced_notes), len(chord_events), fusion_used,
        )
        return result

    except Exception as exc:
        logger.error("Multimodal pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise

    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass
        # Keep video for debugging; remove temp YouTube downloads if needed


@celery_app.task(
    bind=True,
    name="app.tasks.run_pipeline_from_url",
    max_retries=0,
    acks_late=True,
    soft_time_limit=480,  # 8 minutes (yt-dlp download + full pipeline)
)
def run_pipeline_from_url(self: Task, url: str, job_id: str) -> Dict[str, Any]:
    """
    P13 YouTube pipeline: download via yt-dlp then run multimodal transcription.
    """
    t_start = time.monotonic()
    project_root = Path(os.getenv("GUITARAI_ROOT", "/app"))

    # Use a temp directory for YouTube downloads
    dl_dir = project_root / "outputs" / "youtube_downloads" / job_id
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        _progress(self, "Downloading video (yt-dlp)", 5)
        audio_path, video_path = _download_youtube(url, str(dl_dir))

        if not audio_path:
            raise RuntimeError(f"yt-dlp failed to download: {url}")

        if not video_path or audio_path == video_path:
            # Only audio downloaded — run audio-only pipeline
            logger.info("Only audio available from yt-dlp — running audio-only pipeline")
            _progress(self, "Running audio pipeline (no video)", 20)
            # Delegate to the audio-only task logic inline
            y, duration_sec = _load_audio(audio_path)
            guitar_path = _run_stem_separation(audio_path)
            working = guitar_path or audio_path
            raw_notes = _run_basic_pitch(working)
            _progress(self, "Detecting chords", 55)
            chord_events = _run_chord_cnn(working)
            _progress(self, "Assigning voicings", 75)
            voiced_notes = _run_voicing_lstm(raw_notes, project_root)
            models_used = ["Basic Pitch ONNX", "ChordCNN"]
            models_used.append("VoicingLSTM" if voiced_notes and voiced_notes[0].get("voicing_source") == "lstm" else "Greedy heuristic")
            _progress(self, "Rendering tablature", 90)
            tab_str = _render_ascii_tab(voiced_notes, duration_sec)
            processing_time = time.monotonic() - t_start
            return {
                "job_id": job_id,
                "chords": chord_events,
                "tab": tab_str,
                "notes": voiced_notes,
                "pipeline": {
                    "stem_separation": guitar_path is not None,
                    "models_used": models_used,
                    "audio_duration_sec": round(duration_sec, 2),
                    "processing_time_sec": round(processing_time, 2),
                    "note_count": len(voiced_notes),
                    "chord_count": len(chord_events),
                    "has_video": False,
                    "fusion_used": False,
                    "video_source": "youtube",
                },
            }

        # Full multimodal pipeline with downloaded video
        _progress(self, "Video downloaded — running audio + vision pipelines", 20)
        y, duration_sec = _load_audio(audio_path)

        def _audio_branch():
            guitar_path = _run_stem_separation(audio_path)
            working = guitar_path or audio_path
            return {
                "stem_applied": guitar_path is not None,
                "working_path": working,
                "raw_notes": _run_basic_pitch(working),
                "chord_events": _run_chord_cnn(working),
            }

        def _vision_branch():
            return _run_vision_pipeline(video_path, project_root)

        _progress(self, "Processing audio + video in parallel", 30)
        with ThreadPoolExecutor(max_workers=2) as executor:
            audio_future  = executor.submit(_audio_branch)
            vision_future = executor.submit(_vision_branch)
            audio_data    = audio_future.result()
            video_features = vision_future.result()

        stem_applied  = audio_data["stem_applied"]
        raw_notes     = audio_data["raw_notes"]
        chord_events  = audio_data["chord_events"]
        models_used   = []
        if stem_applied:
            models_used.append("Demucs htdemucs")
        models_used.extend(["Basic Pitch ONNX", "ChordCNN"])

        _progress(self, "Fusing audio + video (FusionModel)", 70)
        fusion_used = False
        if video_features:
            voiced_notes = _run_fusion_model(raw_notes, video_features, project_root)
            if voiced_notes and voiced_notes[0].get("voicing_source") == "fusion":
                models_used.append("FusionModel (P12)")
                fusion_used = True
            else:
                models_used.append("VoicingLSTM (fallback)")
        else:
            voiced_notes = _run_voicing_lstm(raw_notes, project_root)
            models_used.append("VoicingLSTM")

        _progress(self, "Rendering tablature", 90)
        tab_str = _render_ascii_tab(voiced_notes, duration_sec)
        processing_time = time.monotonic() - t_start

        return {
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
                "has_video":           True,
                "fusion_used":         fusion_used,
                "video_source":        "youtube",
            },
        }

    except Exception as exc:
        logger.error("YouTube pipeline failed: %s\n%s", exc, traceback.format_exc())
        raise
