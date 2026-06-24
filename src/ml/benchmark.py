"""
benchmarker.py
--------------
P5: Transcription Benchmarker

Compares three systems on the GuitarSet test split:
  1. Chroma Baseline   — simple chromagram peak → chord quality (no ML)
  2. Basic Pitch       — Spotify's audio-to-MIDI model
  3. ChordCNN          — our trained model from P4

Outputs:
  • Console comparison table
  • outputs/benchmark_results.json
  • outputs/benchmark_results.png  (bar chart)

Run:  python -m src.ml.benchmarker
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import librosa
import mir_eval
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Local imports ──────────────────────────────────────────
import torch
from src.ml.models import load_model
from src.ml.train_chord import ChordDataset

# ── Optional: Basic Pitch ──────────────────────────────────
try:
    import os as _os
    import basic_pitch as _bp_pkg
    from basic_pitch.inference import predict as bp_predict

    # Locate the ONNX model file explicitly.
    # basic-pitch 0.4.0 bundles nmp.onnx next to the TF SavedModel directory.
    # We must pass this path directly to predict() — if we let it default to
    # ICASSP_2022_MODEL_PATH it resolves to the TF *directory* (nmp/) which
    # fails on TF 2.16+ with AttributeError('add_slot').
    _bp_models_dir = _os.path.join(
        _os.path.dirname(_bp_pkg.__file__),
        "saved_models", "icassp_2022",
    )
    _BP_MODEL_PATH = _os.path.join(_bp_models_dir, "nmp.onnx")

    if not _os.path.exists(_BP_MODEL_PATH):
        # List what is actually available so the error is actionable
        _available = _os.listdir(_bp_models_dir) if _os.path.isdir(_bp_models_dir) else []
        raise FileNotFoundError(
            f"nmp.onnx not found in {_bp_models_dir}\n"
            f"  Available files: {_available}\n"
            f"  Fix: pip install --force-reinstall 'basic-pitch==0.4.0'"
        )

    HAS_BASIC_PITCH = True
except ImportError:
    HAS_BASIC_PITCH = False
    print("ℹ️  basic-pitch not installed — skipping that model.")
    print("   pip install 'basic-pitch==0.4.0'")
except FileNotFoundError as _e:
    HAS_BASIC_PITCH = False
    print(f"⚠️  Basic Pitch ONNX model missing — skipping.\n   {_e}")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SR              = 22050
MODEL_PATH      = "models/chord_cnn.pth"
LABEL_MAP_FILE  = Path("data/processed/chord_dataset/label_map.json")
SPLITS_FILE     = Path("data/splits/splits_guitarset.json")
ANNOT_DIR       = Path("data/raw/guitarset/annotation")
OUT_DIR         = Path("outputs")

# mir_eval tolerance: a note is "correct" if onset within 50ms and same pitch
ONSET_TOLERANCE = 0.05   # seconds
PITCH_TOLERANCE = 0.5    # semitones (0.5 = must be exact semitone)

# Number of test files to evaluate (full test set is 60; cap for speed)
MAX_FILES       = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────
# GUITARSET ANNOTATION LOADER
# ─────────────────────────────────────────────────────────
def load_guitarset_ground_truth(wav_path: str) -> dict:
    """
    Loads ground truth from GuitarSet JAMS annotation.
    Returns:
      notes:  list of (onset_sec, offset_sec, midi_pitch)
      chords: list of (onset_sec, offset_sec, chord_label)
    """
    import jams
    stem      = Path(wav_path).stem
    # Strip audio-type suffix (_mic, _hex_cln, etc.)
    for suf in ["_mic", "_hex_cln", "_hex_orig", "_mix", "_hex"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    jams_path = ANNOT_DIR / f"{stem}.jams"
    if not jams_path.exists():
        return {"notes": [], "chords": []}

    jam = jams.load(str(jams_path))

    # ── Chord ground truth ────────────────────────────────
    chord_annots = jam.search(namespace="chord")
    chords = []
    if chord_annots:
        for interval, value in zip(*chord_annots[0].to_interval_values()):
            chords.append((float(interval[0]), float(interval[1]), str(value)))

    # ── Note ground truth (from note_midi namespace) ──────
    note_annots = jam.search(namespace="note_midi")
    notes = []
    if note_annots:
        for interval, value in zip(*note_annots[0].to_interval_values()):
            midi_pitch = int(round(float(value)))
            notes.append((float(interval[0]), float(interval[1]), midi_pitch))

    return {"notes": notes, "chords": chords}


# ─────────────────────────────────────────────────────────
# EVALUATION METRICS  (via mir_eval)
# ─────────────────────────────────────────────────────────
def evaluate_notes(
    ref_notes: list[tuple],
    est_notes: list[tuple],
) -> dict[str, float]:
    """
    Note-level F1 using mir_eval.
    ref/est_notes: list of (onset_sec, offset_sec, midi_pitch)

    Returns precision, recall, f1 (all in [0, 1]).
    """
    if not ref_notes or not est_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_intervals = np.array([(n[0], n[1]) for n in ref_notes])
    ref_pitches   = np.array([n[2] for n in ref_notes], dtype=float)
    est_intervals = np.array([(n[0], n[1]) for n in est_notes])
    est_pitches   = np.array([n[2] for n in est_notes], dtype=float)

    # mir_eval wants Hz for pitch comparison in some modes;
    # we use the midi mode via precision/recall/f_measure directly
    # with a semitone tolerance check
    try:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance  = ONSET_TOLERANCE,
            pitch_tolerance  = PITCH_TOLERANCE,
            offset_ratio     = None,   # ignore offset, only match onset+pitch
        )
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
    }


def _normalise_chord_label(label: str) -> str:
    """
    Normalise a chord label to mir_eval's expected colon format.

    mir_eval understands: "C:maj", "G:min", "A:7", "N" (no chord), "X" (unknown).
    GuitarSet JAMS already uses this format.
    Our ChordCNN may have been trained on shorthand like "C", "Gm", "silence".
    This maps common shorthand → mir_eval format so scores are fair.
    """
    label = label.strip()
    if label.lower() in ("n", "none", "silence", "no chord", "nc", "rest", ""):
        return "N"
    if ":" in label:
        return label                    # already mir_eval format
    if label.endswith("m") and len(label) >= 2 and not label[-2].isdigit():
        return f"{label[:-1]}:min"      # "Gm" → "G:min"
    if label.endswith("7") and len(label) <= 3:
        return f"{label[:-1]}:7"        # "G7" → "G:7"
    return f"{label}:maj"               # bare root → major


def evaluate_chords(
    ref_chords: list[tuple],
    est_chords: list[tuple],
    duration:   float,
) -> dict[str, float]:
    """
    Framewise chord accuracy using mir_eval.
    ref/est_chords: list of (start_sec, end_sec, chord_label)

    Labels are normalised before scoring so shorthand CNN labels
    ("C", "Gm") are compared fairly against GuitarSet colon-format
    ("C:maj", "G:min").

    Returns overlap score in [0, 1].
    """
    if not ref_chords or not est_chords:
        return {"chord_acc": 0.0}

    ref_intervals = np.array([(c[0], c[1]) for c in ref_chords])
    ref_labels    = [_normalise_chord_label(c[2]) for c in ref_chords]
    est_intervals = np.array([(c[0], c[1]) for c in est_chords])
    est_labels    = [_normalise_chord_label(c[2]) for c in est_chords]

    try:
        score = mir_eval.chord.weighted_accuracy(
            *mir_eval.chord.encode_many(ref_labels, ref_intervals),
            *mir_eval.chord.encode_many(est_labels, est_intervals),
        )
    except Exception:
        # Fall back to simple frame overlap
        try:
            ref_i, ref_l = mir_eval.util.intervals_to_samples(
                ref_intervals, ref_labels, sample_size=0.01)
            est_i, est_l = mir_eval.util.intervals_to_samples(
                est_intervals, est_labels, sample_size=0.01)
            min_len = min(len(ref_l), len(est_l))
            score = sum(r == e for r, e in zip(ref_l[:min_len], est_l[:min_len])) / max(min_len, 1)
        except Exception:
            score = 0.0

    return {"chord_acc": round(float(score), 4)}


# ─────────────────────────────────────────────────────────
# MODEL WRAPPERS  (unified interface)
# ─────────────────────────────────────────────────────────
# All wrappers take (audio: np.ndarray, sr: int)
# and return {"notes": [...], "chords": [...]}

class ChromaBaseline:
    """
    Purely signal-based baseline. No ML.
    Extracts chromagram, finds dominant pitch class per frame,
    then segments into constant-pitch regions.

    This is the "floor" — any ML model should beat this.
    """
    name = "Chroma Baseline"

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        hop    = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)
        # Dominant pitch class per frame
        pitch_class = np.argmax(chroma, axis=0)   # (T,)
        times       = librosa.frames_to_time(np.arange(len(pitch_class)),
                                              sr=sr, hop_length=hop)

        # Convert pitch class to MIDI (use middle octave C4=60)
        MIDI_OFFSET = 60   # C4
        midi_seq    = pitch_class + MIDI_OFFSET

        # Segment: group consecutive identical pitches into note events
        notes = []
        if len(midi_seq) > 0:
            onset = 0
            cur   = midi_seq[0]
            for i in range(1, len(midi_seq)):
                if midi_seq[i] != cur:
                    dur = times[i] - times[onset]
                    if dur >= 0.05:   # min 50ms
                        notes.append((times[onset], times[i], int(cur)))
                    onset = i
                    cur   = midi_seq[i]
            # Last segment
            notes.append((times[onset], times[-1], int(cur)))

        # Chord = just report the most common pitch class as "X:quality"
        # (placeholder — chroma baseline has no real chord model)
        dominant = int(np.bincount(pitch_class).argmax())
        NAMES    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        chords   = [(0.0, times[-1] if len(times) else 1.0,
                     f"{NAMES[dominant]}:maj")]

        return {"notes": notes, "chords": chords}


class BasicPitchWrapper:
    """
    Wrapper around Spotify's Basic Pitch.
    Returns per-note MIDI events.
    No chord output (Basic Pitch is a monophonic/polyphonic note transcriber,
    not a chord recogniser).
    """
    name = "Basic Pitch"

    def __init__(self):
        if not HAS_BASIC_PITCH:
            raise RuntimeError("basic-pitch not installed")

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        import tempfile, soundfile as sf
        # Basic Pitch wants a file path or audio array + sr
        # Write to a temp wav for reliability
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, audio, sr)

        try:
            model_output, midi_data, note_events = bp_predict(
                tmp,
                _BP_MODEL_PATH,
                onset_threshold    = 0.5,
                frame_threshold    = 0.3,
                minimum_note_length= 58,    # ms
                minimum_frequency  = 82.41, # E2
                maximum_frequency  = 2000,  # well above top fret
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

        # note_events: list of (start_sec, end_sec, pitch_midi, amplitude, ...)
        notes = [(float(n[0]), float(n[1]), int(n[2])) for n in note_events]
        return {"notes": notes, "chords": []}


class ChordCNNWrapper:
    """
    Wrapper around our trained ChordCNN from P4.
    Outputs chord events (no per-note transcription).
    """
    name = "ChordCNN (ours)"

    def __init__(self, model_path: str, label_map: dict):
        self.model     = load_model(model_path).to(DEVICE)
        self.model.eval()
        self.label_map = label_map
        self.inv_map   = {v: k for k, v in label_map.items()}

        # CQT settings must match preprocess_chord.py
        self.hop       = 256
        self.n_bins    = 84
        self.fmin      = librosa.note_to_hz("E2")
        self.win_fr    = 87    # T_FRAMES

    def _audio_to_cqt(self, audio: np.ndarray, sr: int) -> np.ndarray:
        y    = librosa.effects.preemphasis(audio, coef=0.97)
        peak = np.max(np.abs(y))
        if peak > 1e-6:
            y = y / peak
        C = librosa.cqt(y, sr=sr, hop_length=self.hop,
                        fmin=self.fmin, n_bins=self.n_bins, bins_per_octave=12)
        C_mag = np.abs(C)
        c_max = C_mag.max()
        if c_max > 1e-6:
            C_mag = C_mag / c_max
        return C_mag

    def predict(self, audio: np.ndarray, sr: int) -> dict:
        cqt      = self._audio_to_cqt(audio, sr)   # (84, T)
        total_fr = cqt.shape[1]
        stride   = self.win_fr // 2   # 50% overlap

        chords   = []
        start_fr = 0

        while start_fr + self.win_fr <= total_fr:
            window   = cqt[:, start_fr:start_fr + self.win_fr]
            x        = torch.from_numpy(window[np.newaxis, np.newaxis]).float().to(DEVICE)

            with torch.no_grad():
                logits = self.model(x)
                pred   = int(logits.argmax(dim=-1).item())

            label      = self.inv_map.get(pred, "N")
            onset_t    = start_fr * self.hop / sr
            offset_t   = (start_fr + self.win_fr) * self.hop / sr
            chords.append((onset_t, offset_t, label))
            start_fr  += stride

        # Merge consecutive identical chords
        merged = []
        for c in chords:
            if merged and merged[-1][2] == c[2]:
                merged[-1] = (merged[-1][0], c[1], c[2])
            else:
                merged.append(list(c))

        return {"notes": [], "chords": [(c[0], c[1], c[2]) for c in merged]}


# ─────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────
def load_test_files() -> list[str]:
    """Loads GuitarSet test split audio paths."""
    with open(SPLITS_FILE) as f:
        raw = json.load(f)

    test_items = raw.get("test", [])
    paths = []
    for item in test_items:
        if isinstance(item, dict):
            paths.append(item.get("audio", ""))
        elif isinstance(item, str):
            paths.append(item)
    return [p for p in paths if p and Path(p).exists()]


def run_benchmark():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GuitarAI — P5 Transcription Benchmarker")
    print(f"  Device : {DEVICE.upper()}")
    print("=" * 60)

    # ── Load label map ────────────────────────────────────
    with open(LABEL_MAP_FILE) as f:
        label_map = json.load(f)

    # ── Instantiate models ────────────────────────────────
    models = [ChromaBaseline()]

    if HAS_BASIC_PITCH:
        try:
            models.append(BasicPitchWrapper())
            print(f"✅ Basic Pitch loaded  ← ONNX @ {_BP_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Basic Pitch failed to load: {e}")
    else:
        print("⚠️  Basic Pitch skipped — see error above for fix.")

    try:
        models.append(ChordCNNWrapper(MODEL_PATH, label_map))
        print("✅ ChordCNN loaded")
    except Exception as e:
        print(f"⚠️  ChordCNN failed to load: {e}")

    # ── Load test files ───────────────────────────────────
    test_files = load_test_files()[:MAX_FILES]
    print(f"\n📂 Evaluating on {len(test_files)} test files\n")

    if not test_files:
        print("❌ No test files found. Check SPLITS_FILE path.")
        return

    # ── Results accumulator ───────────────────────────────
    # {model_name: {metric: [values]}}
    results = {m.name: {"note_f1": [], "note_precision": [],
                         "note_recall": [], "chord_acc": [],
                         "inference_ms": []}
               for m in models}

    _diag_done = False  # label diagnostic: print once then stop
    # ── Main eval loop ────────────────────────────────────
    for wav_path in tqdm(test_files, desc="Files"):
        try:
            audio, sr = librosa.load(wav_path, sr=SR, mono=True)
        except Exception as e:
            print(f"\n  ⚠️  Could not load {Path(wav_path).name}: {e}")
            continue

        gt = load_guitarset_ground_truth(wav_path)
        if not gt["notes"] and not gt["chords"]:
            continue

        duration = len(audio) / sr

        for model in models:
            t0 = time.perf_counter()
            try:
                output = model.predict(audio, sr)
            except Exception as e:
                print(f"\n  ⚠️  {model.name} failed on {Path(wav_path).name}: {e}")
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # One-time label diagnostic so you can verify vocab alignment
            if not _diag_done and output.get("chords") and gt.get("chords"):
                tqdm.write("\n── Label diagnostic (first chord-producing file) ──")
                tqdm.write(f"  GT labels  : {[c[2] for c in gt['chords'][:4]]}")
                tqdm.write(f"  {model.name:<12}: {[c[2] for c in output['chords'][:4]]}")
                tqdm.write("────────────────────────────────────────────────────\n")
                _diag_done = True

            # Note metrics (only if model produces notes)
            if output["notes"] and gt["notes"]:
                nm = evaluate_notes(gt["notes"], output["notes"])
                results[model.name]["note_f1"].append(nm["f1"])
                results[model.name]["note_precision"].append(nm["precision"])
                results[model.name]["note_recall"].append(nm["recall"])

            # Chord metrics (only if model produces chords)
            if output["chords"] and gt["chords"]:
                cm = evaluate_chords(gt["chords"], output["chords"], duration)
                results[model.name]["chord_acc"].append(cm["chord_acc"])

            results[model.name]["inference_ms"].append(elapsed_ms)

    # ── Aggregate ─────────────────────────────────────────
    def mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {
            "note_f1":        mean(metrics["note_f1"]),
            "note_precision": mean(metrics["note_precision"]),
            "note_recall":    mean(metrics["note_recall"]),
            "chord_acc":      mean(metrics["chord_acc"]),
            "inference_ms":   mean(metrics["inference_ms"]),
            "n_files":        len(metrics["inference_ms"]),
        }

    # ── Print table ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Model':<22} {'Note F1':>9} {'Chord Acc':>10} {'Inf (ms)':>10}")
    print("  " + "─" * 66)
    for name, s in summary.items():
        nf1  = f"{s['note_f1']:.4f}"  if s['note_f1']  is not None else "  N/A  "
        cacc = f"{s['chord_acc']:.4f}" if s['chord_acc'] is not None else "  N/A  "
        inf  = f"{s['inference_ms']:.1f}" if s['inference_ms'] is not None else "N/A"
        print(f"  {name:<22} {nf1:>9} {cacc:>10} {inf:>10}")
    print("=" * 70)

    # ── Save JSON ─────────────────────────────────────────
    out_json = OUT_DIR / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved → {out_json}")

    # ── Bar chart ─────────────────────────────────────────
    _plot_results(summary)


def _plot_results(summary: dict):
    """Saves a simple bar chart comparing chord accuracy across models."""
    try:
        import matplotlib.pyplot as plt

        models_with_chord = {k: v for k, v in summary.items()
                             if v["chord_acc"] is not None}
        if not models_with_chord:
            return

        names  = list(models_with_chord.keys())
        accs   = [models_with_chord[n]["chord_acc"] for n in names]
        colors = ["#4C72B0", "#DD8452", "#55A868"][:len(names)]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Chord Accuracy (mir_eval)")
        ax.set_title("P5 Benchmark — Chord Recognition Accuracy")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        out_png = OUT_DIR / "benchmark_results.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"📊 Chart saved → {out_png}")

    except Exception as e:
        print(f"  ℹ️  Chart skipped: {e}")


if __name__ == "__main__":
    run_benchmark()