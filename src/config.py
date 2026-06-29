"""
GuitarAI — Shared Configuration
================================
Central configuration for all modules. All paths are derived from
PROJECT_ROOT so that nothing is hardcoded to a single machine.

Usage:
    from src.config import PROJECT_ROOT, DATA_RAW_DIR, SR
"""

import math
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────
# Automatically resolves to the GuitarAI root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR           = PROJECT_ROOT / "data"
DATA_RAW_DIR       = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SPLITS_DIR    = DATA_DIR / "splits"

OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
MODELS_DIR         = PROJECT_ROOT / "models"

# ── Aliases used by preprocess_chord.py and train_chord.py ──
DATA_ROOT     = DATA_DIR           # old name → same object
SPLITS_DIR    = DATA_SPLITS_DIR    # old name → same object
PROCESSED_DIR = DATA_PROCESSED_DIR # old name → same object

# ─────────────────────────────────────────────────────────
# DATASET ROOTS
# ─────────────────────────────────────────────────────────
# Expected layout after unzipping:
#   data/raw/guitarset/          ← GuitarSet (annotation.zip, audio_*.zip)
#   data/raw/guitar_techs/       ← Guitar-TECHS (P1_*.zip, P2_*.zip, P3_*.zip)
#   data/raw/idmt_guitar/        ← IDMT-SMT-Guitar (IDMT-SMT-GUITAR_V2.zip)

DATASET_ROOTS = {
    "guitarset":    DATA_RAW_DIR / "guitarset",
    "guitar_techs": DATA_RAW_DIR / "guitar_techs",
    "idmt":         DATA_RAW_DIR / "idmt_guitar",
}

# GuitarSet subdirectories (expected after unzip)
GUITARSET_AUDIO_MIC   = DATASET_ROOTS["guitarset"] / "audio_mono-mic"
GUITARSET_AUDIO_HEX   = DATASET_ROOTS["guitarset"] / "audio_hex-pickup_debleeded"
GUITARSET_ANNOTATIONS = DATASET_ROOTS["guitarset"] / "annotation"

# ─────────────────────────────────────────────────────────
# AUDIO SETTINGS
# ─────────────────────────────────────────────────────────
SR               = 22050    # Canonical sample rate for all feature extraction
SR_RAW           = 44100    # Native sample rate of all three datasets
DURATION_PREVIEW = 10       # Seconds for visualization previews

# CQT settings (guitar-optimized)
# Note: two hop-length values exist because dataset_lab.py (P3) used 512
# and preprocess_chord.py (P4) uses 256 for finer time resolution.
# Use CQT_HOP_LENGTH for P3 visualizations, HOP_LENGTH for P4 training.
CQT_HOP_LENGTH   = 512     # P3 dataset explorer
HOP_LENGTH       = 256     # P4 feature extraction  (~11.6 ms/frame)

CQT_FMIN         = "E2"   # Lowest guitar note (~82 Hz) — string form for librosa
FMIN_HZ          = 82.41  # Same value as float for direct use
CQT_N_BINS       = 84     # 7 octaves (E2–E9)
CQT_BINS_PER_OCT = 12

# Aliases so both naming conventions resolve without import errors
N_BINS          = CQT_N_BINS
BINS_PER_OCTAVE = CQT_BINS_PER_OCT

# Sliding window (P4 preprocessing)
WINDOW_SEC  = 1.0   # window length in seconds
OVERLAP_SEC = 0.5   # stride = WINDOW_SEC - OVERLAP_SEC

# Derived: fixed frame count per 1-second CQT window
T_FRAMES = math.ceil(SR * WINDOW_SEC / HOP_LENGTH)   # = 87

# ─────────────────────────────────────────────────────────
# GUITARSET PLAYER IDS (for leakage-free splits)
# ─────────────────────────────────────────────────────────
# GuitarSet recordings are named like: "00_BN1-129-Eb_comp_..."
# The first two digits are the player ID (00–05 → 6 players)
GUITARSET_PLAYERS = ["00", "01", "02", "03", "04", "05"]

# Default split: players 00-03 train, 04 val, 05 test
GUITARSET_SPLIT_PLAYERS = {
    "train": ["00", "01", "02", "03"],
    "val":   ["04"],
    "test":  ["05"],
}

# ─────────────────────────────────────────────────────────
# GUITAR-TECHS PLAYER IDS
# ─────────────────────────────────────────────────────────
GUITAR_TECHS_PLAYERS = ["P1", "P2", "P3"]

# Default split: P1+P2 train, P3 test (P3 only has music excerpts)
GUITAR_TECHS_SPLIT_PLAYERS = {
    "train": ["P1"],
    "val":   ["P2"],
    "test":  ["P3"],
}

# ─────────────────────────────────────────────────────────
# TRAINING (P4+)
# ─────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
PATIENCE     = 8        # early stopping patience (epochs)

# ─────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ─────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────
MODEL_PATH    = MODELS_DIR / "chord_cnn.pth"
TRAINING_LOG  = MODELS_DIR / "training_log.json"
EVAL_REPORT   = OUTPUTS_DIR / "eval_report.json"
CONFUSION_MAT = OUTPUTS_DIR / "confusion_matrix.png"

# Chord dataset paths (P4)
LABEL_MAP_FILE = PROCESSED_DIR / "chord_dataset" / "label_map.json"

# ─────────────────────────────────────────────────────────
# P10: GUITAR VISION MODEL
# ─────────────────────────────────────────────────────────
# Dataset paths
VISION_DATASET_DIR  = DATA_PROCESSED_DIR / "vision_dataset"
NECK_DATASET_DIR    = VISION_DATASET_DIR / "neck"
CHORD_DATASET_DIR   = VISION_DATASET_DIR / "chords"

# Model checkpoints
NECK_MODEL_PATH     = MODELS_DIR / "neck_detector.pt"
CHORD_SHAPE_MODEL   = MODELS_DIR / "chord_shape_cnn.pth"

# Chord shape classes (6 common open chords + 'none' for empty/unclear frames)
CHORD_SHAPE_CLASSES = ["C", "Am", "G", "Em", "D", "F", "none"]
NUM_CHORD_SHAPES    = len(CHORD_SHAPE_CLASSES)   # 7

# Fretboard warp dimensions (must match P9 warp_fretboard.py)
WARP_W = 600
WARP_H = 200

# Chord CNN input dimensions (resized from WARP_W × WARP_H)
CHORD_INPUT_W = 200
CHORD_INPUT_H = 64

# P10 training hyperparameters
P10_BATCH_SIZE = 32
P10_EPOCHS     = 50
P10_LR         = 1e-3
P10_PATIENCE   = 8
P10_DROPOUT    = 0.4

# ─────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────
NO_CHORD = "N"   # label for silence / no active chord

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required output directories if they don't exist."""
    for d in [
        DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
        OUTPUTS_DIR, MODELS_DIR,
        OUTPUTS_DIR / "explorer",
        OUTPUTS_DIR / "splitter",
        OUTPUTS_DIR / "viz",
        PROCESSED_DIR / "chord_dataset",
        # P10: Vision dataset directories
        VISION_DATASET_DIR,
        NECK_DATASET_DIR,
        CHORD_DATASET_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def verify_datasets():
    """Check which datasets are available and report status."""
    print("─── 📁 Dataset Verification ───")
    for name, root in DATASET_ROOTS.items():
        if root.exists():
            wav_count = len(list(root.rglob("*.wav")))
            print(f"  ✅ {name:15s} → {root}  ({wav_count} WAV files)")
        else:
            print(f"  ❌ {name:15s} → {root}  (NOT FOUND)")

    # GuitarSet specifics
    for label, path in [
        ("annotations",    GUITARSET_ANNOTATIONS),
        ("audio_mono-mic", GUITARSET_AUDIO_MIC),
        ("audio_hex",      GUITARSET_AUDIO_HEX),
    ]:
        status = "✅" if path.exists() else "❌"
        count  = len(list(path.iterdir())) if path.exists() else 0
        suffix = f"{count} files" if path.exists() else "NOT FOUND"
        print(f"     └─ {status} {label:20s}: {suffix}")


if __name__ == "__main__":
    ensure_dirs()
    verify_datasets()

    print("\n─── ⚙️  Audio Settings ───")
    print(f"  SR            : {SR} Hz")
    print(f"  HOP_LENGTH    : {HOP_LENGTH}  ({1000 * HOP_LENGTH / SR:.1f} ms/frame)")
    print(f"  T_FRAMES      : {T_FRAMES}  (frames per 1-sec window)")
    print(f"  N_BINS        : {N_BINS}")
    print(f"  DEVICE        : {DEVICE.upper()}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print(f"  EPOCHS        : {EPOCHS}")