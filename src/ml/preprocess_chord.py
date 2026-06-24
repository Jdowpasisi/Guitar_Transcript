"""
preprocess_chord.py
-------------------
Converts raw audio + annotations from all three datasets into
fixed-size CQT windows (.npy) + a unified label map (JSON).

Outputs  →  data/processed/chord_dataset/
              ├── X_train.npy   shape: (N, 1, 84, 87)
              ├── y_train.npy   shape: (N,)
              ├── X_val.npy
              ├── y_val.npy
              ├── X_test.npy
              ├── y_test.npy
              └── label_map.json   {"N": 0, "C:maj": 1, ...}

Run:  python -m src.ml.preprocess_chord
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob

import numpy as np
import librosa
import jams
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIG  (mirrors src/config.py values)
# ─────────────────────────────────────────────────────────
SR           = 22050
HOP_LENGTH   = 256          # ~11.6 ms per frame
WINDOW_SEC   = 1.0          # 1-second CQT windows
OVERLAP_SEC  = 0.5          # stride = 0.5 s  →  2× data via overlap
N_BINS       = 84           # 7 octaves (E2–E9), 12 bins/octave
FMIN         = librosa.note_to_hz("E2")   # ~82 Hz, lowest guitar note

# Fixed time-frame count for a 1-second window at these settings
#   frames = ceil(SR * WINDOW_SEC / HOP_LENGTH) = ceil(22050/256) = 87
T_FRAMES     = int(np.ceil(SR * WINDOW_SEC / HOP_LENGTH))  # 87

DATASET_ROOTS = {
    "guitarset":    Path("data/raw/guitarset"),
    "guitar_techs": Path("data/raw/guitar_techs"),
    "idmt":         Path("data/raw/idmt_guitar"),
}

SPLITS_FILE  = Path("data/splits/splits_guitarset.json")
OUT_DIR      = Path("data/processed/chord_dataset")
LABEL_MAP_F  = OUT_DIR / "label_map.json"

NO_CHORD     = "N"          # label used when no chord is active


# ─────────────────────────────────────────────────────────
# AUDIO CLEANING + CQT
# ─────────────────────────────────────────────────────────
def load_and_clean(path: str, sr: int = SR) -> np.ndarray:
    """
    Loads mono audio with three pro-level cleaning steps:
      1. High-Pass Filter  — removes sub-80 Hz mud/rumble
      2. Peak normalization — prevents quiet/loud confusion
      3. (De-reverb via pre-emphasis is baked into step 1)
    """
    y, _ = librosa.load(path, sr=sr, mono=True)

    # 1. High-pass filter: pre-emphasis approximates HPF at ~85 Hz
    y = librosa.effects.preemphasis(y, coef=0.97)

    # 2. Peak normalization
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak

    return y


def compute_cqt(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Returns magnitude CQT, shape (N_BINS, T), values in [0, 1].
    CQT is preferred over mel for guitar — finer low-freq resolution
    and pitch shifts = simple bin translations.
    """
    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=12,
    )
    C_mag = np.abs(C)

    # Normalize to [0, 1] per clip
    c_max = C_mag.max()
    if c_max > 1e-6:
        C_mag = C_mag / c_max
    return C_mag


# ─────────────────────────────────────────────────────────
# WINDOWING
# ─────────────────────────────────────────────────────────
def slice_windows(
    cqt: np.ndarray,
    chord_events: list[tuple[float, float, str]],
    window_sec: float = WINDOW_SEC,
    overlap_sec: float = OVERLAP_SEC,
    sr: int = SR,
) -> list[tuple[np.ndarray, str]]:
    """
    Sliding-window segmentation of a CQT spectrogram.

    chord_events: list of (start_sec, end_sec, chord_label)

    For each window the label is the chord active at the *centre*
    timestamp — robust to boundary ambiguity.

    Returns a list of (window_array[84, T_FRAMES], label_str) pairs.
    """
    stride_sec  = window_sec - overlap_sec
    stride_fr   = int(stride_sec * sr / HOP_LENGTH)
    window_fr   = T_FRAMES
    total_fr    = cqt.shape[1]

    samples = []
    start_fr = 0

    while start_fr + window_fr <= total_fr:
        end_fr    = start_fr + window_fr
        centre_t  = (start_fr + window_fr / 2) * HOP_LENGTH / sr

        # Look up chord at centre timestamp
        label = NO_CHORD
        for ev_start, ev_end, ev_label in chord_events:
            if ev_start <= centre_t < ev_end:
                label = ev_label
                break

        window = cqt[:, start_fr:end_fr]          # (84, T_FRAMES)
        samples.append((window, label))
        start_fr += stride_fr

    return samples


# ─────────────────────────────────────────────────────────
# ANNOTATION PARSERS
# ─────────────────────────────────────────────────────────
def parse_jams_chords(jams_path: str) -> list[tuple[float, float, str]]:
    """GuitarSet: reads chord namespace from a .jams file."""
    jam    = jams.load(jams_path)
    annots = jam.search(namespace="chord")
    if not annots:
        return []

    events = []
    for interval, value in zip(*annots[0].to_interval_values()):
        start, end = float(interval[0]), float(interval[1])
        events.append((start, end, str(value)))
    return events


def parse_idmt_xml_chords(xml_path: str) -> list[tuple[float, float, str]]:
    """
    IDMT-SMT-Guitar Subset 4: reads chord annotations from XML.
    Relevant tag structure:
      <transcription>
        <event>
          <onsetSec>0.0</onsetSec>
          <offsetSec>1.0</offsetSec>
          <chord>C:maj</chord>
        </event>
      </transcription>
    Adjust tag names if your XML differs — inspect one file first.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    events = []
    for event in root.iter("event"):
        onset  = event.findtext("onsetSec")  or event.findtext("onset")
        offset = event.findtext("offsetSec") or event.findtext("offset")
        chord  = event.findtext("chord")     or event.findtext("label")

        if onset and offset and chord:
            events.append((float(onset), float(offset), chord.strip()))
    return events


def guitar_techs_label_from_path(wav_path: str) -> str:
    """
    Guitar-TECHS quality is in the FILENAME, not the folder.
    Pattern:  directinput_Set1_aug.wav  →  "X:aug"
              directinput_Drop3_Maj7.wav → "X:maj7"
              micamp_Set2_min.wav        → "X:min"
    Split stem on '_', check each token from the end.
    """
    quality_map = {
        "maj": "maj", "min": "min", "aug": "aug", "dim": "dim",
        "Maj7": "maj7", "7": "7", "m7": "min7", "m7b5": "min7b5",
    }
    stem   = Path(wav_path).stem      # e.g. "directinput_Set1_aug"
    tokens = stem.split("_")          # ["directinput", "Set1", "aug"]
    for token in reversed(tokens):    # quality token is always last
        if token in quality_map:
            return f"X:{quality_map[token]}"
    return NO_CHORD


# ─────────────────────────────────────────────────────────
# DATASET PROCESSORS
# ─────────────────────────────────────────────────────────
def process_guitarset(
    audio_paths: list[str],
) -> list[tuple[np.ndarray, str]]:
    """
    Processes GuitarSet mono-mic recordings with JAMS chord annotations.

    Annotation dir is data/raw/guitarset/annotation/ (no 's', no nesting).
    JAMS stems drop the mic/hex suffix, e.g.:
      audio:  00_BN1-129-Eb_comp_mic.wav
      jams:   00_BN1-129-Eb_comp.jams   ← strip _mic / _hex_cln / _mix
    """
    annot_dir   = DATASET_ROOTS["guitarset"] / "annotation"
    all_samples = []
    skipped     = 0

    # Suffixes appended to audio filenames that are absent from JAMS stems
    AUDIO_SUFFIXES = ["_mic", "_hex_cln", "_hex_orig", "_mix", "_hex"]

    for wav_path in tqdm(audio_paths, desc="GuitarSet"):
        wav_path = str(wav_path)   # ensure str in case Path slipped through
        stem     = Path(wav_path).stem

        # Strip audio-type suffix to get the bare JAMS stem
        jams_stem = stem
        for suf in AUDIO_SUFFIXES:
            if jams_stem.endswith(suf):
                jams_stem = jams_stem[: -len(suf)]
                break

        jams_path = annot_dir / f"{jams_stem}.jams"
        if not jams_path.exists():
            skipped += 1
            continue

        chords = parse_jams_chords(str(jams_path))
        if not chords:
            skipped += 1
            continue

        try:
            y   = load_and_clean(wav_path)
            cqt = compute_cqt(y)
            all_samples.extend(slice_windows(cqt, chords))
        except Exception as e:
            print(f"\n  ⚠️  Skipping {Path(wav_path).name}: {e}")
            skipped += 1

    if skipped:
        print(f"  ℹ️  GuitarSet: {skipped} files skipped (no matching JAMS or load error)")
    return all_samples


def process_guitar_techs() -> list[tuple[np.ndarray, str]]:
    """
    Processes Guitar-TECHS chord recordings.
    Structure: P1_chords/audio/directinput/*.wav
               P2_chords/audio/directinput/*.wav
    Uses directinput only (one signal per chord, no mic duplicates).
    Quality label extracted from filename stem.
    """
    root        = DATASET_ROOTS["guitar_techs"]
    chord_dirs  = [d for d in root.iterdir()
                   if d.is_dir() and "chord" in d.name.lower()]

    all_wav = []
    for chord_dir in sorted(chord_dirs):
        di_dir = chord_dir / "audio" / "directinput"
        if di_dir.exists():
            all_wav.extend(sorted(di_dir.glob("*.wav")))

    all_samples = []
    skipped     = 0

    for wav_path in tqdm(all_wav, desc="Guitar-TECHS chords"):
        label = guitar_techs_label_from_path(str(wav_path))
        if label == NO_CHORD:
            skipped += 1
            continue

        try:
            y        = load_and_clean(str(wav_path))
            cqt      = compute_cqt(y)
            duration = y.shape[0] / SR
            chord_events = [(0.0, duration, label)]
            all_samples.extend(slice_windows(cqt, chord_events))
        except Exception as e:
            print(f"\n  ⚠️  Skipping {wav_path.name}: {e}")
            skipped += 1

    if skipped:
        print(f"  ℹ️  Guitar-TECHS: {skipped} files skipped")
    return all_samples


def process_idmt() -> list[tuple[np.ndarray, str]]:
    """
    Processes IDMT-SMT-Guitar — searches ALL dataset subfolders
    (dataset1–4) for paired WAV+XML files.

    Subset 4 is the chord recognition subset; subsets 1-3 contain
    individual note events with pitch/technique annotations, not chords.
    We process all subsets but only keep windows where chord XML tags
    are found — subset 4 files will contribute, others will be skipped
    silently via empty chord_events.

    Audio/annotation layout per file:
      .../dataset4/<guitar>/audio/<stem>.wav
      .../dataset4/<guitar>/annotation/<stem>.xml
    """
    idmt_v2 = DATASET_ROOTS["idmt"] / "IDMT-SMT-GUITAR_V2"
    if not idmt_v2.exists():
        print(f"  ⚠️  IDMT not found at {idmt_v2} — skipping.")
        return []

    # Find dataset4 specifically (chord recognition subset)
    dataset4 = idmt_v2 / "dataset4"
    if dataset4.exists():
        search_root = dataset4
        print(f"  ✅ Using IDMT dataset4 (chord subset): {dataset4}")
    else:
        # Fall back to searching all datasets — only chord XMLs will yield windows
        search_root = idmt_v2
        print(f"  ℹ️  dataset4 not found — searching all IDMT subsets for chord XMLs")

    wav_files   = sorted(search_root.rglob("*.wav"))
    all_samples = []
    skipped     = 0

    for wav_path in tqdm(wav_files, desc="IDMT"):
        # XML is in sibling annotation/ folder with same stem
        xml_path = wav_path.parent.parent / "annotation" / f"{wav_path.stem}.xml"
        if not xml_path.exists():
            # Try same directory
            xml_path = wav_path.with_suffix(".xml")
        if not xml_path.exists():
            skipped += 1
            continue

        chords = parse_idmt_xml_chords(str(xml_path))
        if not chords:
            skipped += 1
            continue

        try:
            y   = load_and_clean(str(wav_path))
            cqt = compute_cqt(y)
            all_samples.extend(slice_windows(cqt, chords))
        except Exception as e:
            print(f"\n  ⚠️  Skipping {wav_path.name}: {e}")
            skipped += 1

    print(f"  ℹ️  IDMT: {skipped} files skipped (no chord XML or load error)")
    return all_samples


# ─────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────
def build_label_map(samples: list[tuple[np.ndarray, str]]) -> dict[str, int]:
    """
    Creates a stable label_map sorted alphabetically, with 'N' always at 0.
    """
    labels = sorted(set(label for _, label in samples))
    if NO_CHORD in labels:
        labels.remove(NO_CHORD)
    label_map = {NO_CHORD: 0}
    label_map.update({lbl: i + 1 for i, lbl in enumerate(labels)})
    return label_map


def encode(
    samples: list[tuple[np.ndarray, str]],
    label_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts list of (window, label_str) into (X, y) numpy arrays.
    X shape: (N, 1, 84, T_FRAMES)   — the '1' is the channel dim for Conv2d
    y shape: (N,)                   — integer class indices
    """
    X, y = [], []
    for window, label in samples:
        # Skip labels not in the map (rare chords from validation leak)
        if label not in label_map:
            continue
        X.append(window[np.newaxis, :, :])   # add channel dim → (1, 84, T)
        y.append(label_map[label])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────────────────
# SPLIT LOADING
# ─────────────────────────────────────────────────────────
def _flatten_split(value) -> list[str]:
    """
    Normalises one split value into a flat list of file-path strings.

    Handles every shape P3 scripts have been known to produce:
      • list of str        → ["path/a.wav", ...]          (ideal)
      • dict of lists      → {"mic": [...], "hex": [...]}  (nested)
      • list of dicts      → [{"mic": "p.wav"}, ...]       (unusual)
      • single str         → "path/a.wav"                  (edge case)
    """
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, dict):
                # GuitarSet splits JSON: {"audio": "/path/file.wav", "jams": "..."}
                # Prefer the "audio" key; fall back to first string value found
                if "audio" in item:
                    flat.append(item["audio"])
                else:
                    for v in item.values():
                        if isinstance(v, str):
                            flat.append(v)
                            break
                        elif isinstance(v, list):
                            flat.extend(str(x) for x in v if isinstance(x, str))
            else:
                flat.append(str(item))
        return flat

    if isinstance(value, dict):
        # e.g. {"mic": ["a.wav", "b.wav"], "hex": ["c.wav"]}
        flat = []
        for v in value.values():
            flat.extend(_flatten_split(v))
        return flat

    return []


def load_guitarset_splits() -> dict[str, list[str]]:
    """
    Loads the pre-computed player-wise split from P3 and normalises it
    into a guaranteed  {"train": [str, ...], "val": [...], "test": [...]}
    regardless of the exact JSON shape dataset_lab.py produced.
    """
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Splits file not found: {SPLITS_FILE}\n"
            "Run dataset_lab.py first to generate splits.\n"
            f"Expected path: {SPLITS_FILE.resolve()}"
        )

    with open(SPLITS_FILE) as f:
        raw = json.load(f)

    # Diagnose structure on first load so future bugs are obvious
    example_val = next(iter(raw.values())) if raw else None
    if isinstance(example_val, dict):
        print(f"  ℹ️  Splits JSON has nested structure — flattening automatically.")
    elif isinstance(example_val, list) and example_val and isinstance(example_val[0], dict):
        print(f"  ℹ️  Splits JSON contains list-of-dicts — flattening automatically.")

    normalised = {split: _flatten_split(paths) for split, paths in raw.items()}

    for split, paths in normalised.items():
        print(f"  📂 {split:6s}: {len(paths)} audio files")

    return normalised


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 55)
    print("  GuitarAI — Chord Preprocessing Pipeline")
    print("=" * 55)

    # ── 1. GuitarSet (player-wise split) ──────────────────
    splits    = load_guitarset_splits()
    gs_train  = process_guitarset(splits["train"])
    gs_val    = process_guitarset(splits["val"])
    gs_test   = process_guitarset(splits["test"])

    # ── 2. Guitar-TECHS (chord folders only) ──────────────
    gt_all    = process_guitar_techs()

    # Rough 80/10/10 split for Guitar-TECHS (no player split available)
    rng       = np.random.default_rng(seed=42)
    indices   = rng.permutation(len(gt_all))
    n         = len(gt_all)
    n_test    = max(1, int(n * 0.10))
    n_val     = max(1, int(n * 0.10))
    gt_test   = [gt_all[i] for i in indices[:n_test]]
    gt_val    = [gt_all[i] for i in indices[n_test:n_test + n_val]]
    gt_train  = [gt_all[i] for i in indices[n_test + n_val:]]

    # ── 3. IDMT (auto-discovers dataset4 or all subsets) ──
    idmt_all = process_idmt()

    # Same 80/10/10 for IDMT
    if idmt_all:
        rng2    = np.random.default_rng(seed=99)
        idx2    = rng2.permutation(len(idmt_all))
        m       = len(idmt_all)
        m_test  = max(1, int(m * 0.10))
        m_val   = max(1, int(m * 0.10))
        idmt_test  = [idmt_all[i] for i in idx2[:m_test]]
        idmt_val   = [idmt_all[i] for i in idx2[m_test:m_test + m_val]]
        idmt_train = [idmt_all[i] for i in idx2[m_test + m_val:]]
    else:
        idmt_train = idmt_val = idmt_test = []
        print("⚠️  IDMT produced 0 windows — check XML chord tag names.")

    # ── 4. Combine splits ─────────────────────────────────
    train_all = gs_train + gt_train + idmt_train
    val_all   = gs_val   + gt_val   + idmt_val
    test_all  = gs_test  + gt_test  + idmt_test

    print(f"\n📦 Raw window counts:")
    print(f"   Train : {len(train_all):,}")
    print(f"   Val   : {len(val_all):,}")
    print(f"   Test  : {len(test_all):,}")

    # ── 5. Build label map from TRAIN only ────────────────
    label_map = build_label_map(train_all)
    print(f"\n🏷️  Unique chord classes : {len(label_map)}")

    with open(LABEL_MAP_F, "w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"   Saved → {LABEL_MAP_F}")

    # ── 6. Encode and save .npy ───────────────────────────
    for split_name, samples in [
        ("train", train_all),
        ("val",   val_all),
        ("test",  test_all),
    ]:
        X, y = encode(samples, label_map)
        np.save(OUT_DIR / f"X_{split_name}.npy", X)
        np.save(OUT_DIR / f"y_{split_name}.npy", y)
        print(f"   ✅ {split_name}: X={X.shape}  y={y.shape}")

    print("\n✅ Preprocessing complete.")
    print(f"   Output → {OUT_DIR}")


if __name__ == "__main__":
    main()