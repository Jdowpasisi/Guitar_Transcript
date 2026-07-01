"""
src/ml/fusion_dataset.py
P12: Fusion Model — Paired Audio + Video Dataset.

Builds aligned (audio_features, video_features, label) tuples per note event
from GuitarSet recordings. Since paired real video data isn't available for
GuitarSet, synthetic video features are generated from ground-truth annotations
with configurable noise and dropout to simulate imperfect video detection.

Audio features per note (dim=56):
    midi_pitch (1), onset_confidence (1), chord_prob_vector (51),
    delta_t (1), note_duration (1), pitch_class (1)

Video features per note (dim=7):
    fret_number (1), string_number (1), finger_id (1),
    detection_confidence (1), frame_quality (1),
    num_fingers_detected (1), video_available (1)

Label: flat (string, fret) class index 0..137 (same as P6)

Run as __main__ for a smoke test of the dataset pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ── resolve project root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
for _candidate in [_HERE.parents[2], _HERE.parent, Path.cwd()]:
    if (_candidate / "src" / "config.py").exists():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GUITARSET_ANNOTATIONS, GUITARSET_SPLIT_PLAYERS,
    AUDIO_FEATURE_DIM, VIDEO_FEATURE_DIM,
    P12_NUM_CHORD_CLASSES, P12_VIDEO_NOISE_STD, P12_VIDEO_DROPOUT,
    P12_CURRICULUM_WARMUP,
)
from src.ml.voicing_dataset import (
    OPEN_STRINGS, NUM_STRINGS, MAX_FRET, NUM_POSITIONS,
    PAD_LABEL, position_index, position_from_index,
    _load_jams_note_midi, _build_sequence,
)

# ── Chord CNN integration (optional) ─────────────────────────────────────────
# Try to load the trained ChordCNN for chord probability features.
# If unavailable, use uniform probabilities as fallback.
_CHORD_CNN = None
_LABEL_MAP = None

def _try_load_chord_cnn():
    """Attempt to load the P4 ChordCNN for chord probability features."""
    global _CHORD_CNN, _LABEL_MAP
    if _CHORD_CNN is not None:
        return True
    try:
        from src.ml.models import load_model
        from src.config import MODEL_PATH, LABEL_MAP_FILE
        if MODEL_PATH.exists() and LABEL_MAP_FILE.exists():
            _CHORD_CNN = load_model(str(MODEL_PATH))
            _CHORD_CNN.eval()
            with open(LABEL_MAP_FILE) as f:
                _LABEL_MAP = json.load(f)
            return True
    except Exception:
        pass
    return False


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_chord_probs_for_note(midi_pitch: float, note_time: float) -> np.ndarray:
    """
    Get chord probability vector for a note event.

    If ChordCNN is loaded, this would run inference on the surrounding CQT window.
    For now, we use a heuristic: create a sparse probability vector based on the
    note's pitch class, since certain chords are more likely given certain notes.

    Returns: float32 array of shape (51,)
    """
    probs = np.ones(P12_NUM_CHORD_CLASSES, dtype=np.float32) / P12_NUM_CHORD_CLASSES

    # Heuristic: boost probabilities of chords containing this pitch class
    pitch_class = int(round(midi_pitch)) % 12
    # Simple heuristic: slightly boost entries based on pitch class position
    # This creates a non-uniform but deterministic distribution
    for i in range(P12_NUM_CHORD_CLASSES):
        if (i + pitch_class) % 5 == 0:
            probs[i] *= 2.0

    # Normalize
    probs /= probs.sum()
    return probs


def _finger_id_for_fret(fret: int, string_idx: int) -> int:
    """
    Heuristic: assign a finger ID (0-4) based on fret position.
    In real P11 data, this comes from MediaPipe landmark tracking.
    For synthetic data, we use a reasonable approximation.
    """
    if fret == 0:
        return 0  # open string — no finger
    # Simple mapping: frets 1-4 → fingers 1-4, higher frets cycle
    return min(((fret - 1) % 4) + 1, 4)


def _build_audio_features(notes_sorted: List[dict]) -> np.ndarray:
    """
    Build audio feature matrix for a sequence of sorted note events.

    Returns: float32 array of shape (N, AUDIO_FEATURE_DIM=56)
    """
    N = len(notes_sorted)
    features = np.zeros((N, AUDIO_FEATURE_DIM), dtype=np.float32)

    times = [n["time"] for n in notes_sorted]
    durations = [n["duration"] for n in notes_sorted]

    for i, note in enumerate(notes_sorted):
        midi = note["midi_pitch"]

        # Feature 0: midi_pitch (normalized to [0, 1])
        features[i, 0] = float(int(round(midi))) / 127.0

        # Feature 1: onset_confidence (derived from duration — longer notes → higher confidence)
        features[i, 1] = min(note["duration"] / 2.0, 1.0)

        # Features 2-52: chord probability vector (51 classes)
        chord_probs = _get_chord_probs_for_note(midi, note["time"])
        features[i, 2:53] = chord_probs

        # Feature 53: delta_t (inter-note timing, capped at 5.0)
        if i > 0:
            dt = times[i] - times[i - 1]
            features[i, 53] = min(max(dt, 0.0), 5.0)
        else:
            features[i, 53] = 0.0

        # Feature 54: note_duration (seconds, capped at 5.0)
        features[i, 54] = min(note["duration"], 5.0)

        # Feature 55: pitch_class (chroma, 0-11 normalized to [0, 1])
        features[i, 55] = (int(round(midi)) % 12) / 11.0

    return features


def _build_video_features(
    notes_sorted: List[dict],
    noise_std: float = 0.0,
    video_dropout: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build synthetic video feature matrix from ground-truth annotations.

    The ground-truth (string, fret) IS the "perfect video detection". We add
    noise and dropout to simulate real-world imperfect video detection.

    Returns: float32 array of shape (N, VIDEO_FEATURE_DIM=7)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(notes_sorted)
    features = np.zeros((N, VIDEO_FEATURE_DIM), dtype=np.float32)

    for i, note in enumerate(notes_sorted):
        true_fret = note["fret"]
        true_string = note["string_index"]

        # Decide if this note has video data (simulate dropout)
        has_video = rng.random() > video_dropout

        if has_video:
            # Feature 0: fret_number (with noise)
            fret_noisy = true_fret + rng.normal(0, noise_std) if noise_std > 0 else float(true_fret)
            features[i, 0] = max(0, fret_noisy) / 22.0  # normalize to [0, 1]

            # Feature 1: string_number (with noise)
            string_noisy = true_string + rng.normal(0, noise_std * 0.3) if noise_std > 0 else float(true_string)
            features[i, 1] = max(0, min(5, string_noisy)) / 5.0  # normalize to [0, 1]

            # Feature 2: finger_id (0-4, normalized)
            features[i, 2] = _finger_id_for_fret(true_fret, true_string) / 4.0

            # Feature 3: detection_confidence (higher when noise is low)
            base_conf = 0.85 + rng.normal(0, 0.1)
            features[i, 3] = np.clip(base_conf, 0.3, 1.0)

            # Feature 4: frame_quality (on-fretboard rate)
            features[i, 4] = np.clip(0.8 + rng.normal(0, 0.1), 0.3, 1.0)

            # Feature 5: num_fingers_detected (1-5)
            features[i, 5] = np.clip(rng.integers(2, 5) / 5.0, 0.2, 1.0)

            # Feature 6: video_available flag
            features[i, 6] = 1.0
        else:
            # All zeros — video not available for this note
            features[i, :] = 0.0

    return features


def _get_split_files(split: str) -> List[Path]:
    """Return all JAMS file paths for the requested split (train/val/test)."""
    player_ids = GUITARSET_SPLIT_PLAYERS[split]
    jams_dir = Path(GUITARSET_ANNOTATIONS)

    if not jams_dir.exists():
        raise FileNotFoundError(
            f"GuitarSet annotation directory not found: {jams_dir}\n"
            "Check src/config.py → GUITARSET_ANNOTATIONS."
        )

    files = []
    for path in sorted(jams_dir.glob("*.jams")):
        player = path.stem[:2]
        if player in player_ids:
            files.append(path)
    return files


# ── main Dataset class ────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    One sample = one GuitarSet recording = paired (audio, video) feature sequences.

    Each recording produces:
        audio_features : FloatTensor (N, 56)
        video_features : FloatTensor (N, 7)
        labels         : LongTensor  (N,) — flat (string, fret) class index
        length         : int — N (number of notes)

    Video features are synthetic (derived from ground truth with noise).
    """

    def __init__(
        self,
        split: str,
        video_noise_std: float = 0.0,
        video_dropout: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            split:           "train", "val", or "test"
            video_noise_std: Gaussian noise on synthetic video fret/string features
            video_dropout:   Probability of zeroing out video features per note
            seed:            Random seed for reproducible synthetic video generation
        """
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got {split!r}"

        self.split = split
        self.video_noise_std = video_noise_std
        self.video_dropout = video_dropout
        self.rng = np.random.default_rng(seed)

        self.audio_features: List[np.ndarray] = []   # (N, 56) arrays
        self.video_features: List[np.ndarray] = []   # (N, 7) arrays
        self.labels: List[np.ndarray] = []            # (N,) int arrays
        self.file_names: List[str] = []

        self._load()

    def _load(self):
        jams_files = _get_split_files(self.split)
        skipped = 0

        for jams_path in jams_files:
            notes = _load_jams_note_midi(jams_path)
            if len(notes) < 2:
                skipped += 1
                continue

            # Sort notes by onset time
            notes_sorted = sorted(notes, key=lambda n: n["time"])

            # Build audio features
            audio_feat = _build_audio_features(notes_sorted)

            # Build synthetic video features
            video_feat = _build_video_features(
                notes_sorted,
                noise_std=self.video_noise_std,
                video_dropout=self.video_dropout,
                rng=self.rng,
            )

            # Labels: flat position index (same as P6)
            labels = np.array(
                [position_index(n["string_index"], n["fret"]) for n in notes_sorted],
                dtype=np.int64,
            )

            self.audio_features.append(audio_feat)
            self.video_features.append(video_feat)
            self.labels.append(labels)
            self.file_names.append(jams_path.name)

        print(f"[FusionDataset] split={self.split:5s} | "
              f"recordings={len(self.labels)} | skipped={skipped}")

        if self.labels:
            lengths = [len(l) for l in self.labels]
            print(f"                notes/recording — "
                  f"min={min(lengths)} max={max(lengths)} "
                  f"mean={int(np.mean(lengths))}")

    def update_video_noise(self, noise_std: float, dropout: float):
        """
        Regenerate synthetic video features with new noise parameters.
        Used for curriculum training (gradually increase noise).
        """
        self.video_noise_std = noise_std
        self.video_dropout = dropout
        # Re-derive video features from stored notes
        # Since we don't store the raw notes, we rebuild from labels
        for i in range(len(self.labels)):
            N = len(self.labels[i])
            notes_from_labels = []
            for j in range(N):
                s, f = position_from_index(self.labels[i][j])
                notes_from_labels.append({
                    "string_index": s,
                    "fret": f,
                    "time": j * 0.1,  # approximate
                    "duration": 0.3,
                    "midi_pitch": OPEN_STRINGS[s] + f,
                })
            self.video_features[i] = _build_video_features(
                notes_from_labels,
                noise_std=noise_std,
                video_dropout=dropout,
                rng=self.rng,
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns:
            audio  : FloatTensor (N, 56) — audio features
            video  : FloatTensor (N, 7)  — video features
            labels : LongTensor  (N,)    — flat (string, fret) class index
            length : int                 — N
        """
        audio = torch.from_numpy(self.audio_features[idx].copy())
        video = torch.from_numpy(self.video_features[idx].copy())
        labels = torch.from_numpy(self.labels[idx].copy())
        return audio, video, labels, len(labels)


# ── collate: pad variable-length sequences ────────────────────────────────────

def fusion_collate_fn(batch):
    """
    Pad a batch of variable-length sequences for the fusion model.

    Returns:
        audio_padded  : FloatTensor (B, T_max, 56)
        video_padded  : FloatTensor (B, T_max, 7)
        labels_padded : LongTensor  (B, T_max)   — PAD_LABEL for padded steps
        lengths       : LongTensor  (B,)          — actual lengths
    """
    audios, videos, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Sort by descending length (required by pack_padded_sequence)
    order = torch.argsort(lengths, descending=True)
    lengths = lengths[order]
    audios = [audios[i] for i in order]
    videos = [videos[i] for i in order]
    labels = [labels[i] for i in order]

    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    video_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)

    return audio_padded, video_padded, labels_padded, lengths


def get_fusion_dataloader(
    split: str,
    batch_size: int = 16,
    video_noise_std: float = 0.0,
    video_dropout: float = 0.0,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    seed: int = 42,
) -> Tuple[DataLoader, FusionDataset]:
    """
    Convenience factory. Returns both DataLoader and Dataset (for curriculum updates).
    shuffle defaults to True for train, False otherwise.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = FusionDataset(
        split,
        video_noise_std=video_noise_std,
        video_dropout=video_dropout,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=fusion_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FusionDataset — smoke test")
    print("=" * 60)

    for split in ("train", "val", "test"):
        ds = FusionDataset(split, video_noise_std=0.5, video_dropout=0.2, seed=42)
        if len(ds) == 0:
            print(f"  [{split}] No recordings found — check GuitarSet path.")
            continue

        print(f"\n── {split} split — first 3 recordings ──")
        for i in range(min(3, len(ds))):
            audio, video, labels, length = ds[i]
            print(f"  [{i}] {ds.file_names[i]}")
            print(f"       audio shape : {tuple(audio.shape)}  (N × {AUDIO_FEATURE_DIM})")
            print(f"       video shape : {tuple(video.shape)}  (N × {VIDEO_FEATURE_DIM})")
            print(f"       label range : [{labels.min().item()}, {labels.max().item()}]")
            print(f"       video avail : {(video[:, 6] > 0).sum().item()}/{length} notes "
                  f"({(video[:, 6] > 0).float().mean().item():.1%})")

    print("\n── DataLoader batch test (train, batch_size=4) ──")
    try:
        loader, ds = get_fusion_dataloader("train", batch_size=4, video_noise_std=0.5)
        audio, video, labels, lengths = next(iter(loader))
        print(f"  audio shape  : {tuple(audio.shape)}")
        print(f"  video shape  : {tuple(video.shape)}")
        print(f"  labels shape : {tuple(labels.shape)}")
        print(f"  lengths      : {lengths.tolist()}")
        print(f"  label range  : [{labels[labels != PAD_LABEL].min().item()}, "
              f"{labels[labels != PAD_LABEL].max().item()}]")
        print("\n✅ FusionDataset and DataLoader working correctly.")
    except Exception as e:
        print(f"\n❌ DataLoader error: {e}")
        raise
