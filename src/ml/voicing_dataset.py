"""
src/ml/voicing_dataset.py
P6: Voicing LSTM — Dataset builder and DataLoader.

Pulls note sequences from GuitarSet's note_midi JAMS namespace via the
existing get_sample() API, collates them per recording into sorted sequences
of (midi_pitch, delta_t, string_index, fret) tuples, and splits by player:
  train : players 00-03
  val   : player 04
  test  : player 05

Each sample in the PyTorch Dataset is a single recording's full note sequence.
Variable-length sequences are padded and packed in the training loop.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ── resolve project root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
# Works whether file lives at src/ml/voicing_dataset.py or at project root
for _candidate in [_HERE.parents[2], _HERE.parent, Path.cwd()]:
    if (_candidate / "src" / "config.py").exists():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GUITARSET_ANNOTATIONS, DATA_SPLITS_DIR, GUITARSET_SPLIT_PLAYERS

# ── constants ─────────────────────────────────────────────────────────────────
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]   # E2 A2 D3 G3 B3 E4 (MIDI)
NUM_STRINGS  = 6
MAX_FRET     = 22
NUM_POSITIONS = NUM_STRINGS * (MAX_FRET + 1)   # 138

# Padding token used to fill sequences to the same length within a batch
PAD_LABEL = -1   # ignored by CrossEntropyLoss(ignore_index=PAD_LABEL)


# ── helpers ───────────────────────────────────────────────────────────────────

def position_index(string_idx: int, fret: int) -> int:
    """Map (string, fret) → flat class index 0..137."""
    return string_idx * (MAX_FRET + 1) + fret


def position_from_index(idx: int) -> Tuple[int, int]:
    """Inverse of position_index."""
    return divmod(idx, MAX_FRET + 1)


def _load_jams_note_midi(jams_path: Path) -> List[Dict]:
    """
    Parse the note_midi namespace from a GuitarSet JAMS file without
    requiring the jams library (plain JSON parse, matching dataset_lab.py style).

    Returns a list of dicts:
        {"time": float, "duration": float, "midi_pitch": float,
         "string_index": int, "fret": int}

    Only notes with fret in [0, MAX_FRET] are included.
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        jams = json.load(f)

    annotations = jams.get("annotations", [])
    notes = []

    string_counter = 0
    for ann in annotations:
        namespace = ann.get("namespace", "")
        if namespace != "note_midi":
            continue

        string_idx = string_counter
        string_counter += 1
        if string_idx >= NUM_STRINGS:
            break

        open_midi = OPEN_STRINGS[string_idx]
        data = ann.get("data", [])

        for event in data:
            # JAMS data entries: {time, duration, value, confidence}
            time     = float(event.get("time", 0.0))
            duration = float(event.get("duration", 0.0))
            value    = event.get("value", None)

            if value is None:
                continue

            # value is the MIDI pitch (float for micro-tonal; round to nearest)
            midi_pitch = float(value)
            fret = int(round(midi_pitch)) - open_midi

            if fret < 0 or fret > MAX_FRET:
                continue   # outside playable range for this string

            notes.append({
                "time":         time,
                "duration":     duration,
                "midi_pitch":   midi_pitch,
                "string_index": string_idx,
                "fret":         fret,
            })

    return notes


def _build_sequence(notes: List[Dict]) -> Optional[np.ndarray]:
    """
    Sort notes by onset time and build the feature array for one recording.

    Returns float32 array of shape (N, 4):
        col 0 : midi_pitch   (float, rounded to int for embedding)
        col 1 : delta_t      (seconds since previous note; 0 for first note)
        col 2 : string_index (0-5)
        col 3 : fret         (0-22)

    Returns None if the recording has < 2 notes.
    """
    if len(notes) < 2:
        return None

    notes_sorted = sorted(notes, key=lambda n: n["time"])

    times  = np.array([n["time"]         for n in notes_sorted], dtype=np.float32)
    pitches = np.array([n["midi_pitch"]  for n in notes_sorted], dtype=np.float32)
    strings = np.array([n["string_index"] for n in notes_sorted], dtype=np.int32)
    frets   = np.array([n["fret"]         for n in notes_sorted], dtype=np.int32)

    delta_t = np.zeros(len(times), dtype=np.float32)
    delta_t[1:] = np.diff(times)
    delta_t = np.clip(delta_t, 0.0, 5.0)   # cap outliers (gaps between sections)

    seq = np.stack([pitches, delta_t, strings.astype(np.float32),
                    frets.astype(np.float32)], axis=1)   # (N, 4)
    return seq


def _get_split_files(split: str) -> List[Path]:
    """Return all JAMS file paths for the requested split (train/val/test)."""
    player_ids = GUITARSET_SPLIT_PLAYERS[split]
    jams_dir   = Path(GUITARSET_ANNOTATIONS)

    if not jams_dir.exists():
        raise FileNotFoundError(
            f"GuitarSet annotation directory not found: {jams_dir}\n"
            "Check src/config.py → GUITARSET_ANNOTATIONS."
        )

    files = []
    for path in sorted(jams_dir.glob("*.jams")):
        # GuitarSet filenames: 00_BN1-129-Eb_solo.jams  (player prefix is first 2 chars)
        player = path.stem[:2]
        if player in player_ids:
            files.append(path)

    return files


# ── main Dataset class ────────────────────────────────────────────────────────

class VoicingDataset(Dataset):
    """
    One sample = one GuitarSet recording = one note sequence.

    Each sequence is a float32 tensor of shape (N, 4):
        [midi_pitch, delta_t, string_index, fret]

    Labels are long tensors of shape (N,) — flat (string, fret) class indices.
    Both are returned by __getitem__; the DataLoader collate_fn pads them.
    """

    def __init__(self, split: str, max_delta_t: float = 5.0):
        """
        Args:
            split:       "train", "val", or "test"
            max_delta_t: cap on inter-note timing (seconds) to prevent outlier features
        """
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got {split!r}"

        self.split       = split
        self.max_delta_t = max_delta_t
        self.sequences: List[np.ndarray] = []   # (N, 4) arrays
        self.labels:    List[np.ndarray] = []   # (N,) int arrays
        self.file_names: List[str]        = []

        self._load()

    def _load(self):
        jams_files = _get_split_files(self.split)
        skipped = 0

        for jams_path in jams_files:
            notes = _load_jams_note_midi(jams_path)
            seq   = _build_sequence(notes)

            if seq is None:
                skipped += 1
                continue

            # Labels: flat position index for each note
            label = np.array(
                [position_index(int(row[2]), int(row[3])) for row in seq],
                dtype=np.int64,
            )

            self.sequences.append(seq)
            self.labels.append(label)
            self.file_names.append(jams_path.name)

        print(f"[VoicingDataset] split={self.split:5s} | "
              f"recordings={len(self.sequences)} | skipped={skipped}")

        if self.sequences:
            lengths = [len(s) for s in self.sequences]
            print(f"                 notes/recording — "
                  f"min={min(lengths)} max={max(lengths)} "
                  f"mean={int(np.mean(lengths))}")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Returns:
            seq   : FloatTensor (N, 4)  — [midi_pitch, delta_t, string, fret]
            labels: LongTensor  (N,)    — flat (string, fret) class index
            length: int                 — N (used for packing)
        """
        seq    = torch.from_numpy(self.sequences[idx].copy())
        labels = torch.from_numpy(self.labels[idx].copy())
        return seq, labels, len(seq)


# ── collate: pad variable-length sequences ────────────────────────────────────

def voicing_collate_fn(batch):
    """
    Pad a batch of variable-length sequences.

    Returns:
        seqs_padded   : FloatTensor (B, T_max, 4)
        labels_padded : LongTensor  (B, T_max)   — PAD_LABEL for padded steps
        lengths       : LongTensor  (B,)          — actual lengths (for packing)
    """
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Sort by descending length (required by pack_padded_sequence)
    order   = torch.argsort(lengths, descending=True)
    lengths = lengths[order]
    seqs    = [seqs[i]   for i in order]
    labels  = [labels[i] for i in order]

    seqs_padded   = pad_sequence(seqs,   batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=PAD_LABEL)

    return seqs_padded, labels_padded, lengths


def get_dataloader(split: str, batch_size: int = 16, num_workers: int = 0,
                   shuffle: Optional[bool] = None) -> DataLoader:
    """
    Convenience factory. shuffle defaults to True for train, False otherwise.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = VoicingDataset(split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=voicing_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("VoicingDataset — smoke test")
    print("=" * 60)

    for split in ("train", "val", "test"):
        ds = VoicingDataset(split)
        if len(ds) == 0:
            print(f"  [{split}] No recordings found — check GuitarSet path.")
            continue

        # Print a few sample sequences
        print(f"\n── {split} split — first 3 recordings ──")
        for i in range(min(3, len(ds))):
            seq, labels, length = ds[i]
            print(f"  [{i}] {ds.file_names[i]}")
            print(f"       seq shape : {tuple(seq.shape)}   (N notes × 4 features)")
            print(f"       label range: [{labels.min().item()}, {labels.max().item()}]")
            print(f"       first 5 notes:")
            print(f"         {'midi':>6}  {'delta_t':>8}  {'string':>6}  {'fret':>5}  {'label':>6}")
            for j in range(min(5, length)):
                midi, dt, s, fr = seq[j].tolist()
                lbl = labels[j].item()
                print(f"         {midi:6.1f}  {dt:8.4f}  {int(s):6d}  {int(fr):5d}  {lbl:6d}")

    print("\n── DataLoader batch test (train, batch_size=4) ──")
    try:
        loader = get_dataloader("train", batch_size=4)
        seqs, lbls, lengths = next(iter(loader))
        print(f"  seqs shape   : {tuple(seqs.shape)}")
        print(f"  labels shape : {tuple(lbls.shape)}")
        print(f"  lengths      : {lengths.tolist()}")
        print(f"  label range  : [{lbls[lbls != PAD_LABEL].min().item()}, "
              f"{lbls[lbls != PAD_LABEL].max().item()}]")
        print("\n✅ Dataset and DataLoader working correctly.")
    except Exception as e:
        print(f"\n❌ DataLoader error: {e}")
        raise