"""
train_chord.py
--------------
Training engine for ChordCNN.

Features:
  • PyTorch Dataset + DataLoader with WeightedRandomSampler
    (solves class imbalance — rare chords get equal training time)
  • Adam optimiser with CosineAnnealingLR scheduler
  • Train / Validation loop with early stopping
  • Saves best model checkpoint automatically
  • Logs epoch metrics to training_log.json

Run:  python -m src.ml.train_chord
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.ml.models import build_model, save_model

# ─────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
DATA_DIR       = Path("data/processed/chord_dataset")
LABEL_MAP_FILE = DATA_DIR / "label_map.json"
MODEL_OUT      = "models/chord_cnn.pth"
LOG_OUT        = "models/training_log.json"

BATCH_SIZE     = 64
EPOCHS         = 50
LR             = 3e-4          # Adam default; works well for audio CNNs
WEIGHT_DECAY   = 1e-4          # L2 regularisation
DROPOUT        = 0.5
PATIENCE       = 8             # early-stopping patience (epochs)
NUM_WORKERS    = 4             # DataLoader parallel workers (0 on Windows)

DEVICE = (
    "cuda"  if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else   # Apple Silicon
    "cpu"
)


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class ChordDataset(Dataset):
    """
    Lazy-loading wrapper around pre-computed .npy arrays.
    X: (N, 1, 84, 87) float32
    y: (N,)            int64
    """

    def __init__(self, split: str, augment: bool = False):
        """
        split   : 'train' | 'val' | 'test'
        augment : if True, applies light data augmentation on-the-fly
        """
        self.X       = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
        self.y       = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx].copy()).float()   # (1, 84, 87)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)

        if self.augment:
            x = self._augment(x)
        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight augmentation that preserves harmonic content:
          1. Random pitch shift (±1 bin) — simulates slight detuning
          2. Random time shift (±3 frames) — simulates onset jitter
          3. Random amplitude scaling [0.8, 1.0]
        """
        # 1. Pitch shift: roll along frequency axis
        shift = np.random.randint(-1, 2)   # -1, 0, or +1
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=1)

        # 2. Time jitter: roll along time axis
        t_shift = np.random.randint(-3, 4)
        if t_shift != 0:
            x = torch.roll(x, shifts=t_shift, dims=2)

        # 3. Amplitude scaling
        scale = np.random.uniform(0.8, 1.0)
        x     = x * scale

        return x


# ─────────────────────────────────────────────────────────
# WEIGHTED SAMPLER (solves class imbalance)
# ─────────────────────────────────────────────────────────
def make_weighted_sampler(dataset: ChordDataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler so that every chord class appears
    equally often during training, regardless of raw class frequencies.

    Without this, a dataset with 10× more 'G:maj' than 'Eb:dim' will
    produce a model that simply ignores rare chords.
    """
    labels      = dataset.y[:]           # numpy array
    class_counts = np.bincount(labels)
    # weight per class = 1 / count  (rare classes get high weight)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    # assign each sample its class weight
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(dataset),
        replacement = True,
    )


# ─────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────
def run_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer | None,
    device:      str,
    is_training: bool,
) -> tuple[float, float]:
    """
    Runs one full epoch.
    Returns (avg_loss, accuracy_percent).
    Pass optimizer=None for validation / test.
    """
    model.train(is_training)
    total_loss  = 0.0
    correct     = 0
    total       = 0

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  GuitarAI — Chord CNN Training")
    print(f"  Device : {DEVICE.upper()}")
    print("=" * 55)

    # ── Load label map ────────────────────────────────────
    with open(LABEL_MAP_FILE) as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"\n🏷️  Chord classes : {num_classes}")

    # ── Datasets ──────────────────────────────────────────
    train_ds = ChordDataset("train", augment=True)
    val_ds   = ChordDataset("val",   augment=False)

    print(f"📦 Train samples : {len(train_ds):,}")
    print(f"   Val   samples : {len(val_ds):,}")

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size   = BATCH_SIZE,
        sampler      = sampler,          # replaces shuffle=True
        num_workers  = NUM_WORKERS,
        pin_memory   = (DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = BATCH_SIZE * 2,   # larger batch for val (no grad)
        shuffle      = False,
        num_workers  = NUM_WORKERS,
        pin_memory   = (DEVICE == "cuda"),
    )

    # ── Model, loss, optimiser ────────────────────────────
    model     = build_model(num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    # CosineAnnealingLR: decays LR smoothly to near-zero by epoch EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────
    best_val_acc = 0.0
    patience_cnt = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, is_training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, DEVICE, is_training=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # ── Checkpoint best model ─────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            save_model(model, MODEL_OUT)
            print(f"  ⭐ New best val_acc={best_val_acc:.1f}% — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc, 3),
            "val_loss":   round(val_loss, 5),
            "val_acc":    round(val_acc, 3),
            "lr":         current_lr,
        })

    # ── Save training log ─────────────────────────────────
    Path(LOG_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📈 Training log saved → {LOG_OUT}")
    print(f"🏆 Best validation accuracy : {best_val_acc:.1f}%")


if __name__ == "__main__":
    train()