"""
src/ml/train_voicing.py
P6: Voicing LSTM — Training loop.

Key design choices:
  • Teacher forcing: ground-truth (string, fret) is fed as prev_position input
    at every step, not the model's own prediction.  This stabilises early training.
  • Pack-padded sequences: variable-length recordings are packed so the LSTM
    never sees padding tokens.
  • Adam, LR=1e-3, early stopping on val Tab Accuracy, patience=8.
  • Best checkpoint → models/voicing_lstm.pth

Run:
    python -m src.ml.train_voicing
or:
    python train_voicing.py          (if run from the project root)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

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

from src.config import MODELS_DIR, DEVICE
from src.ml.voicing_dataset import get_dataloader, PAD_LABEL, NUM_POSITIONS
from src.ml.models import VoicingLSTM

# ── hyperparameters ───────────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 16
MAX_EPOCHS   = 60
PATIENCE     = 8
CHECKPOINT   = str(MODELS_DIR / "voicing_lstm.pth")
LOG_FILE     = str(MODELS_DIR / "voicing_training_log.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_teacher_forced_inputs(
    labels_padded: torch.Tensor,   # (B, T) long — ground-truth position indices
    lengths: torch.Tensor,          # (B,)   long
) -> torch.Tensor:
    """
    Build the prev_positions tensor for teacher forcing.

    prev_positions[:, t] = label at step t-1 (the ground truth from the
    previous timestep).  At t=0, use position 0 (open E2 string) as BOS.

    Returns: LongTensor (B, T)
    """
    B, T = labels_padded.shape
    # Shift right: insert a '0' at position 0, drop the last element
    bos   = torch.zeros(B, 1, dtype=torch.long, device=labels_padded.device)
    prev  = torch.cat([bos, labels_padded[:, :-1]], dim=1)   # (B, T)

    # Replace any PAD_LABEL slots with 0 (they won't affect the loss anyway
    # because CrossEntropyLoss ignores those output positions)
    prev = prev.clamp(min=0)
    return prev


def _compute_tab_accuracy(
    logits:        torch.Tensor,   # (B, T, 138)
    labels_padded: torch.Tensor,   # (B, T) — contains PAD_LABEL for padding
) -> float:
    """Tab Accuracy: fraction of non-padded steps where argmax == label."""
    preds = logits.argmax(dim=-1)              # (B, T)
    mask  = labels_padded != PAD_LABEL
    correct = (preds == labels_padded) & mask
    return correct.sum().item() / mask.sum().item()


# ── one epoch: train or validate ──────────────────────────────────────────────

def _run_epoch(
    model:     VoicingLSTM,
    loader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer | None,
    device:    str,
    train:     bool,
) -> tuple[float, float]:
    """
    Returns (mean_loss, tab_accuracy) over the full epoch.
    If train=True, runs backward pass; optimizer must not be None.
    """
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_valid   = 0
    n_batches     = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seqs, labels, lengths in loader:
            seqs    = seqs.to(device)       # (B, T, 4)
            labels  = labels.to(device)     # (B, T)
            lengths = lengths.to(device)

            # Split feature columns
            midi_pitches = seqs[:, :, 0].long().clamp(0, 127)   # (B, T)
            delta_t      = seqs[:, :, 1]                         # (B, T)
            # ground-truth string/fret not used as model INPUT (teacher forcing
            # uses labels shifted by one step instead)

            # Teacher-forced previous positions
            prev_positions = _build_teacher_forced_inputs(labels, lengths)

            # Forward pass
            logits = model(midi_pitches, prev_positions, delta_t, lengths)
            # logits : (B, T, 138)

            # Flatten for loss: (B*T, 138) vs (B*T,)
            B, T, C = logits.shape
            loss = criterion(
                logits.reshape(B * T, C),
                labels.reshape(B * T),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            n_batches  += 1

            mask = labels != PAD_LABEL
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == labels) & mask).sum().item()
            total_valid   += mask.sum().item()

    mean_loss = total_loss / max(n_batches, 1)
    tab_acc   = total_correct / max(total_valid, 1)
    return mean_loss, tab_acc


# ── main training loop ────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("P6 Voicing LSTM — Training")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  LR          : {LR}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Max epochs  : {MAX_EPOCHS}")
    print(f"  Patience    : {PATIENCE}")
    print(f"  Checkpoint  : {CHECKPOINT}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets ...")
    train_loader = get_dataloader("train", batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader("val",   batch_size=BATCH_SIZE, shuffle=False)

    if len(train_loader.dataset) == 0:
        print("\n❌ No training data found. Check GuitarSet path in src/config.py.")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VoicingLSTM(num_positions=NUM_POSITIONS).to(DEVICE)
    print(f"\nModel: VoicingLSTM — {model.num_parameters:,} parameters")

    # ── Optimiser + Loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = -1.0
    patience_left = PATIENCE
    history       = []

    os.makedirs(os.path.dirname(CHECKPOINT) or ".", exist_ok=True)

    print()
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer=None, device=DEVICE, train=False
        )

        elapsed = time.time() - t0
        scheduler.step(val_acc)

        # Log
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
        }
        history.append(row)

        improved = val_acc > best_val_acc
        marker   = " ✓" if improved else ""
        print(f"{epoch:>6}  {train_loss:>11.5f}  {train_acc:>9.2%}  "
              f"{val_loss:>9.5f}  {val_acc:>7.2%}  {elapsed:>5.1f}s{marker}")

        if improved:
            best_val_acc  = val_acc
            patience_left = PATIENCE
            model.save(CHECKPOINT)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no val improvement for {PATIENCE} epochs).")
                break

    # ── Save training log ──────────────────────────────────────────────────────
    with open(LOG_FILE, "w") as f:
        json.dump({
            "best_val_acc": round(best_val_acc, 5),
            "epochs_run":   len(history),
            "history":      history,
        }, f, indent=2)

    print(f"\n{'─' * 68}")
    print(f"Training complete.")
    print(f"  Best val Tab Accuracy : {best_val_acc:.2%}")
    print(f"  Checkpoint saved      : {CHECKPOINT}")
    print(f"  Training log saved    : {LOG_FILE}")


if __name__ == "__main__":
    train()