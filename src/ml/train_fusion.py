"""
src/ml/train_fusion.py
P12: Fusion Model — Training Loop.

Trains the cross-attention FusionModel on paired (audio, video) features
from GuitarSet. Uses curriculum training:
    Phase 1 (first P12_CURRICULUM_WARMUP epochs): clean video features
    Phase 2 (remaining epochs): gradually increase video noise + dropout

Run:
    python -m src.ml.train_fusion
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

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
    DEVICE, MODELS_DIR, OUTPUTS_DIR,
    P12_BATCH_SIZE, P12_EPOCHS, P12_LR, P12_WEIGHT_DECAY,
    P12_PATIENCE, P12_GRAD_CLIP,
    P12_CURRICULUM_WARMUP, P12_VIDEO_NOISE_STD, P12_VIDEO_DROPOUT,
    FUSION_MODEL_PATH, FUSION_TRAINING_LOG,
)
from src.ml.fusion_dataset import (
    FusionDataset, fusion_collate_fn, PAD_LABEL,
)
from src.ml.fusion_model import FusionModel


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_epoch(
    model: FusionModel,
    loader,
    criterion: nn.CrossEntropyLoss,
    device: str,
    optimizer=None,
    grad_clip: float = 1.0,
    is_train: bool = True,
):
    """
    Run one epoch of training or evaluation.

    Returns: (avg_loss, tab_accuracy)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_notes = 0
    n_batches = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for audio, video, labels, lengths in loader:
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass
            logits = model(audio, video, lengths)   # (B, T, 138)

            # Flatten for loss computation
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            labels_flat = labels.reshape(B * T)

            loss = criterion(logits_flat, labels_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Tab accuracy (exclude padded positions)
            preds = logits.argmax(dim=-1)   # (B, T)
            mask = labels != PAD_LABEL
            total_correct += ((preds == labels) & mask).sum().item()
            total_notes += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_notes, 1)
    return avg_loss, accuracy


def _get_curriculum_params(epoch: int, warmup: int, max_noise: float, max_dropout: float):
    """
    Compute curriculum training parameters for the current epoch.

    Phase 1 (epoch < warmup): clean video features (no noise, no dropout)
    Phase 2 (epoch >= warmup): linearly ramp noise and dropout
    """
    if epoch < warmup:
        return 0.0, 0.0

    # Linear ramp from 0 to max over 20 epochs after warmup
    ramp_epochs = 20
    progress = min((epoch - warmup) / ramp_epochs, 1.0)
    noise = max_noise * progress
    dropout = max_dropout * progress
    return noise, dropout


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("P12 Fusion Model — Training")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Batch size : {P12_BATCH_SIZE}")
    print(f"  Max epochs : {P12_EPOCHS}")
    print(f"  LR         : {P12_LR}")
    print(f"  Patience   : {P12_PATIENCE}")
    print(f"  Curriculum warmup : {P12_CURRICULUM_WARMUP} epochs")

    # ── Load datasets ─────────────────────────────────────────────────────
    print("\nLoading datasets ...")
    train_ds = FusionDataset("train", video_noise_std=0.0, video_dropout=0.0, seed=42)
    val_ds = FusionDataset("val", video_noise_std=0.0, video_dropout=0.0, seed=123)

    if len(train_ds) == 0:
        print("❌ No training recordings found. Check GuitarSet path.")
        return
    if len(val_ds) == 0:
        print("❌ No validation recordings found. Check GuitarSet path.")
        return

    # ── Build model ───────────────────────────────────────────────────────
    model = FusionModel()
    model.to(DEVICE)
    print(f"\nModel parameters: {model.num_parameters:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=P12_LR, weight_decay=P12_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    checkpoint_path = str(FUSION_MODEL_PATH)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Phase':>8}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VlLoss':>8}  {'VlAcc':>7}  {'LR':>10}  {'Time':>6}")
    print("─" * 72)

    for epoch in range(P12_EPOCHS):
        t0 = time.time()

        # ── Curriculum: update video noise level ──────────────────────────
        noise, dropout = _get_curriculum_params(
            epoch, P12_CURRICULUM_WARMUP, P12_VIDEO_NOISE_STD, P12_VIDEO_DROPOUT,
        )
        phase = "clean" if epoch < P12_CURRICULUM_WARMUP else "noisy"
        train_ds.update_video_noise(noise, dropout)

        # ── Build data loaders (new each epoch for curriculum updates) ────
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_ds, batch_size=P12_BATCH_SIZE, shuffle=True,
            collate_fn=fusion_collate_fn, pin_memory=(DEVICE == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=P12_BATCH_SIZE, shuffle=False,
            collate_fn=fusion_collate_fn, pin_memory=(DEVICE == "cuda"),
        )

        # ── Train epoch ──────────────────────────────────────────────────
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, DEVICE,
            optimizer=optimizer, grad_clip=P12_GRAD_CLIP, is_train=True,
        )

        # ── Validate epoch ───────────────────────────────────────────────
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, DEVICE,
            is_train=False,
        )

        # ── LR scheduler ────────────────────────────────────────────────
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        print(f"{epoch + 1:5d}  {phase:>8s}  {train_loss:8.4f}  {train_acc:6.1%}  "
              f"{val_loss:8.4f}  {val_acc:6.1%}  {current_lr:10.2e}  {elapsed:5.1f}s")

        # ── History ──────────────────────────────────────────────────────
        history.append({
            "epoch": epoch + 1,
            "phase": phase,
            "video_noise": round(noise, 3),
            "video_dropout": round(dropout, 3),
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": current_lr,
        })

        # ── Early stopping ───────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save(checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= P12_PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {P12_PATIENCE} epochs)")
                break

    # ── Save training log ─────────────────────────────────────────────────
    log_path = str(FUSION_TRAINING_LOG)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining log saved → {log_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Best validation Tab Accuracy: {best_val_acc:.1%}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total epochs: {len(history)}")
    print("=" * 60)


if __name__ == "__main__":
    train()
