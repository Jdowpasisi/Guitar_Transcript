"""
P10 — Chord Shape CNN (Part B)
================================
3-layer CNN that classifies chord fingerings from warped fretboard images.

Architecture:
    Input:  (B, 3, 64, 200) — RGB warped fretboard, resized to 64×200
    Block 1: Conv(3→32)  + BN + ReLU + MaxPool → (32, 32, 100)
    Block 2: Conv(32→64) + BN + ReLU + MaxPool → (64, 16, 50)
    Block 3: Conv(64→128)+ BN + ReLU + MaxPool → (128, 8, 25)
    GAP                                        → (128,)
    FC(128→64) + ReLU + Dropout(0.4)
    FC(64→7)
    Output: 7 classes — C, Am, G, Em, D, F, none

This file is self-contained: dataset, model, training loop, evaluation,
and comparison to audio chord predictions — all in one module.

Usage:
    python -m src.vision.chord_shape_cnn --train       # Train on dataset
    python -m src.vision.chord_shape_cnn --evaluate    # Evaluate on val/test
    python -m src.vision.chord_shape_cnn --predict frame.png  # Single image prediction
    python -m src.vision.chord_shape_cnn --summary     # Print model summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CHORD_DATASET_DIR,
    CHORD_INPUT_H,
    CHORD_INPUT_W,
    CHORD_SHAPE_CLASSES,
    CHORD_SHAPE_MODEL,
    DEVICE,
    MODELS_DIR,
    NUM_CHORD_SHAPES,
    OUTPUTS_DIR,
    P10_BATCH_SIZE,
    P10_DROPOUT,
    P10_EPOCHS,
    P10_LR,
    P10_PATIENCE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ChordShapeDataset(Dataset):
    """
    Loads warped fretboard images and their chord labels for training/evaluation.

    Reads images from the directory structure created by generate_training_data.py:
        chord_dataset/images/train/  or  chord_dataset/images/val/

    Each image is a 64×200 warped fretboard PNG. Labels come from labels.csv.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str | Path = CHORD_DATASET_DIR,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == "train")

        # Load labels from CSV
        csv_path = self.data_dir / "labels.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {csv_path}\n"
                f"Run: python -m src.vision.generate_training_data"
            )

        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    img_path = self.data_dir / row["filename"]
                    self.samples.append({
                        "path": img_path,
                        "chord": row["chord"],
                        "label": int(row["label"]),
                    })

        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' in {csv_path}")

        # Class map
        class_map_path = self.data_dir / "class_map.json"
        if class_map_path.exists():
            with open(class_map_path) as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {name: i for i, name in enumerate(CHORD_SHAPE_CLASSES)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        if img is None:
            # Fallback: black image
            img = np.zeros((CHORD_INPUT_H, CHORD_INPUT_W, 3), dtype=np.uint8)

        # Ensure correct size
        if img.shape[:2] != (CHORD_INPUT_H, CHORD_INPUT_W):
            img = cv2.resize(img, (CHORD_INPUT_W, CHORD_INPUT_H))

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augmentations (training only)
        if self.augment:
            img = self._augment(img)

        # To tensor: (H, W, C) → (C, H, W), normalize to [0, 1]
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return tensor, sample["label"]

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """Light augmentation for training robustness."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.fliplr(img).copy()

        # Random brightness
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() < 0.3:
            sigma = np.random.uniform(3, 12)
            noise = np.random.randn(*img.shape) * sigma
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random slight rotation (±3°)
        if np.random.random() < 0.2:
            angle = np.random.uniform(-3, 3)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(30, 30, 30))

        return img


# ══════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════════════

class ChordShapeCNN(nn.Module):
    """
    3-layer CNN for chord shape classification from warped fretboard images.

    Input:  (B, 3, 64, 200)  — RGB fretboard image
    Output: (B, num_classes)  — raw logits (use CrossEntropyLoss)

    Architecture:
        Block 1:  Conv(3→32, k=3)  + BN + ReLU + MaxPool(2)  → (32, 32, 100)
        Block 2:  Conv(32→64, k=3) + BN + ReLU + MaxPool(2)  → (64, 16, 50)
        Block 3:  Conv(64→128, k=3)+ BN + ReLU + MaxPool(2)  → (128, 8, 25)
        GAP:      AdaptiveAvgPool2d(1)                        → (128, 1, 1)
        Flatten                                               → (128,)
        FC1:      128→64 + ReLU + Dropout
        FC2:      64→num_classes

    Total parameters: ~180K (lightweight — designed for small datasets).
    """

    def __init__(self, num_classes: int = NUM_CHORD_SHAPES, dropout: float = P10_DROPOUT):
        super().__init__()
        self.num_classes = num_classes

        # Conv blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 64, 200) float tensor, pixel values in [0, 1].

        Returns:
            logits: (B, num_classes) raw scores.
        """
        x = self.block1(x)             # (B, 32, 32, 100)
        x = self.block2(x)             # (B, 64, 16, 50)
        x = self.block3(x)             # (B, 128, 8, 25)
        x = self.gap(x)                # (B, 128, 1, 1)
        x = self.flatten(x)            # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))   # (B, 64)
        x = self.fc2(x)                # (B, num_classes)
        return x

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns (predicted_class, confidence).
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        return predicted, confidence

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full probability distribution."""
        return F.softmax(self.forward(x), dim=-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = CHORD_SHAPE_MODEL):
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "num_classes": self.num_classes,
            "class_names": CHORD_SHAPE_CLASSES[:self.num_classes],
            "state_dict": self.state_dict(),
        }, str(path))
        print(f"  [ChordShapeCNN] Saved → {path}")

    @classmethod
    def load(
        cls,
        path: str | Path = CHORD_SHAPE_MODEL,
        device: str = "cpu",
    ) -> "ChordShapeCNN":
        """Load a saved checkpoint."""
        ckpt = torch.load(str(path), map_location=device)
        model = cls(num_classes=ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    is_training: bool,
) -> tuple[float, float]:
    """
    Run one epoch of training or validation.
    Returns (avg_loss, accuracy_percent).
    """
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_chord_shape(
    data_dir: str | Path = CHORD_DATASET_DIR,
    epochs: int = P10_EPOCHS,
    batch_size: int = P10_BATCH_SIZE,
    lr: float = P10_LR,
    patience: int = P10_PATIENCE,
    device: str = DEVICE,
) -> dict:
    """
    Train the ChordShapeCNN on the chord shape dataset.

    Returns:
        dict with training history and best metrics.
    """
    print("=" * 55)
    print("  P10 — Chord Shape CNN Training")
    print(f"  Device:     {device.upper()}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print("=" * 55)

    # Datasets
    train_ds = ChordShapeDataset("train", data_dir, augment=True)
    val_ds = ChordShapeDataset("val", data_dir, augment=False)

    print(f"\n📦 Train samples: {len(train_ds)}")
    print(f"   Val   samples: {len(val_ds)}")
    print(f"   Classes:       {NUM_CHORD_SHAPES} ({CHORD_SHAPE_CLASSES})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    # Model, loss, optimizer
    model = ChordShapeCNN(num_classes=NUM_CHORD_SHAPES, dropout=P10_DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2,
    )

    print(f"\n🧠 ChordShapeCNN: {model.num_parameters:,} parameters\n")

    # Training loop
    best_val_acc = 0.0
    patience_cnt = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, is_training=False
        )
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            model.save()
            print(f"  ⭐ New best val_acc={best_val_acc:.1f}% — checkpoint saved")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 3),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 3),
            "lr": current_lr,
        })

    # Save training log
    log_path = MODELS_DIR / "chord_shape_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n📈 Training log saved → {log_path}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.1f}%")

    return {"history": history, "best_val_acc": best_val_acc}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_chord_shape(
    model_path: str | Path = CHORD_SHAPE_MODEL,
    data_dir: str | Path = CHORD_DATASET_DIR,
    split: str = "val",
    device: str = DEVICE,
) -> dict:
    """
    Evaluate the ChordShapeCNN on the specified split.

    Returns per-class accuracy and overall accuracy.
    """
    print("=" * 55)
    print("  P10 — Chord Shape CNN Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Split: {split}")
    print("=" * 55)

    model = ChordShapeCNN.load(str(model_path), device=device).to(device)
    model.eval()

    dataset = ChordShapeDataset(split, data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=P10_BATCH_SIZE * 2, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(CHORD_SHAPE_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean() * 100
            per_class[name] = {
                "accuracy": round(acc, 1),
                "count": int(mask.sum()),
                "correct": int((all_preds[mask] == i).sum()),
            }

    # Confusion matrix
    n_classes = len(CHORD_SHAPE_CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(all_preds, all_labels):
        confusion[true][pred] += 1

    results = {
        "overall_accuracy": round(overall_acc, 2),
        "num_samples": len(all_labels),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": CHORD_SHAPE_CLASSES,
    }

    # Print results
    print(f"\n  Overall accuracy: {overall_acc:.1f}%  ({len(all_labels)} samples)\n")
    print(f"  {'Chord':<8} {'Accuracy':>8}  {'Correct':>7} / {'Total':>5}")
    print(f"  {'─'*35}")
    for name, info in per_class.items():
        bar = "█" * int(info["accuracy"] / 5) + "░" * (20 - int(info["accuracy"] / 5))
        print(f"  {name:<8} {info['accuracy']:>7.1f}%  {info['correct']:>7} / {info['count']:>5}  {bar}")

    # Save results
    eval_path = OUTPUTS_DIR / "chord_shape_eval.json"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        # Convert numpy types for JSON
        json.dump(results, f, indent=2, default=int)
    print(f"\n  Results saved → {eval_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Single-Image Prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_chord_from_image(
    image_path: str | Path,
    model_path: str | Path = CHORD_SHAPE_MODEL,
    device: str = DEVICE,
) -> dict:
    """
    Predict the chord shape from a single warped fretboard image.

    Args:
        image_path: Path to a fretboard image (warped 600×200 or 200×64).
        model_path: Path to the trained model checkpoint.

    Returns:
        dict with predicted chord, confidence, and all class probabilities.
    """
    model = ChordShapeCNN.load(str(model_path), device=device).to(device)
    model.eval()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Resize to model input
    img = cv2.resize(img, (CHORD_INPUT_W, CHORD_INPUT_H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # To tensor
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 64, 200)

    with torch.no_grad():
        probs = model.predict_proba(tensor)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_name = CHORD_SHAPE_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "chord": pred_name,
        "confidence": round(confidence, 4),
        "label_index": pred_idx,
        "probabilities": {
            name: round(float(probs[i]), 4)
            for i, name in enumerate(CHORD_SHAPE_CLASSES)
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P10 Part B: Chord Shape CNN — classify chord fingerings from fretboard images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true",
                        help="Train the ChordShapeCNN")
    group.add_argument("--evaluate", action="store_true",
                        help="Evaluate on validation set")
    group.add_argument("--predict", type=str, metavar="IMAGE",
                        help="Predict chord from a single fretboard image")
    group.add_argument("--summary", action="store_true",
                        help="Print model architecture summary")

    parser.add_argument("--epochs", type=int, default=P10_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=P10_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=P10_LR)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Split for evaluation")
    args = parser.parse_args()

    try:
        if args.train:
            train_chord_shape(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

        elif args.evaluate:
            evaluate_chord_shape(split=args.split)

        elif args.predict:
            result = predict_chord_from_image(args.predict)
            print(f"\n🎸 Predicted chord: {result['chord']}")
            print(f"   Confidence:      {result['confidence']:.1%}")
            print(f"\n   All probabilities:")
            for name, prob in sorted(result["probabilities"].items(),
                                     key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 40)
                print(f"     {name:<5s} {prob:>6.1%}  {bar}")

        elif args.summary:
            model = ChordShapeCNN()
            print(f"\nChordShapeCNN (P10)")
            print(f"  Classes:    {model.num_classes} ({CHORD_SHAPE_CLASSES})")
            print(f"  Parameters: {model.num_parameters:,}")
            print(f"  Input:      (B, 3, {CHORD_INPUT_H}, {CHORD_INPUT_W})")
            B = 2
            x = torch.randn(B, 3, CHORD_INPUT_H, CHORD_INPUT_W)
            out = model(x)
            print(f"  Output:     {tuple(out.shape)}")
            print(f"\n  Architecture:")
            print(f"    Block 1: Conv(3→32)   + BN + ReLU + MaxPool(2)")
            print(f"    Block 2: Conv(32→64)  + BN + ReLU + MaxPool(2)")
            print(f"    Block 3: Conv(64→128) + BN + ReLU + MaxPool(2)")
            print(f"    GAP → FC(128→64) + ReLU + Dropout → FC(64→{model.num_classes})")
            print(f"\n  ✅ Forward pass OK. Output shape: {tuple(out.shape)}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
