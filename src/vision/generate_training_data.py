"""
P10 — Synthetic Training Data Generator
========================================
Generates two datasets for the Guitar Vision Model:

  1. **Neck detection** (YOLO format):
     - Synthetic frames with guitar-neck-like rectangles rendered at
       various positions, scales, and angles on textured backgrounds.
     - Outputs: images/ + labels/ with YOLO-format annotation per image.

  2. **Chord shape classification** (image + CSV):
     - Warped 600×200 fretboard images with finger-dot patterns
       representing the 6 target open chords + "none".
     - Outputs: images per class + labels.csv.

Both datasets include data augmentation (brightness/contrast jitter,
Gaussian noise, slight rotation) to improve model robustness.

Usage:
    python -m src.vision.generate_training_data [--neck N] [--chords N]
    python -m src.vision.generate_training_data              # defaults: 120 neck, 280 chord
    python -m src.vision.generate_training_data --neck 80    # 80 neck images only
    python -m src.vision.generate_training_data --chords 200 # 200 chord images only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    CHORD_DATASET_DIR,
    CHORD_INPUT_H,
    CHORD_INPUT_W,
    CHORD_SHAPE_CLASSES,
    NECK_DATASET_DIR,
    VISION_DATASET_DIR,
    WARP_H,
    WARP_W,
)


# ══════════════════════════════════════════════════════════════════════════════
# Chord finger patterns — (string_index, fret) for each chord
# string 0 = low E (bottom of image), string 5 = high E (top of image)
# Muted strings are not rendered.
# ══════════════════════════════════════════════════════════════════════════════

CHORD_PATTERNS = {
    "C":  [(1, 3), (2, 2), (4, 1)],               # x32010
    "Am": [(1, 0), (2, 2), (3, 2), (4, 1)],       # x02210
    "G":  [(0, 3), (1, 2), (5, 3)],               # 320003
    "Em": [(1, 2), (2, 2)],                        # 022000
    "D":  [(3, 2), (4, 3), (5, 2)],               # xx0232
    "F":  [(0, 1), (1, 1), (2, 2), (3, 3), (4, 1), (5, 1)],  # 133211 (barre)
}


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _random_brightness(img: np.ndarray, low: float = 0.6, high: float = 1.4) -> np.ndarray:
    """Scale pixel intensity randomly."""
    factor = random.uniform(low, high)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _add_gaussian_noise(img: np.ndarray, sigma_range: tuple = (5, 25)) -> np.ndarray:
    """Add Gaussian noise."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _random_blur(img: np.ndarray, max_k: int = 3) -> np.ndarray:
    """Apply slight Gaussian blur."""
    if random.random() < 0.4:
        k = random.choice([1, 3])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def _textured_background(h: int, w: int) -> np.ndarray:
    """Create a varied background — not just flat black."""
    bg_type = random.choice(["dark", "gradient", "noisy", "wood"])
    if bg_type == "dark":
        val = random.randint(10, 50)
        bg = np.full((h, w, 3), val, dtype=np.uint8)
    elif bg_type == "gradient":
        base = random.randint(20, 80)
        grad = np.linspace(base, base + random.randint(20, 60), h).astype(np.uint8)
        bg = np.stack([grad] * w, axis=1)
        bg = np.stack([bg] * 3, axis=2)
    elif bg_type == "noisy":
        bg = np.random.randint(10, 60, (h, w, 3), dtype=np.uint8)
    else:  # "wood" — brownish gradient
        r = np.random.randint(40, 90)
        g = np.random.randint(30, 60)
        b = np.random.randint(10, 40)
        bg = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Neck Detection Dataset (YOLO format)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_neck(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    n_frets: int = 12,
    n_strings: int = 6,
) -> np.ndarray:
    """Draw a fretboard rectangle with strings and frets inside."""
    # Neck body (wood colour with variation)
    wood_b = random.randint(30, 70)
    wood_g = random.randint(50, 100)
    wood_r = random.randint(80, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (wood_b, wood_g, wood_r), -1)

    # Neck edge lines
    edge_col = (wood_b + 30, wood_g + 20, wood_r + 20)
    cv2.rectangle(frame, (x1, y1), (x2, y2), edge_col, 2)

    neck_w = x2 - x1
    neck_h = y2 - y1

    # Frets (vertical lines within the neck)
    for i in range(1, n_frets + 1):
        ratio = 1 - (1 / (2 ** (i / 12)))
        fx = x1 + int(neck_w * ratio * 0.95)
        fret_col = (180 + random.randint(-20, 20),) * 3
        cv2.line(frame, (fx, y1), (fx, y2), fret_col, 1)

    # Strings (horizontal lines within the neck)
    for i in range(n_strings):
        sy = y1 + int(neck_h * (i + 0.5) / n_strings)
        brightness = 140 + i * 10 + random.randint(-10, 10)
        cv2.line(frame, (x1, sy), (x2, sy), (brightness, brightness, brightness), 1)

    # Inlay dots
    inlays = [3, 5, 7, 9]
    mid_y = (y1 + y2) // 2
    for fret_n in inlays:
        r1 = 1 - (1 / (2 ** (fret_n / 12)))
        r0 = 1 - (1 / (2 ** ((fret_n - 1) / 12)))
        ix = x1 + int(neck_w * (r1 + r0) / 2 * 0.95)
        dot_r = max(2, neck_h // 30)
        cv2.circle(frame, (ix, mid_y), dot_r, (180, 160, 100), -1)

    return frame


def generate_neck_dataset(
    n_images: int = 120,
    img_size: int = 640,
    output_dir: Path | None = None,
    train_ratio: float = 0.8,
) -> dict:
    """
    Generate synthetic neck detection images in YOLO format.

    Output structure:
        output_dir/
          images/train/  images/val/
          labels/train/  labels/val/
          dataset.yaml
    """
    if output_dir is None:
        output_dir = NECK_DATASET_DIR
    output_dir = Path(output_dir)

    # Clean and create directories
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    n_train = int(n_images * train_ratio)
    stats = {"train": 0, "val": 0, "total": n_images}

    for i in range(n_images):
        split = "train" if i < n_train else "val"
        bg = _textured_background(img_size, img_size)

        # Random neck position and size
        neck_w = random.randint(int(img_size * 0.35), int(img_size * 0.85))
        neck_h = random.randint(int(img_size * 0.12), int(img_size * 0.35))

        max_x = img_size - neck_w - 10
        max_y = img_size - neck_h - 10
        x1 = random.randint(10, max(11, max_x))
        y1 = random.randint(10, max(11, max_y))
        x2 = x1 + neck_w
        y2 = y1 + neck_h

        frame = _draw_neck(bg, x1, y1, x2, y2)

        # Augmentations
        frame = _random_brightness(frame)
        frame = _add_gaussian_noise(frame)
        frame = _random_blur(frame)

        # Optional slight rotation (±5°)
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), angle, 1.0)
            frame = cv2.warpAffine(frame, M, (img_size, img_size),
                                   borderValue=(30, 30, 30))

        # Save image
        img_name = f"neck_{i:04d}.jpg"
        cv2.imwrite(str(output_dir / f"images/{split}/{img_name}"), frame)

        # YOLO label: class x_center y_center width height (normalized 0-1)
        x_center = ((x1 + x2) / 2) / img_size
        y_center = ((y1 + y2) / 2) / img_size
        w_norm = neck_w / img_size
        h_norm = neck_h / img_size

        label_name = f"neck_{i:04d}.txt"
        with open(output_dir / f"labels/{split}/{label_name}", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        stats[split] += 1

    # Create dataset.yaml for YOLO training
    yaml_content = f"""# P10 Neck Detection Dataset (synthetic)
# Auto-generated by generate_training_data.py

path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: guitar_neck

nc: 1
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)

    print(f"\n📦 Neck detection dataset generated → {output_dir}")
    print(f"   Train: {stats['train']} images")
    print(f"   Val:   {stats['val']} images")
    print(f"   YOLO config: {output_dir / 'dataset.yaml'}")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Chord Shape Dataset
# ══════════════════════════════════════════════════════════════════════════════

def _draw_chord_fretboard(
    chord_name: str,
    w: int = WARP_W,
    h: int = WARP_H,
) -> np.ndarray:
    """
    Render a warped fretboard image with finger dots for the given chord.
    Mimics what a real warped frame would look like from P9.
    """
    # Fretboard wood background
    wood_base = np.array([random.randint(30, 60),
                          random.randint(50, 90),
                          random.randint(80, 140)], dtype=np.uint8)
    frame = np.full((h, w, 3), wood_base, dtype=np.uint8)

    # Add wood grain texture
    for _ in range(random.randint(3, 8)):
        y_line = random.randint(0, h - 1)
        grain_col = tuple(int(c) + random.randint(-10, 10) for c in wood_base)
        grain_col = tuple(max(0, min(255, c)) for c in grain_col)
        cv2.line(frame, (0, y_line), (w, y_line), grain_col, 1)

    n_strings = 6
    n_frets = 12

    # Draw frets (vertical lines)
    for i in range(n_frets + 1):
        ratio = i / n_frets
        fx = int(w * ratio)
        fret_brightness = 170 + random.randint(-20, 20)
        cv2.line(frame, (fx, 0), (fx, h), (fret_brightness,) * 3, 1)

    # Draw strings (horizontal lines)
    string_ys = []
    for s in range(n_strings):
        sy = int(h * (s + 0.5) / n_strings)
        string_ys.append(sy)
        brightness = 130 + s * 8 + random.randint(-5, 5)
        thickness = 1 if s > 2 else 2  # bass strings thicker
        cv2.line(frame, (0, sy), (w, sy), (brightness, brightness, brightness), thickness)

    # Draw inlay dots
    inlay_frets = [3, 5, 7, 9]
    mid_y = h // 2
    for fret_n in inlay_frets:
        ix = int(w * (fret_n - 0.5) / n_frets)
        cv2.circle(frame, (ix, mid_y), 4, (180, 160, 100), -1)
    # Double dot at 12
    ix_12 = int(w * 11.5 / n_frets)
    cv2.circle(frame, (ix_12, mid_y - 15), 3, (180, 160, 100), -1)
    cv2.circle(frame, (ix_12, mid_y + 15), 3, (180, 160, 100), -1)

    # Draw finger dots for the chord (if not "none")
    if chord_name != "none" and chord_name in CHORD_PATTERNS:
        pattern = CHORD_PATTERNS[chord_name]
        for string_idx, fret in pattern:
            if fret == 0:
                continue  # open string — no finger dot
            # Finger position
            fx = int(w * (fret - 0.5) / n_frets)
            fy = string_ys[string_idx]

            # Slight random offset for realism
            fx += random.randint(-3, 3)
            fy += random.randint(-2, 2)

            # Fingertip dot (flesh colour with variation)
            dot_r = random.randint(6, 10)
            flesh_b = random.randint(100, 140)
            flesh_g = random.randint(120, 170)
            flesh_r = random.randint(160, 220)
            cv2.circle(frame, (fx, fy), dot_r, (flesh_b, flesh_g, flesh_r), -1)
            # Dark outline
            cv2.circle(frame, (fx, fy), dot_r, (40, 40, 40), 1)

    return frame


def generate_chord_dataset(
    n_per_class: int = 40,
    output_dir: Path | None = None,
    train_ratio: float = 0.8,
) -> dict:
    """
    Generate synthetic chord classification images.

    Output structure:
        output_dir/
          images/train/<chord>_NNNN.png
          images/val/<chord>_NNNN.png
          labels.csv
          class_map.json
    """
    if output_dir is None:
        output_dir = CHORD_DATASET_DIR
    output_dir = Path(output_dir)

    for sub in ["images/train", "images/val"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    class_map = {name: i for i, name in enumerate(CHORD_SHAPE_CLASSES)}
    rows = []
    stats = {c: {"train": 0, "val": 0} for c in CHORD_SHAPE_CLASSES}

    for chord_name in CHORD_SHAPE_CLASSES:
        for j in range(n_per_class):
            split = "train" if j < int(n_per_class * train_ratio) else "val"

            # Generate base image at warp dimensions
            img = _draw_chord_fretboard(chord_name)

            # Augmentations
            img = _random_brightness(img, 0.7, 1.3)
            img = _add_gaussian_noise(img, (3, 15))
            img = _random_blur(img)

            # Slight perspective jitter (simulates imperfect homography)
            if random.random() < 0.3:
                pts1 = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])
                jitter = random.randint(2, 8)
                pts2 = pts1 + np.random.randint(-jitter, jitter + 1, pts1.shape).astype(np.float32)
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img = cv2.warpPerspective(img, M, (WARP_W, WARP_H),
                                          borderValue=(30, 30, 30))

            # Resize to CNN input dimensions
            img_resized = cv2.resize(img, (CHORD_INPUT_W, CHORD_INPUT_H))

            # Save
            img_name = f"{chord_name}_{j:04d}.png"
            cv2.imwrite(str(output_dir / f"images/{split}/{img_name}"), img_resized)

            rows.append({
                "filename": f"images/{split}/{img_name}",
                "split": split,
                "chord": chord_name,
                "label": class_map[chord_name],
            })
            stats[chord_name][split] += 1

    # Save labels CSV
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "chord", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Save class map
    (output_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))

    print(f"\n📦 Chord shape dataset generated → {output_dir}")
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    print(f"   Total: {total_train + total_val} images ({total_train} train, {total_val} val)")
    print(f"   Classes: {CHORD_SHAPE_CLASSES}")
    for chord, s in stats.items():
        print(f"     {chord:5s}: {s['train']} train, {s['val']} val")
    print(f"   Labels CSV: {csv_path}")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P10: Generate synthetic training data for Guitar Vision Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--neck", type=int, default=120,
                        help="Number of neck detection images to generate")
    parser.add_argument("--chords", type=int, default=40,
                        help="Number of chord images PER CLASS to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 55)
    print("  P10 — Synthetic Training Data Generator")
    print("=" * 55)

    generate_neck_dataset(n_images=args.neck)
    generate_chord_dataset(n_per_class=args.chords)

    print("\n" + "─" * 55)
    print("  ✅ All training data generated!")
    print("─" * 55)


if __name__ == "__main__":
    main()
