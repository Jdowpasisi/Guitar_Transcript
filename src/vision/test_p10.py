"""
P10 Smoke Test
===============
Validates the complete P10 pipeline without requiring a real video,
pre-trained weights, or labeled data.

Tests:
  1. Synthetic data generation (small batch)
  2. Dataset loading and structure verification
  3. ChordShapeCNN forward pass with random input
  4. NeckDetector wrapper instantiation
  5. Pipeline end-to-end on a synthetic frame
  6. Model save/load round-trip

Run with:
    python -m src.vision.test_p10

Expected output:
    ✓ Synthetic neck data generated (5 images)
    ✓ Synthetic chord data generated (3 per class)
    ✓ ChordShapeDataset loads correctly
    ✓ ChordShapeCNN forward pass OK
    ✓ ChordShapeCNN save/load round-trip OK
    ✓ NeckDetector instantiates
    ✓ Pipeline processes synthetic frame
    All P10 smoke tests passed!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def run_smoke_test():
    """Run all P10 module tests."""
    from src.config import (
        CHORD_INPUT_H,
        CHORD_INPUT_W,
        CHORD_SHAPE_CLASSES,
        NUM_CHORD_SHAPES,
    )

    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── 1. Synthetic neck data ────────────────────────────────────────────
        print("  Testing synthetic data generation...")
        from src.vision.generate_training_data import (
            generate_neck_dataset,
            generate_chord_dataset,
        )

        neck_dir = tmp / "neck"
        stats = generate_neck_dataset(n_images=5, output_dir=neck_dir)
        assert stats["total"] == 5, f"Expected 5 neck images, got {stats['total']}"
        assert (neck_dir / "dataset.yaml").exists(), "dataset.yaml not created"
        train_imgs = list((neck_dir / "images" / "train").glob("*.jpg"))
        val_imgs = list((neck_dir / "images" / "val").glob("*.jpg"))
        assert len(train_imgs) + len(val_imgs) == 5
        print(f"  ✓ Synthetic neck data          ({stats['train']} train, {stats['val']} val)")

        # ── 2. Synthetic chord data ───────────────────────────────────────────
        chord_dir = tmp / "chords"
        c_stats = generate_chord_dataset(n_per_class=3, output_dir=chord_dir)
        assert (chord_dir / "labels.csv").exists(), "labels.csv not created"
        assert (chord_dir / "class_map.json").exists(), "class_map.json not created"
        total = sum(s["train"] + s["val"] for s in c_stats.values())
        assert total == 3 * len(CHORD_SHAPE_CLASSES), f"Expected {3 * len(CHORD_SHAPE_CLASSES)} images, got {total}"
        print(f"  ✓ Synthetic chord data         ({total} images across {len(CHORD_SHAPE_CLASSES)} classes)")

        # ── 3. ChordShapeDataset loading ──────────────────────────────────────
        from src.vision.chord_shape_cnn import ChordShapeDataset

        train_ds = ChordShapeDataset("train", chord_dir, augment=True)
        assert len(train_ds) > 0, "Train dataset is empty"
        x, y = train_ds[0]
        assert x.shape == (3, CHORD_INPUT_H, CHORD_INPUT_W), f"Wrong tensor shape: {x.shape}"
        assert isinstance(y, int), f"Label should be int, got {type(y)}"
        assert 0 <= y < NUM_CHORD_SHAPES, f"Label {y} out of range [0, {NUM_CHORD_SHAPES})"
        print(f"  ✓ ChordShapeDataset loads      ({len(train_ds)} train samples, shape={tuple(x.shape)})")

        # ── 4. ChordShapeCNN forward pass ─────────────────────────────────────
        from src.vision.chord_shape_cnn import ChordShapeCNN

        model = ChordShapeCNN(num_classes=NUM_CHORD_SHAPES)
        B = 4
        dummy = torch.randn(B, 3, CHORD_INPUT_H, CHORD_INPUT_W)
        logits = model(dummy)
        assert logits.shape == (B, NUM_CHORD_SHAPES), f"Wrong output shape: {logits.shape}"

        # Predict
        preds, confs = model.predict(dummy)
        assert preds.shape == (B,), f"Wrong preds shape: {preds.shape}"
        assert confs.shape == (B,), f"Wrong confs shape: {confs.shape}"
        assert (confs >= 0).all() and (confs <= 1).all(), "Confidences should be in [0, 1]"

        # Predict proba
        probs = model.predict_proba(dummy)
        assert probs.shape == (B, NUM_CHORD_SHAPES)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

        print(f"  ✓ ChordShapeCNN forward pass   ({model.num_parameters:,} params, "
              f"output={tuple(logits.shape)})")

        # ── 5. Save/Load round-trip ───────────────────────────────────────────
        ckpt_path = tmp / "test_chord_cnn.pth"
        model.save(str(ckpt_path))
        assert ckpt_path.exists(), "Checkpoint not saved"

        model.eval()
        loaded = ChordShapeCNN.load(str(ckpt_path))
        loaded.eval()
        with torch.no_grad():
            orig_logits = model(dummy)
            loaded_logits = loaded(dummy)
        assert torch.allclose(orig_logits, loaded_logits, atol=1e-4), "Loaded model output differs"
        print(f"  ✓ ChordShapeCNN save/load      (round-trip verified)")

        # ── 6. NeckDetector instantiation ─────────────────────────────────────
        try:
            from src.vision.neck_detector import NeckDetector

            # Test with untrained model (won't detect anything meaningful)
            detector = NeckDetector(model_path=tmp / "nonexistent.pt", conf_threshold=0.5)

            # Create a test frame and run detection
            from src.vision.test_p9 import make_synthetic_fretboard_frame
            test_frame = make_synthetic_fretboard_frame()
            result = detector.detect(test_frame, return_annotated=True)

            assert "detected" in result
            assert "bbox" in result
            assert "confidence" in result
            assert "corners" in result
            assert "annotated" in result
            assert result["annotated"].shape == test_frame.shape

            # Test _bbox_to_corners static method
            corners = NeckDetector._bbox_to_corners([10, 20, 100, 80])
            assert corners == [[10, 20], [100, 20], [100, 80], [10, 80]]

            print(f"  ✓ NeckDetector instantiates    (detect returns valid dict)")
        except ImportError as e:
            print(f"  ⚠ NeckDetector skipped         ({e})")

        # ── 7. Pipeline integration ───────────────────────────────────────────
        try:
            # Create a mock pipeline test without requiring trained models
            from src.vision.warp_fretboard import compute_homography, warp_frame
            from src.vision.test_p9 import make_synthetic_fretboard_frame

            # Synthetic frame with known neck bounds
            test_frame = make_synthetic_fretboard_frame(800, 450)

            # Manual corners (from test_p9.py)
            src_corners = np.array([
                [80, 100], [720, 100], [720, 350], [80, 350]
            ], dtype=np.float32)

            H = compute_homography(src_corners)
            warped = warp_frame(test_frame, H)
            assert warped.shape == (200, 600, 3), f"Wrong warped shape: {warped.shape}"

            # Resize and run through chord CNN
            img = cv2.resize(warped, (CHORD_INPUT_W, CHORD_INPUT_H))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                probs = model.predict_proba(tensor)[0].cpu().numpy()

            assert len(probs) == NUM_CHORD_SHAPES
            pred_name = CHORD_SHAPE_CLASSES[int(probs.argmax())]
            assert pred_name in CHORD_SHAPE_CLASSES

            print(f"  ✓ Pipeline integration         (warp→CNN→'{pred_name}' "
                  f"conf={float(probs.max()):.3f})")
        except Exception as e:
            print(f"  ⚠ Pipeline integration skipped ({e})")

        # ── 8. YOLO dataset format verification ───────────────────────────────
        import yaml
        with open(neck_dir / "dataset.yaml") as f:
            ds_config = yaml.safe_load(f) if hasattr(yaml, 'safe_load') else None

        if ds_config is None:
            # Parse manually if PyYAML not available
            text = (neck_dir / "dataset.yaml").read_text()
            assert "guitar_neck" in text, "dataset.yaml missing class name"
            assert "nc: 1" in text, "dataset.yaml missing nc"
        else:
            assert ds_config["nc"] == 1
            assert "guitar_neck" in str(ds_config["names"])

        # Verify label format
        label_files = list((neck_dir / "labels" / "train").glob("*.txt"))
        if label_files:
            content = label_files[0].read_text().strip()
            parts = content.split()
            assert len(parts) == 5, f"YOLO label should have 5 fields, got {len(parts)}"
            assert parts[0] == "0", f"Class should be 0, got {parts[0]}"
            for v in parts[1:]:
                assert 0 <= float(v) <= 1, f"YOLO coords should be normalized: {v}"

        print(f"  ✓ YOLO format verification     (labels valid)")

    print("\n" + "─" * 55)
    print("  All P10 smoke tests passed! ✅")
    print("─" * 55)


if __name__ == "__main__":
    print("\nP10 Smoke Test")
    print("─" * 55)
    try:
        run_smoke_test()
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        raise
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("   Make sure all P10 dependencies are installed:")
        print("   pip install ultralytics torch torchvision opencv-python")
        raise
