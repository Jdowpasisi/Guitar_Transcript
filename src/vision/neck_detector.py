"""
P10 — Neck Detector (Part A)
==============================
YOLOv8n fine-tuning pipeline for automatic guitar neck detection.

Replaces P9's manual corner clicking with an ML-based detector:
  video frame → YOLOv8 → bounding box → estimate 4 fretboard corners

The detected bounding box is converted to 4 corner points (TL/TR/BR/BL)
that can be fed directly into P9's compute_homography() for warping.

Usage:
    # Train on synthetic or labeled data
    python -m src.vision.neck_detector --train

    # Detect neck in a single frame
    python -m src.vision.neck_detector --detect frame.png

    # Evaluate on validation set
    python -m src.vision.neck_detector --evaluate

    # Detect + warp (replaces manual P9 workflow)
    python -m src.vision.neck_detector --detect frame.png --warp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import (
    DEVICE,
    MODELS_DIR,
    NECK_DATASET_DIR,
    NECK_MODEL_PATH,
    OUTPUTS_DIR,
    WARP_H,
    WARP_W,
)


# ══════════════════════════════════════════════════════════════════════════════
# Neck Detector Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class NeckDetector:
    """
    Wraps a fine-tuned YOLOv8 model for guitar neck detection.

    The detector finds the guitar neck bounding box in a frame and converts
    it to 4 fretboard corner points compatible with P9's homography pipeline.

    Attributes:
        model: The loaded YOLO model.
        conf_threshold: Minimum confidence to accept a detection.
    """

    def __init__(
        self,
        model_path: str | Path = NECK_MODEL_PATH,
        conf_threshold: float = 0.5,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for NeckDetector. "
                "Install with: pip install ultralytics"
            )
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold

        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
            print(f"  [NeckDetector] Loaded model from {self.model_path}")
        else:
            # Load pretrained yolov8n as fallback (untrained for neck)
            self.model = YOLO("yolov8n.pt")
            print(f"  [NeckDetector] ⚠ No fine-tuned model at {self.model_path}")
            print(f"  [NeckDetector]   Using pretrained yolov8n (run --train first)")

    def detect(
        self,
        frame: np.ndarray,
        return_annotated: bool = False,
    ) -> dict:
        """
        Detect the guitar neck in a single frame.

        Args:
            frame: BGR uint8 image.
            return_annotated: If True, includes the annotated frame in the result.

        Returns:
            dict with keys:
                detected: bool — whether a neck was found
                bbox: [x1, y1, x2, y2] — pixel coordinates (or None)
                confidence: float (or 0.0)
                corners: [[x,y]×4] — TL/TR/BR/BL corner estimates (or None)
                annotated: np.ndarray (only if return_annotated=True)
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)

        result = {
            "detected": False,
            "bbox": None,
            "confidence": 0.0,
            "corners": None,
        }

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Take the highest-confidence detection
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            best_conf = float(boxes.conf[best_idx].cpu().item())

            result["detected"] = True
            result["bbox"] = best_box  # [x1, y1, x2, y2]
            result["confidence"] = best_conf
            result["corners"] = self._bbox_to_corners(best_box)

        if return_annotated:
            result["annotated"] = self._annotate_frame(
                frame.copy(), result["bbox"], result["confidence"]
            )

        return result

    def detect_batch(
        self,
        frames_dir: str | Path,
        pattern: str = "frame_*.png",
    ) -> list[dict]:
        """Detect neck in all frames in a directory."""
        frames_dir = Path(frames_dir)
        frames = sorted(frames_dir.glob(pattern))
        if not frames:
            print(f"  ⚠ No frames matching '{pattern}' in {frames_dir}")
            return []

        results = []
        detected_count = 0
        for i, fp in enumerate(frames):
            frame = cv2.imread(str(fp))
            if frame is None:
                results.append({"error": f"Cannot load {fp}", "file": str(fp)})
                continue

            r = self.detect(frame)
            r["file"] = str(fp)
            results.append(r)
            if r["detected"]:
                detected_count += 1

            if i % 20 == 0 or i == len(frames) - 1:
                print(f"  [{i+1}/{len(frames)}] detected={detected_count}")

        pct = detected_count / len(frames) * 100 if frames else 0
        print(f"\n  Neck detected in {detected_count}/{len(frames)} frames ({pct:.0f}%)")
        return results

    @staticmethod
    def _bbox_to_corners(bbox: list[int]) -> list[list[int]]:
        """
        Convert a bounding box [x1, y1, x2, y2] to 4 fretboard corners
        in TL → TR → BR → BL order, compatible with P9's homography.

        For a horizontal guitar neck, the bbox IS the fretboard rectangle,
        so corners are simply the 4 corners of the bbox.
        """
        x1, y1, x2, y2 = bbox
        return [
            [x1, y1],  # TL
            [x2, y1],  # TR
            [x2, y2],  # BR
            [x1, y2],  # BL
        ]

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        bbox: list[int] | None,
        confidence: float,
    ) -> np.ndarray:
        """Draw detection box and label on frame."""
        if bbox is None:
            cv2.putText(frame, "No neck detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        x1, y1, x2, y2 = bbox
        # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label
        label = f"guitar_neck {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Corner dots
        corners = NeckDetector._bbox_to_corners(bbox)
        for i, (cx, cy) in enumerate(corners):
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (cx + 7, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return frame


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_neck_detector(
    dataset_yaml: str | Path | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str | Path | None = None,
    name: str = "neck_detector",
) -> Path:
    """
    Fine-tune YOLOv8n on the neck detection dataset.

    Args:
        dataset_yaml: Path to dataset.yaml. Defaults to NECK_DATASET_DIR / dataset.yaml.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size.
        project: Output project directory.
        name: Experiment name.

    Returns:
        Path to the best model weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")

    if dataset_yaml is None:
        dataset_yaml = NECK_DATASET_DIR / "dataset.yaml"
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_yaml}\n"
            f"Run: python -m src.vision.generate_training_data --neck 120"
        )

    if project is None:
        project = str(OUTPUTS_DIR / "yolo_training")

    print("=" * 55)
    print("  P10 — YOLOv8n Neck Detector Training")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs:  {epochs}")
    print(f"  ImgSize: {imgsz}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {DEVICE}")
    print("=" * 55)

    # Load pretrained yolov8n
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE if DEVICE != "mps" else "cpu",  # MPS can be flaky with YOLO
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        patience=15,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
        # Save best weights
        save=True,
    )

    # Copy best weights to standard model path
    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        import shutil
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_path), str(NECK_MODEL_PATH))
        print(f"\n✅ Best model saved → {NECK_MODEL_PATH}")
    else:
        print(f"\n⚠ best.pt not found at {best_path}")

    return NECK_MODEL_PATH


def evaluate_neck_detector(
    model_path: str | Path = NECK_MODEL_PATH,
    dataset_yaml: str | Path | None = None,
) -> dict:
    """
    Evaluate the neck detector on the validation set.

    Returns:
        dict with mAP50, mAP50-95, precision, recall.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")

    if dataset_yaml is None:
        dataset_yaml = NECK_DATASET_DIR / "dataset.yaml"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python -m src.vision.neck_detector --train"
        )

    print("=" * 55)
    print("  P10 — Neck Detector Evaluation")
    print(f"  Model:   {model_path}")
    print(f"  Dataset: {dataset_yaml}")
    print("=" * 55)

    model = YOLO(str(model_path))
    metrics = model.val(data=str(dataset_yaml), verbose=True)

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print(f"\n  mAP@50:    {results['mAP50']:.4f}")
    print(f"  mAP@50-95: {results['mAP50_95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")

    # Save results
    eval_path = OUTPUTS_DIR / "neck_detector_eval.json"
    eval_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {eval_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P10 Part A: YOLOv8 Guitar Neck Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true",
                        help="Train the neck detector on synthetic/labeled data")
    group.add_argument("--detect", type=str, metavar="FRAME",
                        help="Detect neck in a single frame image")
    group.add_argument("--evaluate", action="store_true",
                        help="Evaluate neck detector on validation set")
    group.add_argument("--batch_detect", type=str, metavar="DIR",
                        help="Detect neck in all frames in a directory")

    parser.add_argument("--warp", action="store_true",
                        help="Also warp the detected fretboard (with --detect)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for warped/annotated images")
    args = parser.parse_args()

    try:
        if args.train:
            train_neck_detector(epochs=args.epochs)

        elif args.detect:
            frame_path = Path(args.detect)
            if not frame_path.exists():
                print(f"❌ Frame not found: {frame_path}", file=sys.stderr)
                sys.exit(1)

            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"❌ Cannot load image: {frame_path}", file=sys.stderr)
                sys.exit(1)

            detector = NeckDetector()
            result = detector.detect(frame, return_annotated=True)

            if result["detected"]:
                print(f"\n✅ Neck detected!")
                print(f"   BBox:       {result['bbox']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Corners:    {result['corners']}")

                # Save annotated frame
                out_dir = Path(args.output_dir) if args.output_dir else frame_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                ann_path = out_dir / f"{frame_path.stem}_detected.png"
                cv2.imwrite(str(ann_path), result["annotated"])
                print(f"   Annotated:  {ann_path}")

                # Optionally warp
                if args.warp:
                    from src.vision.warp_fretboard import compute_homography, warp_frame
                    corners = np.array(result["corners"], dtype=np.float32)
                    H = compute_homography(corners)
                    warped = warp_frame(frame, H)
                    warp_path = out_dir / f"{frame_path.stem}_warped.png"
                    cv2.imwrite(str(warp_path), warped)
                    print(f"   Warped:     {warp_path}")
            else:
                print("\n⚠ No neck detected in this frame.")

        elif args.evaluate:
            evaluate_neck_detector()

        elif args.batch_detect:
            detector = NeckDetector()
            detector.detect_batch(args.batch_detect)

    except (FileNotFoundError, ImportError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
