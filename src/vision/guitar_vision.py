"""
P10 — Guitar Vision Pipeline
==============================
End-to-end pipeline: video → neck detection → warp → chord classification.

Ties together Part A (NeckDetector) and Part B (ChordShapeCNN) into
a unified inference pipeline that replaces P9's manual corner-clicking
workflow.

Flow:
    1. Extract frame from video (or load image)
    2. YOLOv8 detects the guitar neck → bounding box
    3. Convert bbox → 4 corners (TL/TR/BR/BL)
    4. Compute homography → warp fretboard to flat rectangle
    5. Resize warped image → Chord Shape CNN → chord label

Graceful fallback:
    - If no neck is detected in a frame: skip chord classification.
    - If models are not trained yet: provides clear error messages.

Usage:
    python -m src.vision.guitar_vision --frame frame.png
    python -m src.vision.guitar_vision --video guitar_video.mp4
    python -m src.vision.guitar_vision --video guitar_video.mp4 --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import (
    CHORD_INPUT_H,
    CHORD_INPUT_W,
    CHORD_SHAPE_CLASSES,
    CHORD_SHAPE_MODEL,
    DEVICE,
    NECK_MODEL_PATH,
    OUTPUTS_DIR,
    WARP_H,
    WARP_W,
)


class GuitarVisionPipeline:
    """
    End-to-end guitar vision pipeline: frame → neck → warp → chord.

    Combines the NeckDetector (Part A) and ChordShapeCNN (Part B) into
    a single inference pipeline. Designed to be used both from the CLI
    and programmatically from P13 (GuitarAI v1 assembly).

    Attributes:
        neck_detector: NeckDetector instance (YOLOv8).
        chord_cnn: ChordShapeCNN instance (loaded from checkpoint).
        device: Torch device for chord CNN inference.
    """

    def __init__(
        self,
        neck_model_path: str | Path = NECK_MODEL_PATH,
        chord_model_path: str | Path = CHORD_SHAPE_MODEL,
        device: str = DEVICE,
        neck_conf: float = 0.5,
    ):
        self.device = device

        # Load neck detector
        from src.vision.neck_detector import NeckDetector
        self.neck_detector = NeckDetector(
            model_path=neck_model_path,
            conf_threshold=neck_conf,
        )

        # Load chord shape CNN
        import torch
        from src.vision.chord_shape_cnn import ChordShapeCNN
        chord_model_path = Path(chord_model_path)
        if chord_model_path.exists():
            self.chord_cnn = ChordShapeCNN.load(str(chord_model_path), device=device)
            self.chord_cnn = self.chord_cnn.to(device)
            print(f"  [Pipeline] ChordShapeCNN loaded from {chord_model_path}")
        else:
            self.chord_cnn = None
            print(f"  [Pipeline] ⚠ No chord CNN at {chord_model_path}")
            print(f"  [Pipeline]   Run: python -m src.vision.chord_shape_cnn --train")

    def process_frame(
        self,
        frame: np.ndarray,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Process a single video frame through the full pipeline.

        Args:
            frame: BGR uint8 image.
            return_intermediates: If True, includes warped image and annotated frame.

        Returns:
            dict with:
                neck_detected:   bool
                neck_bbox:       [x1, y1, x2, y2] or None
                neck_confidence: float
                corners:         [[x,y]×4] or None
                chord_label:     str or None (e.g., "G", "Am", "none")
                chord_confidence: float or 0.0
                chord_probs:     dict{class_name: prob} or None
                warped:          np.ndarray (only if return_intermediates)
                annotated:       np.ndarray (only if return_intermediates)
        """
        import torch
        from src.vision.warp_fretboard import compute_homography, warp_frame

        # Step 1: Detect neck
        neck_result = self.neck_detector.detect(frame, return_annotated=return_intermediates)

        result = {
            "neck_detected": neck_result["detected"],
            "neck_bbox": neck_result["bbox"],
            "neck_confidence": neck_result["confidence"],
            "corners": neck_result["corners"],
            "chord_label": None,
            "chord_confidence": 0.0,
            "chord_probs": None,
        }

        if return_intermediates:
            result["annotated"] = neck_result.get("annotated")
            result["warped"] = None

        if not neck_result["detected"]:
            return result

        # Step 2: Warp fretboard
        try:
            corners = np.array(neck_result["corners"], dtype=np.float32)
            H = compute_homography(corners)
            warped = warp_frame(frame, H)
        except Exception as e:
            print(f"  ⚠ Warp failed: {e}")
            return result

        if return_intermediates:
            result["warped"] = warped

        # Step 3: Classify chord shape
        if self.chord_cnn is not None:
            try:
                # Resize to CNN input
                img = cv2.resize(warped, (CHORD_INPUT_W, CHORD_INPUT_H))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # To tensor
                tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                tensor = tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    probs = self.chord_cnn.predict_proba(tensor)[0].cpu().numpy()

                pred_idx = int(probs.argmax())
                result["chord_label"] = CHORD_SHAPE_CLASSES[pred_idx]
                result["chord_confidence"] = float(probs[pred_idx])
                result["chord_probs"] = {
                    name: float(probs[i])
                    for i, name in enumerate(CHORD_SHAPE_CLASSES)
                }
            except Exception as e:
                print(f"  ⚠ Chord classification failed: {e}")

        return result

    def process_video(
        self,
        video_path: str | Path,
        fps: int = 5,
        output_dir: str | Path | None = None,
        save_annotated: bool = True,
    ) -> list[dict]:
        """
        Process a full video through the pipeline.

        Args:
            video_path: Path to video file.
            fps: Frames per second to extract.
            output_dir: Where to save outputs. Defaults to outputs/vision/<video_stem>/.
            save_annotated: Whether to save annotated frames.

        Returns:
            List of per-frame result dicts.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is None:
            output_dir = OUTPUTS_DIR / "vision" / video_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎥 Processing video: {video_path.name}")
        print(f"   Output: {output_dir}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))

        print(f"   Video FPS: {video_fps:.1f}, extracting every {frame_interval} frames")

        results = []
        frame_idx = 0
        processed = 0
        detected = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                result = self.process_frame(frame, return_intermediates=save_annotated)
                result["frame_idx"] = frame_idx
                result["timestamp"] = frame_idx / video_fps

                if result["neck_detected"]:
                    detected += 1

                # Save annotated frame
                if save_annotated and result.get("annotated") is not None:
                    ann_path = output_dir / f"frame_{processed:04d}_ann.png"
                    cv2.imwrite(str(ann_path), result["annotated"])

                # Save warped frame
                if result.get("warped") is not None:
                    warp_path = output_dir / f"frame_{processed:04d}_warped.png"
                    cv2.imwrite(str(warp_path), result["warped"])

                # Remove numpy arrays from stored results (not JSON serializable)
                result.pop("annotated", None)
                result.pop("warped", None)

                results.append(result)
                processed += 1

                if processed % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"   [{processed} frames] "
                          f"detected={detected}, "
                          f"elapsed={elapsed:.1f}s")

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t0

        # Save results
        results_path = output_dir / "vision_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Summary
        detect_pct = detected / processed * 100 if processed > 0 else 0
        chord_counts = {}
        for r in results:
            if r["chord_label"]:
                chord_counts[r["chord_label"]] = chord_counts.get(r["chord_label"], 0) + 1

        summary = {
            "video": str(video_path),
            "total_video_frames": total_frames,
            "processed_frames": processed,
            "neck_detected": detected,
            "detection_rate": round(detect_pct, 1),
            "chord_distribution": chord_counts,
            "processing_time_sec": round(elapsed, 1),
            "fps_achieved": round(processed / elapsed, 1) if elapsed > 0 else 0,
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Video processing complete!")
        print(f"   Frames processed: {processed}")
        print(f"   Neck detected:    {detected}/{processed} ({detect_pct:.0f}%)")
        print(f"   Chord distribution: {chord_counts}")
        print(f"   Time: {elapsed:.1f}s ({summary['fps_achieved']:.1f} fps)")
        print(f"   Results: {results_path}")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P10: End-to-end Guitar Vision Pipeline — neck detection + chord classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", type=str, help="Process a single frame image")
    group.add_argument("--video", type=str, help="Process a full video")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=5,
                        help="Frame extraction rate for video")
    parser.add_argument("--neck_conf", type=float, default=0.5,
                        help="Minimum neck detection confidence")
    args = parser.parse_args()

    try:
        pipeline = GuitarVisionPipeline(neck_conf=args.neck_conf)

        if args.frame:
            frame_path = Path(args.frame)
            if not frame_path.exists():
                print(f"❌ Frame not found: {frame_path}", file=sys.stderr)
                sys.exit(1)

            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"❌ Cannot load image: {frame_path}", file=sys.stderr)
                sys.exit(1)

            result = pipeline.process_frame(frame, return_intermediates=True)

            print(f"\n🎸 Frame Analysis: {frame_path.name}")
            print(f"   Neck detected:    {result['neck_detected']}")
            if result["neck_detected"]:
                print(f"   Neck confidence:  {result['neck_confidence']:.3f}")
                print(f"   Neck bbox:        {result['neck_bbox']}")
                print(f"   Corners:          {result['corners']}")

                if result["chord_label"]:
                    print(f"   Chord:            {result['chord_label']} "
                          f"({result['chord_confidence']:.1%})")
                    if result["chord_probs"]:
                        print(f"   All probs:")
                        for name, prob in sorted(result["chord_probs"].items(),
                                                 key=lambda x: x[1], reverse=True):
                            bar = "█" * int(prob * 30)
                            print(f"     {name:<5s} {prob:>6.1%}  {bar}")

                # Save outputs
                out_dir = Path(args.output_dir) if args.output_dir else frame_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                if result.get("annotated") is not None:
                    ann_path = out_dir / f"{frame_path.stem}_vision.png"
                    cv2.imwrite(str(ann_path), result["annotated"])
                    print(f"\n   Annotated → {ann_path}")

                if result.get("warped") is not None:
                    warp_path = out_dir / f"{frame_path.stem}_warped.png"
                    cv2.imwrite(str(warp_path), result["warped"])
                    print(f"   Warped    → {warp_path}")
            else:
                print("   ⚠ No guitar neck detected in this frame.")

        elif args.video:
            pipeline.process_video(
                args.video,
                fps=args.fps,
                output_dir=args.output_dir,
            )

    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
