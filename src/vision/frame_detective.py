"""
P9 — Frame Detective Pipeline
===============================
One command to run the complete P9 pipeline on a guitar video:

  1. Extract frames at 5fps (FFmpeg)
  2. Run edge detection + line overlay on every frame (Canny + Hough)
  3. Interactively pick 4 fretboard corners on the first good frame
  4. Warp all frames to a flat 600×200 fretboard rectangle (homography)
  5. Save a summary JSON

Usage:
    python -m src.vision.frame_detective <video>
    python -m src.vision.frame_detective <video> --fps 3 --output_dir my_outputs
    python -m src.vision.frame_detective <video> --corners "[[42,85],[612,78],[618,195],[38,200]]"
    python -m src.vision.frame_detective <video> --corners_json path/to/corners.json
    python -m src.vision.frame_detective <video> --skip_lines  # skip Canny/Hough, just warp
    python -m src.vision.frame_detective <video> --skip_warp   # only extract + detect lines

Outputs tree:
    <output_dir>/<video_stem>/
      ├── frames/                     # Raw PNG frames
      │   ├── frame_000001.png
      │   └── ...
      ├── audio.wav                   # Extracted audio (→ pass to P7 API)
      ├── lines/                      # Per-frame: lines + edges overlaid
      │   ├── frame_000001_lines.png
      │   ├── frame_000001_edges.png
      │   └── ...
      ├── warped/                     # Per-frame: flat fretboard rectangle
      │   ├── frame_000001_warped.png
      │   ├── frame_000001_side_by_side.png
      │   └── ...
      ├── corners.json                # 4 corner coordinates (reuse with --corners_json)
      ├── homography.npy              # 3×3 H matrix for P11 finger mapping
      ├── meta.json                   # Video metadata from ffprobe
      └── summary.json                # Pipeline summary stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .extract_frames import extract_frames
from .detect_lines   import batch_process as batch_detect_lines
from .warp_fretboard import batch_warp, warp_fretboard


def _pick_reference_frame(frames_dir: Path, line_results: list[dict]) -> Path:
    """
    Pick the best frame for interactive corner selection.
    Prefers frames with ≥4 filtered lines; falls back to the middle frame.
    """
    # Frames with ≥4 filtered lines, sorted by count desc
    good = sorted(
        [r for r in line_results if r.get("n_filtered", 0) >= 4],
        key=lambda r: r["n_filtered"], reverse=True,
    )
    if good:
        return Path(good[0]["frame_path"])

    # Fallback: middle frame
    frames = sorted(frames_dir.glob("frame_*.png"))
    return frames[len(frames) // 2]


def run_pipeline(
    video_path: str,
    output_root: str = "outputs/frames",
    fps: float = 5.0,
    corners: list | None = None,
    corners_json: str | None = None,
    skip_lines: bool = False,
    skip_warp: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Full P9 pipeline.

    Args:
        video_path:   Path to guitar video.
        output_root:  Root output directory.
        fps:          Frames per second to extract.
        corners:      Pre-specified 4 corners (skips interactive picker).
        corners_json: Path to saved corners.json.
        skip_lines:   Skip Canny/Hough edge detection step.
        skip_warp:    Skip homography warp step (implies skip_lines=False).
        overwrite:    Re-extract even if outputs exist.

    Returns:
        summary dict (also saved to summary.json).
    """
    t_total = time.monotonic()
    video_path = Path(video_path)
    stem       = video_path.stem
    job_dir    = Path(output_root) / stem
    frames_dir = job_dir / "frames"

    print("\n" + "═" * 60)
    print("  🎸 GuitarAI P9 — Frame Detective")
    print("═" * 60)
    print(f"  Video : {video_path}")
    print(f"  Output: {job_dir}")
    print(f"  FPS   : {fps}")
    print("═" * 60)

    summary = {
        "video":       str(video_path),
        "job_dir":     str(job_dir),
        "fps":         fps,
        "steps":       {},
    }

    # ── Step 1: Extract frames ─────────────────────────────────────────────────
    print("\n── Step 1/4: Extract frames ──────────────────────────────")
    t0   = time.monotonic()
    meta = extract_frames(video_path, output_root, fps=fps, overwrite=overwrite)
    t1   = time.monotonic()
    n_frames = meta.get("extracted_frames", 0)
    summary["steps"]["extract"] = {
        "duration_sec":    round(t1 - t0, 2),
        "frames_extracted": n_frames,
        "audio_path":       meta.get("audio_path"),
    }
    summary["video_meta"] = meta

    # ── Step 2: Edge + line detection ─────────────────────────────────────────
    line_results = []
    if not skip_lines:
        print("\n── Step 2/4: Detect edges + lines ────────────────────────")
        t0 = time.monotonic()
        line_results = batch_detect_lines(
            frames_dir, output_dir=job_dir / "lines", verbose=True
        )
        t1 = time.monotonic()
        n_good = sum(1 for r in line_results if r.get("n_filtered", 0) >= 4)
        pct_good = n_good / max(len(line_results), 1) * 100
        summary["steps"]["detect_lines"] = {
            "duration_sec":    round(t1 - t0, 2),
            "frames_processed": len(line_results),
            "frames_with_4plus_lines": n_good,
            "pct_good_frames": round(pct_good, 1),
        }
    else:
        print("\n── Step 2/4: Detect lines — SKIPPED (--skip_lines) ──────")
        summary["steps"]["detect_lines"] = {"skipped": True}

    # ── Step 3: Corner selection ──────────────────────────────────────────────
    print("\n── Step 3/4: Fretboard corner selection ──────────────────")

    # Load corners from: arg > json file > interactive picker
    if corners is not None:
        print(f"  Using corners from --corners argument: {corners}")
    elif corners_json is not None:
        cj_path = Path(corners_json)
        if not cj_path.exists():
            # Check if saved in job_dir
            cj_path = job_dir / "corners.json"
        with open(cj_path) as f:
            corners = json.load(f)
        print(f"  Loaded corners from {cj_path}: {corners}")
    else:
        # Interactive
        ref_frame = _pick_reference_frame(frames_dir, line_results)
        print(f"\n  Reference frame: {ref_frame.name}")
        print("  (This frame had the most detected lines — good for corner selection)")

        # Open the line-annotated version if it exists (easier to see the fretboard)
        lines_version = job_dir / "lines" / f"{ref_frame.stem}_lines.png"
        pick_frame    = lines_version if lines_version.exists() else ref_frame

        result = warp_fretboard(
            pick_frame,
            corners=None,
            output_dir=job_dir / "warped",
            save_homography=True,
            interactive=True,
        )
        corners = result["corners"]

    summary["corners"] = corners

    # ── Step 4: Batch warp ────────────────────────────────────────────────────
    if not skip_warp:
        print("\n── Step 4/4: Batch warp all frames ───────────────────────")
        t0 = time.monotonic()

        # Save corners.json in job_dir if not already there
        corners_path = job_dir / "corners.json"
        corners_path.write_text(json.dumps(corners, indent=2))

        warp_results = batch_warp(
            frames_dir,
            corners_json=str(corners_path),
            output_dir=job_dir / "warped",
        )
        t1 = time.monotonic()
        n_ok   = sum(1 for r in warp_results if "error" not in r)
        pct_ok = n_ok / max(len(warp_results), 1) * 100
        summary["steps"]["warp"] = {
            "duration_sec":   round(t1 - t0, 2),
            "frames_warped":  n_ok,
            "frames_failed":  len(warp_results) - n_ok,
            "pct_successful": round(pct_ok, 1),
        }
    else:
        print("\n── Step 4/4: Warp — SKIPPED (--skip_warp) ───────────────")
        summary["steps"]["warp"] = {"skipped": True}

    # ── Summary ───────────────────────────────────────────────────────────────
    summary["total_duration_sec"] = round(time.monotonic() - t_total, 2)

    summary_path = job_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    warp_pct = summary["steps"].get("warp", {}).get("pct_successful", "—")
    good_pct  = summary["steps"].get("detect_lines", {}).get("pct_good_frames", "—")

    print("\n" + "═" * 60)
    print("  ✅ Pipeline complete!")
    print("═" * 60)
    print(f"  Frames extracted:     {n_frames}")
    if not skip_lines:
        print(f"  Frames with ≥4 lines: {good_pct}%")
    if not skip_warp:
        print(f"  Frames warped OK:     {warp_pct}%")
    print(f"  Total time:           {summary['total_duration_sec']}s")
    print(f"  Output:               {job_dir}")
    print(f"  Audio for P7:         {meta.get('audio_path', '—')}")
    h_path = job_dir / "homography.npy"
    if h_path.exists():
        print(f"  Homography for P11:   {h_path}")
    print(f"  Summary:              {summary_path}")
    print("═" * 60 + "\n")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P9 Frame Detective: Extract → Detect lines → Warp fretboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video",       type=str,  help="Path to guitar video file")
    parser.add_argument("--fps",       type=float, default=5.0,
                        help="Frames per second to extract")
    parser.add_argument("--output_dir",type=str,  default="outputs/frames",
                        help="Root output directory")
    parser.add_argument("--corners",   type=str,  default=None,
                        help="JSON string of 4 corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    parser.add_argument("--corners_json", type=str, default=None,
                        help="Path to saved corners.json (skip interactive picking)")
    parser.add_argument("--skip_lines", action="store_true",
                        help="Skip Canny/Hough line detection step")
    parser.add_argument("--skip_warp",  action="store_true",
                        help="Skip homography warp step")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-extract even if output folder exists")
    args = parser.parse_args()

    corners = None
    if args.corners:
        try:
            corners = json.loads(args.corners)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid --corners JSON: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        run_pipeline(
            video_path=args.video,
            output_root=args.output_dir,
            fps=args.fps,
            corners=corners,
            corners_json=args.corners_json,
            skip_lines=args.skip_lines,
            skip_warp=args.skip_warp,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
