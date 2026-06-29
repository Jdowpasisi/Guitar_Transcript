"""
P9 — Edge & Line Detector
==========================
Per-frame: Canny edge detection → Hough Line Transform → angle-filtered lines.

Usage (single frame):
    python -m src.vision.detect_lines frame.png [--output_dir outputs/frames]

Usage (batch over a frame folder):
    python -m src.vision.detect_lines --batch outputs/frames/my_video/frames/

What it does:
    1. Load frame (BGR uint8)
    2. Convert to grayscale
    3. Gaussian blur (removes high-freq noise before Canny)
    4. Canny edge detection (two thresholds)
    5. Hough Probabilistic Line Transform
    6. Filter: keep lines whose angle is within ±30° of horizontal
       (fretboard frets run roughly horizontal in most guitar videos)
    7. Draw filtered lines on the original frame in green
    8. Save: annotated frame alongside original

Outputs:
    <output_dir>/<stem>_lines.png   — original with detected lines overlaid
    <output_dir>/<stem>_edges.png   — Canny edge map (grayscale)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Detection parameters ──────────────────────────────────────────────────────

BLUR_KERNEL       = (5, 5)     # Gaussian kernel size — odd, larger = smoother
CANNY_LOW         = 50         # Lower threshold (weak edges)
CANNY_HIGH        = 150        # Upper threshold (strong edges)

HOUGH_RHO         = 1          # Distance resolution in pixels
HOUGH_THETA       = np.pi / 180  # Angle resolution in radians
HOUGH_THRESHOLD   = 80         # Accumulator threshold — higher = fewer, longer lines
HOUGH_MIN_LENGTH  = 60         # Minimum line length in pixels
HOUGH_MAX_GAP     = 15         # Maximum gap between collinear points

# Angle filter — keep lines within ±ANGLE_TOLERANCE_DEG of horizontal (0°)
# Frets are (nearly) horizontal; nut/saddle are nearly vertical — we want frets + neck edges
ANGLE_TOLERANCE_DEG = 30

# Visualisation colours (BGR)
LINE_COLOUR   = (0, 255, 0)    # Green
LINE_THICKNESS = 2
EDGE_ALPHA     = 0.4           # For edge overlay blending (not used by default)


# ── Core functions ────────────────────────────────────────────────────────────

def load_frame(path: str | Path) -> np.ndarray:
    """Load an image as a BGR uint8 NumPy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """BGR → 8-bit grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """
    Gaussian blur + Canny edge detection.
    Returns binary edge map (uint8, 0/255).
    """
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, sigmaX=0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    return edges


def detect_lines(edges: np.ndarray) -> np.ndarray | None:
    """
    Probabilistic Hough Line Transform on an edge map.
    Returns array of shape (N, 1, 4) — each row is (x1, y1, x2, y2) — or None.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )
    return lines  # shape: (N, 1, 4) or None


def filter_lines_by_angle(
    lines: np.ndarray | None,
    tolerance_deg: float = ANGLE_TOLERANCE_DEG,
) -> list[tuple[int, int, int, int]]:
    """
    Keep only lines within ±tolerance_deg of horizontal (0°) or vertical (90°).

    For guitar fretboard detection we want:
    - Horizontal-ish lines  → fret positions (within ±30°)
    - Vertical-ish lines    → neck edges / nut / saddle (within ±30° of 90°)

    Returns list of (x1, y1, x2, y2) tuples.
    """
    if lines is None:
        return []

    filtered = []
    tol = tolerance_deg * (np.pi / 180)  # to radians

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            continue

        angle = abs(np.arctan2(dy, dx))  # 0 = horizontal, π/2 = vertical

        is_horizontal = angle < tol or angle > (np.pi - tol)
        is_vertical   = abs(angle - np.pi / 2) < tol

        if is_horizontal or is_vertical:
            filtered.append((x1, y1, x2, y2))

    return filtered


def draw_lines(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    colour: tuple = LINE_COLOUR,
    thickness: int = LINE_THICKNESS,
) -> np.ndarray:
    """Draw lines on a copy of the frame. Returns annotated BGR image."""
    annotated = frame.copy()
    for (x1, y1, x2, y2) in lines:
        cv2.line(annotated, (x1, y1), (x2, y2), colour, thickness)
    return annotated


def process_frame(
    frame_path: str | Path,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> dict:
    """
    Full per-frame pipeline: load → grayscale → edges → lines → filter → annotate.

    Args:
        frame_path: Path to the frame PNG.
        output_dir: Where to save outputs. Defaults to same folder as frame_path.
        save:       Whether to save output images.

    Returns:
        dict with keys: n_raw_lines, n_filtered_lines, annotated (np.ndarray), edges (np.ndarray)
    """
    frame_path = Path(frame_path)
    if output_dir is None:
        output_dir = frame_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline
    frame    = load_frame(frame_path)
    gray     = to_grayscale(frame)
    edges    = detect_edges(gray)
    raw      = detect_lines(edges)
    filtered = filter_lines_by_angle(raw)
    annotated = draw_lines(frame, filtered)

    n_raw      = 0 if raw is None else len(raw)
    n_filtered = len(filtered)

    if save:
        stem = frame_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_lines.png"), annotated)
        cv2.imwrite(str(output_dir / f"{stem}_edges.png"), edges)

    return {
        "frame_path":   str(frame_path),
        "n_raw_lines":  n_raw,
        "n_filtered":   n_filtered,
        "lines":        filtered,
        "annotated":    annotated,
        "edges":        edges,
    }


def batch_process(
    frames_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "frame_*.png",
    verbose: bool = True,
) -> list[dict]:
    """
    Run detect_lines on every frame in a directory.

    Args:
        frames_dir: Directory containing frame PNGs.
        output_dir: Where to save annotated frames. Defaults to <frames_dir>/../lines/
        pattern:    Glob pattern to match frames.

    Returns:
        List of per-frame result dicts.
    """
    frames_dir = Path(frames_dir)
    if output_dir is None:
        output_dir = frames_dir.parent / "lines"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching '{pattern}' in {frames_dir}")

    print(f"\n🔍 Processing {len(frames)} frames in {frames_dir}")
    print(f"   Outputs → {output_dir}")

    results = []
    for i, fp in enumerate(frames):
        result = process_frame(fp, output_dir=output_dir, save=True)
        results.append(result)
        if verbose and (i % 10 == 0 or i == len(frames) - 1):
            print(f"   [{i+1}/{len(frames)}] {fp.name} — "
                  f"{result['n_raw_lines']} raw, {result['n_filtered']} filtered lines")

    # Summary
    avg_lines = sum(r["n_filtered"] for r in results) / max(len(results), 1)
    good = sum(1 for r in results if r["n_filtered"] >= 4)
    pct  = good / max(len(results), 1) * 100
    print(f"\n   Summary: {good}/{len(frames)} frames ({pct:.0f}%) have ≥4 filtered lines")
    print(f"   Average filtered lines per frame: {avg_lines:.1f}")
    print(f"\n✅ Batch complete → {output_dir}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P9: Detect fretboard edges in guitar video frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("frame",    nargs="?",  type=str, help="Path to single frame PNG")
    group.add_argument("--batch",  type=str,   help="Directory of frame PNGs to process in batch")

    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--angle_tol", type=float, default=ANGLE_TOLERANCE_DEG,
        help="±degrees from horizontal/vertical to keep lines",
    )
    args = parser.parse_args()

    try:
        if args.batch:
            batch_process(args.batch, output_dir=args.output_dir)
        else:
            result = process_frame(args.frame, output_dir=args.output_dir)
            print(f"Raw lines: {result['n_raw_lines']}, Filtered: {result['n_filtered']}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
