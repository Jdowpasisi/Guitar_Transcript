"""
P9 Smoke Test
=============
Validates the P9 pipeline without requiring a real video file.

Creates a synthetic 800×450 test frame with fretboard-like geometry
(horizontal lines simulating frets, two vertical lines for neck edges),
then runs the full detect → warp cycle.

Run with:
    python -m src.vision.test_p9

Expected output:
    ✓ Frame created (800×450)
    ✓ Grayscale conversion
    ✓ Edge detection — N edges
    ✓ Hough lines — N raw, N filtered
    ✓ Annotated frame saved
    ✓ Homography computed
    ✓ Warped frame: 200×600 shape
    ✓ Side-by-side saved
    All P9 smoke tests passed!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def make_synthetic_fretboard_frame(w: int = 800, h: int = 450) -> np.ndarray:
    """
    Create a BGR uint8 image with fretboard-like geometry:
    - Dark background
    - 2 near-vertical lines  (neck edges)
    - 6 near-horizontal lines (frets at increasing intervals, simulating equal temperament)
    - 6 slightly different-coloured lines (guitar strings)
    """
    frame = np.full((h, w, 3), fill_value=30, dtype=np.uint8)  # near-black

    # Neck edges (left and right sides of fretboard)
    left_x  = 80
    right_x = w - 80
    top_y   = 100
    bot_y   = h - 100

    cv2.line(frame, (left_x,  top_y), (left_x,  bot_y), (160, 130, 80), 3)  # left neck edge
    cv2.line(frame, (right_x, top_y), (right_x, bot_y), (160, 130, 80), 3)  # right neck edge

    # Frets (horizontal, equal-temperament spacing)
    n_frets = 12
    neck_w  = right_x - left_x
    for i in range(n_frets + 1):
        # Equal temperament: each fret = previous / 2^(1/12)
        ratio = 1 - (1 / (2 ** (i / 12)))
        x = left_x + int(neck_w * ratio * 0.9)
        cv2.line(frame, (x, top_y), (x, bot_y), (200, 200, 200), 2)

    # 6 Strings (horizontal-ish lines, slightly different brightness per string)
    string_positions = np.linspace(top_y + 20, bot_y - 20, 6).astype(int)
    for i, y in enumerate(string_positions):
        brightness = 120 + i * 15
        cv2.line(frame, (left_x, y), (right_x, y), (brightness, brightness, brightness), 1)

    # Inlay dots at positions 3, 5, 7, 9, 12
    inlay_frets = [3, 5, 7, 9, 12]
    mid_y = (top_y + bot_y) // 2
    for fret_n in inlay_frets:
        ratio = 1 - (1 / (2 ** (fret_n / 12)))
        ratio_prev = 1 - (1 / (2 ** ((fret_n - 1) / 12)))
        x = left_x + int(neck_w * (ratio + ratio_prev) / 2 * 0.9)
        cv2.circle(frame, (x, mid_y), 5, (200, 180, 100), -1)

    return frame


def run_smoke_test():
    """Run all P9 module tests on a synthetic frame."""
    from src.vision.detect_lines  import (
        to_grayscale, detect_edges, detect_lines,
        filter_lines_by_angle, draw_lines, process_frame,
    )
    from src.vision.warp_fretboard import (
        compute_homography, warp_frame, make_side_by_side, DST_CORNERS,
    )

    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        frame_path = tmp / "test_frame.png"

        # ── 1. Synthetic frame ─────────────────────────────────────────────
        frame = make_synthetic_fretboard_frame()
        cv2.imwrite(str(frame_path), frame)
        assert frame.shape == (450, 800, 3), f"Wrong shape: {frame.shape}"
        assert frame.dtype == np.uint8
        print(f"  ✓ Synthetic frame created  (800×450, {frame.dtype})")

        # ── 2. Grayscale ───────────────────────────────────────────────────
        gray = to_grayscale(frame)
        assert gray.shape == (450, 800), f"Wrong grayscale shape: {gray.shape}"
        assert gray.ndim == 2
        print(f"  ✓ Grayscale conversion     ({gray.shape})")

        # ── 3. Edge detection ──────────────────────────────────────────────
        edges = detect_edges(gray)
        n_edge_pixels = int(np.sum(edges > 0))
        assert n_edge_pixels > 100, f"Too few edge pixels: {n_edge_pixels}"
        print(f"  ✓ Canny edge detection     ({n_edge_pixels} edge pixels)")

        # ── 4. Hough line detection ────────────────────────────────────────
        raw_lines = detect_lines(edges)
        n_raw = 0 if raw_lines is None else len(raw_lines)
        filtered = filter_lines_by_angle(raw_lines, tolerance_deg=30)
        n_filt = len(filtered)
        assert n_filt >= 2, f"Expected ≥2 filtered lines, got {n_filt}"
        print(f"  ✓ Hough line detection     ({n_raw} raw → {n_filt} filtered)")

        # ── 5. Annotation drawing ──────────────────────────────────────────
        annotated = draw_lines(frame, filtered)
        assert annotated.shape == frame.shape
        annotated_path = tmp / "test_frame_lines.png"
        cv2.imwrite(str(annotated_path), annotated)
        assert annotated_path.exists()
        print(f"  ✓ Annotated frame saved    ({annotated_path.name})")

        # ── 6. process_frame integration ───────────────────────────────────
        result = process_frame(frame_path, output_dir=tmp, save=True)
        assert result["n_filtered"] >= 2
        assert Path(tmp / "test_frame_lines.png").exists()
        assert Path(tmp / "test_frame_edges.png").exists()
        print(f"  ✓ process_frame()          ({result['n_filtered']} filtered lines, files saved)")

        # ── 7. Homography ──────────────────────────────────────────────────
        # Use the neck-edge corners from our synthetic frame
        src_corners = np.array([
            [80,  100],   # TL: left neck edge, top
            [720, 100],   # TR: right neck edge, top
            [720, 350],   # BR: right neck edge, bottom
            [80,  350],   # BL: left neck edge, bottom
        ], dtype=np.float32)

        H = compute_homography(src_corners)
        assert H.shape == (3, 3), f"Wrong H shape: {H.shape}"
        print(f"  ✓ Homography computed      (3×3 matrix, det={np.linalg.det(H):.4f})")

        # ── 8. Perspective warp ────────────────────────────────────────────
        warped = warp_frame(frame, H)
        assert warped.shape == (200, 600, 3), f"Wrong warped shape: {warped.shape}"
        # Check it's not all black (warp actually mapped pixels)
        assert warped.mean() > 5, f"Warped image looks empty (mean={warped.mean():.1f})"
        print(f"  ✓ warpPerspective          (output: {warped.shape})")

        # ── 9. Side-by-side ────────────────────────────────────────────────
        sbs = make_side_by_side(frame, warped)
        assert sbs.ndim == 3
        assert sbs.shape[2] == 3
        sbs_path = tmp / "sbs.png"
        cv2.imwrite(str(sbs_path), sbs)
        assert sbs_path.exists()
        print(f"  ✓ Side-by-side canvas      ({sbs.shape})")

        # ── 10. Warped pixel sanity ────────────────────────────────────────
        # The warped image should have some bright pixels (the fret lines we drew)
        bright_pixels = int(np.sum(warped > 100))
        assert bright_pixels > 500, f"Warped image has too few bright pixels: {bright_pixels}"
        print(f"  ✓ Warped pixel check       ({bright_pixels} bright pixels > threshold)")

    print("\n" + "─" * 50)
    print("  All P9 smoke tests passed! ✅")
    print("─" * 50)


if __name__ == "__main__":
    print("\nP9 Smoke Test")
    print("─" * 50)
    try:
        run_smoke_test()
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        raise
    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("   Make sure opencv-python is installed: pip install opencv-python")
        raise
