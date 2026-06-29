"""
P9 — Fretboard Warper
======================
Two modes:

  INTERACTIVE (default):
    Opens an OpenCV window. Click 4 corners of the fretboard in order:
      1. Top-left (nut, high-E string)
      2. Top-right (body end, high-E string)
      3. Bottom-right (body end, low-E string)
      4. Bottom-left (nut, low-E string)
    Press Enter to compute homography and warp. Press 'r' to reset points.

  HEADLESS (--corners):
    Supply corners directly (e.g. from an automated detector in P10).

Usage:
    # Interactive
    python -m src.vision.warp_fretboard frame.png

    # Headless — corners as JSON string: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    python -m src.vision.warp_fretboard frame.png \\
        --corners "[[42,85],[612,78],[618,195],[38,200]]"

    # Batch: warp every frame in a folder using a saved corners.json
    python -m src.vision.warp_fretboard --batch outputs/frames/myvid/frames/ \\
        --corners_json outputs/frames/myvid/corners.json

Outputs:
    <output_dir>/<stem>_warped.png    — flat 600×200 fretboard rectangle
    <output_dir>/<stem>_side_by_side.png  — original + warped, side by side
    <job_dir>/corners.json            — saved corner coordinates (reuse for batch)
    <job_dir>/homography.npy          — 3×3 homography matrix (for P11)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Warp target dimensions ────────────────────────────────────────────────────
WARP_W = 600   # width in pixels of output fretboard rectangle
WARP_H = 200   # height in pixels

# ── UI colours ────────────────────────────────────────────────────────────────
CORNER_COLOUR   = (0,   255, 255)  # Cyan dot for each clicked corner
ACTIVE_COLOUR   = (0,   165, 255)  # Orange — next corner to click
LINE_COLOUR     = (0,   255,   0)  # Green — outline polygon
FONT            = cv2.FONT_HERSHEY_SIMPLEX
CORNER_LABELS   = ["TL (nut, high-E)", "TR (body, high-E)",
                   "BR (body, low-E)", "BL (nut, low-E)"]


# ── Homography helpers ────────────────────────────────────────────────────────

DST_CORNERS = np.array([
    [0,      0     ],   # top-left
    [WARP_W, 0     ],   # top-right
    [WARP_W, WARP_H],   # bottom-right
    [0,      WARP_H],   # bottom-left
], dtype=np.float32)


def compute_homography(src_corners: np.ndarray) -> np.ndarray:
    """
    Compute the 3×3 homography matrix mapping src_corners → DST_CORNERS.

    Args:
        src_corners: (4, 2) float32 array in TL/TR/BR/BL order.

    Returns:
        H: (3, 3) float64 homography matrix.
    """
    H, mask = cv2.findHomography(
        src_corners.astype(np.float32),
        DST_CORNERS,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if H is None:
        raise RuntimeError("findHomography failed — corners may be collinear or degenerate.")
    return H


def warp_frame(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography H to warp the fretboard to a flat WARP_W × WARP_H rectangle.

    Args:
        frame: BGR uint8 source image.
        H:     3×3 homography matrix.

    Returns:
        warped: BGR uint8 image of shape (WARP_H, WARP_W, 3).
    """
    return cv2.warpPerspective(frame, H, (WARP_W, WARP_H),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def make_side_by_side(original: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """
    Stack original (resized to same height as warped) and warped side by side.
    Adds a label bar at the bottom.
    """
    h_w = WARP_H
    # Resize original to the same height, preserving aspect ratio
    h_orig, w_orig = original.shape[:2]
    scale   = h_w / h_orig
    w_scaled = int(w_orig * scale)
    left = cv2.resize(original, (w_scaled, h_w))

    # Pad warped to same height (already correct)
    right = warped.copy()

    # Divider line
    divider = np.zeros((h_w, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)

    canvas = np.concatenate([left, divider, right], axis=1)

    # Label bar
    bar_h = 24
    bar   = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "Original", (10, 17), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, f"Warped ({WARP_W}x{WARP_H})",
                (w_scaled + 14, 17), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.concatenate([canvas, bar], axis=0)


# ── Interactive corner picker ─────────────────────────────────────────────────

def _check_cv2_gui_available() -> bool:
    """Test whether OpenCV can open a GUI window and handle mouse events."""
    try:
        test_name = "__p9_gui_probe__"
        cv2.namedWindow(test_name, cv2.WINDOW_NORMAL)
        # This is the call that actually fails on broken Qt/Wayland —
        # namedWindow may "succeed" but produce a NULL internal handle
        cv2.setMouseCallback(test_name, lambda *a: None)
        cv2.destroyWindow(test_name)
        cv2.waitKey(1)
        return True
    except cv2.error:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        return False


class CornerPicker:
    """
    OpenCV mouse-callback UI for picking 4 fretboard corners.

    Cycle:
      Click 4 corners → polygon draws live → Enter to confirm → warp → save.
      'r' resets the points.  'q' / ESC quits without saving.
    """

    WINDOW = "P9 — Click 4 fretboard corners (TL → TR → BR → BL), then press Enter"

    def __init__(self, frame: np.ndarray):
        self.original = frame.copy()
        self.display  = frame.copy()
        self.points: list[list[int]] = []
        self.done     = False
        self.confirmed = False

    def _redraw(self):
        self.display = self.original.copy()
        for i, pt in enumerate(self.points):
            colour = CORNER_COLOUR
            cv2.circle(self.display, tuple(pt), 7, colour, -1)
            cv2.circle(self.display, tuple(pt), 9, (255, 255, 255), 1)
            # Label
            label = f"{i+1}"
            cv2.putText(self.display, label,
                        (pt[0] + 10, pt[1] - 6), FONT, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(self.display, label,
                        (pt[0] + 10, pt[1] - 6), FONT, 0.55,
                        colour, 1, cv2.LINE_AA)
        # Draw polygon outline when ≥2 points selected
        if len(self.points) >= 2:
            pts_arr = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(self.display, [pts_arr], isClosed=(len(self.points) == 4),
                          color=LINE_COLOUR, thickness=1)
        # Status bar at bottom
        h, w = self.display.shape[:2]
        overlay = self.display.copy()
        cv2.rectangle(overlay, (0, h - 36), (w, h), (20, 20, 20), -1)
        self.display = cv2.addWeighted(overlay, 0.7, self.display, 0.3, 0)
        n = len(self.points)
        if n < 4:
            msg = f"Click corner {n+1}/4: {CORNER_LABELS[n]}"
        else:
            msg = "4 corners selected — press Enter to warp, 'r' to reset, 'q' to quit"
        cv2.putText(self.display, msg, (10, h - 12), FONT, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            self._redraw()
            cv2.imshow(self.WINDOW, self.display)

    def run(self) -> Optional[list[list[int]]]:
        """Open window, collect corners. Returns corner list or None if cancelled."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1200, 700)
        cv2.setMouseCallback(self.WINDOW, self.mouse_cb)
        self._redraw()
        cv2.imshow(self.WINDOW, self.display)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):   # Enter
                if len(self.points) == 4:
                    self.confirmed = True
                    break
                else:
                    print(f"  Need 4 corners, only have {len(self.points)} — keep clicking.")
            elif key == ord('r'):
                self.points = []
                self._redraw()
                cv2.imshow(self.WINDOW, self.display)
                print("  Points reset.")
            elif key in (27, ord('q')):  # ESC or q
                print("  Cancelled by user.")
                break

        cv2.destroyAllWindows()
        return self.points if self.confirmed else None


class MatplotlibCornerPicker:
    """
    Fallback corner picker using matplotlib when OpenCV GUI is unavailable.
    Uses plt.ginput() to collect 4 clicks, then closes the figure.
    """

    def __init__(self, frame: np.ndarray):
        self.frame = frame  # BGR uint8

    def run(self) -> Optional[list[list[int]]]:
        """Open matplotlib window, collect 4 corners. Returns list or None."""
        try:
            import matplotlib
            # Try non-interactive backends that work in various environments
            gui_backends = ['TkAgg', 'GTK3Agg', 'Qt5Agg', 'macosx']
            backend_set = False
            for backend in gui_backends:
                try:
                    matplotlib.use(backend)
                    backend_set = True
                    break
                except ImportError:
                    continue
            if not backend_set:
                return None

            import matplotlib.pyplot as plt
        except ImportError:
            return None

        # Convert BGR → RGB for matplotlib
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.imshow(rgb)
        ax.set_title(
            "Click 4 fretboard corners: TL → TR → BR → BL\n"
            "then close the window (or right-click to undo last point)",
            fontsize=12, fontweight='bold',
        )
        for i, label in enumerate(CORNER_LABELS):
            ax.text(0.01, 0.01 + i * 0.04, f"  {i+1}. {label}",
                    transform=ax.transAxes, fontsize=9, color='cyan',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.tight_layout()

        print("  📌 Click 4 corners in the matplotlib window, then close it.")
        print("     Right-click to undo the last point.")

        try:
            pts = plt.ginput(n=4, timeout=0, mouse_pop=3, mouse_stop=2)
            plt.close(fig)
        except Exception:
            plt.close(fig)
            return None

        if len(pts) != 4:
            print(f"  ⚠ Expected 4 points, got {len(pts)} — cancelled.")
            return None

        corners = [[int(round(x)), int(round(y))] for x, y in pts]
        print(f"  ✓ Corners selected: {corners}")
        return corners


def pick_corners(frame: np.ndarray) -> Optional[list[list[int]]]:
    """
    Try to pick 4 fretboard corners interactively.
    Tries OpenCV GUI first, falls back to matplotlib, then gives a clear error.
    """
    # Attempt 1: OpenCV GUI
    if _check_cv2_gui_available():
        print("  (Using OpenCV GUI)")
        try:
            picker = CornerPicker(frame)
            result = picker.run()
            if result is not None:
                return result
        except cv2.error as e:
            print(f"  ⚠ OpenCV GUI crashed: {e}")
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # Attempt 2: Matplotlib fallback
    print("  ⚠ OpenCV GUI unavailable (no display or Qt plugin missing).")
    print("  ↳ Trying matplotlib fallback…")
    mpl_picker = MatplotlibCornerPicker(frame)
    result = mpl_picker.run()
    if result is not None:
        return result

    # Both failed — give actionable guidance
    print("\n  ❌ No GUI backend available for interactive corner selection.")
    print("     Use one of these headless alternatives instead:\n")
    print("     1. Supply corners as a JSON string:")
    print('        --corners "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"')
    print("     2. Supply corners from a saved file:")
    print("        --corners_json path/to/corners.json")
    print("     3. Open the reference frame in an image viewer,")
    print("        note the 4 corner pixel coordinates, and pass them in.\n")
    return None


# ── Main warp function ────────────────────────────────────────────────────────

def warp_fretboard(
    frame_path: str | Path,
    corners: list[list[int]] | None = None,
    output_dir: str | Path | None = None,
    save_homography: bool = True,
    interactive: bool = True,
) -> dict:
    """
    Warp the fretboard region of a frame to a flat rectangle.

    Args:
        frame_path:       Path to frame PNG.
        corners:          4 corner points [[x,y]×4] in TL/TR/BR/BL order.
                          If None and interactive=True, opens UI for manual picking.
        output_dir:       Where to save outputs.
        save_homography:  If True, saves homography.npy to the parent of output_dir.
        interactive:      Whether to allow the interactive picker UI.

    Returns:
        Dict with keys: corners, H, warped, side_by_side, output_paths.
    """
    frame_path = Path(frame_path)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot load frame: {frame_path}")

    if output_dir is None:
        output_dir = frame_path.parent.parent / "warped"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Get corners ───────────────────────────────────────────────────────────
    if corners is None:
        if not interactive:
            raise ValueError("corners=None and interactive=False — cannot proceed.")
        print(f"\n🖱  Opening interactive corner picker on: {frame_path.name}")
        print("   Click 4 corners in TL → TR → BR → BL order, then press Enter.")
        corners = pick_corners(frame)
        if corners is None:
            raise RuntimeError("No corners selected — cancelled.")

    corners_arr = np.array(corners, dtype=np.float32)
    if corners_arr.shape != (4, 2):
        raise ValueError(f"Expected 4 corners of shape (4,2), got {corners_arr.shape}")

    # ── Homography + warp ─────────────────────────────────────────────────────
    H       = compute_homography(corners_arr)
    warped  = warp_frame(frame, H)
    sbs     = make_side_by_side(frame, warped)

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = frame_path.stem
    warped_path = output_dir / f"{stem}_warped.png"
    sbs_path    = output_dir / f"{stem}_side_by_side.png"
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(sbs_path),   sbs)

    output_paths = {
        "warped":       str(warped_path),
        "side_by_side": str(sbs_path),
    }

    # Save homography and corners to the job root (parent of output_dir)
    job_dir = output_dir.parent
    if save_homography:
        H_path = job_dir / "homography.npy"
        np.save(str(H_path), H)
        output_paths["homography"] = str(H_path)
        print(f"   💾 Homography matrix saved → {H_path}")

    corners_path = job_dir / "corners.json"
    corners_path.write_text(json.dumps(corners, indent=2))
    output_paths["corners"] = str(corners_path)

    print(f"   ✓ Warped fretboard → {warped_path}")
    print(f"   ✓ Side-by-side     → {sbs_path}")

    return {
        "corners":      corners,
        "H":            H,
        "warped":       warped,
        "side_by_side": sbs,
        "output_paths": output_paths,
    }


def batch_warp(
    frames_dir: str | Path,
    corners_json: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "frame_*.png",
) -> list[dict]:
    """
    Warp every frame in frames_dir using pre-saved corners.

    Args:
        frames_dir:   Directory of frame PNGs.
        corners_json: Path to a corners.json file (saved by interactive run).
        output_dir:   Output directory. Defaults to <frames_dir>/../warped/
        pattern:      Glob pattern for frames.

    Returns:
        List of result dicts.
    """
    frames_dir   = Path(frames_dir)
    corners_json = Path(corners_json)
    if output_dir is None:
        output_dir = frames_dir.parent / "warped"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(corners_json) as f:
        corners = json.load(f)

    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames matching '{pattern}' in {frames_dir}")

    print(f"\n📐 Batch warping {len(frames)} frames")
    print(f"   Corners from: {corners_json}")
    print(f"   Output dir:   {output_dir}")

    # Pre-compute homography once
    corners_arr = np.array(corners, dtype=np.float32)
    H = compute_homography(corners_arr)
    H_path = output_dir.parent / "homography.npy"
    np.save(str(H_path), H)

    results = []
    good = 0
    for i, fp in enumerate(frames):
        try:
            result = warp_fretboard(
                fp,
                corners=corners,
                output_dir=output_dir,
                save_homography=False,
                interactive=False,
            )
            results.append(result)
            good += 1
            if i % 10 == 0 or i == len(frames) - 1:
                print(f"   [{i+1}/{len(frames)}] {fp.name} ✓")
        except Exception as exc:
            print(f"   [{i+1}/{len(frames)}] {fp.name} — FAILED: {exc}")
            results.append({"error": str(exc), "frame": str(fp)})

    pct = good / len(frames) * 100
    print(f"\n✅ Batch complete: {good}/{len(frames)} ({pct:.0f}%) frames warped")
    print(f"   Homography saved → {H_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P9: Warp fretboard region to a flat 600×200 rectangle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("frame",       nargs="?", type=str, help="Single frame PNG")
    group.add_argument("--batch",     type=str,  help="Directory of frame PNGs (requires --corners_json)")

    parser.add_argument("--corners",      type=str, default=None,
                        help="JSON string of 4 corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    parser.add_argument("--corners_json", type=str, default=None,
                        help="Path to saved corners.json (for batch mode)")
    parser.add_argument("--output_dir",   type=str, default=None, help="Output directory")
    args = parser.parse_args()

    try:
        if args.batch:
            if not args.corners_json:
                print("❌ --batch requires --corners_json", file=sys.stderr)
                sys.exit(1)
            batch_warp(args.batch, args.corners_json, output_dir=args.output_dir)
        else:
            corners = None
            if args.corners:
                corners = json.loads(args.corners)
            result = warp_fretboard(
                args.frame,
                corners=corners,
                output_dir=args.output_dir,
                interactive=(corners is None),
            )
            print(f"\nCorners used: {result['corners']}")
            print(f"Homography H:\n{result['H']}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
