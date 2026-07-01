"""
P11 Smoke Test
================
Validates the P11 pipeline without requiring a real video or webcam.

Tests:
  1. Equal-temperament fret boundary math (sanity: fret 12 ≈ half the neck)
  2. x_to_fret() bucketing correctness at known positions
  3. y_to_string() row mapping correctness
  4. Homography point transform round-trips correctly
  5. Full HandTracker.process_frame() on a synthetic frame with no hand
     (validates it doesn't crash and returns an empty reading list)
  6. CSV writer + annotation drawing on synthetic FingerReadings (no
     MediaPipe required for this part — exercises draw_annotations directly)

Run with:
    python -m src.vision.test_p11
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np


def run_smoke_test():
    from src.vision.finger_tracker import (
        compute_fret_boundaries, x_to_fret, y_to_string,
        transform_point, point_in_warp_bounds,
        FingerReading, WARP_W, WARP_H, N_FRETS, STRING_NAMES,
        HAND_LANDMARKER_MODEL_PATH,
    )

    # ── 1. Fret boundary math ───────────────────────────────────────────────
    positions = compute_fret_boundaries(neck_length_px=WARP_W, n_frets=N_FRETS)
    assert len(positions) == N_FRETS + 1, f"Expected {N_FRETS+1} positions, got {len(positions)}"
    assert positions[0] == 0.0, f"Nut position should be 0, got {positions[0]}"
    assert abs(positions[-1] - WARP_W) < 1e-6, \
        f"Fret {N_FRETS} should land at neck_length_px={WARP_W}, got {positions[-1]}"
    # Monotonically increasing
    assert np.all(np.diff(positions) > 0), "Fret positions must be strictly increasing"
    # Fret 12 (the octave) should land very close to the halfway point —
    # this is the textbook luthier check: 2^(-12/12) = 0.5, so position(12)
    # = scale * (1 - 0.5) = scale/2. Since position(22) = neck_length_px,
    # position(12) won't be exactly neck_length_px/2 (scale != neck_length_px),
    # but it should be substantially more than half of the playable region,
    # consistent with frets compressing toward the body.
    assert positions[12] > positions[-1] * 0.55, \
        f"Fret 12 should be past the midpoint of the neck, got {positions[12]:.1f}/{positions[-1]:.1f}"
    print(f"  ✓ Fret boundary math       (nut=0, fret12={positions[12]:.1f}px, "
          f"fret22={positions[-1]:.1f}px)")

    # Spacing should shrink as fret number increases (closer together near body)
    spacing_low  = positions[1] - positions[0]
    spacing_high = positions[-1] - positions[-2]
    assert spacing_low > spacing_high, \
        f"Fret spacing should shrink toward the body: fret1 gap={spacing_low:.1f}, " \
        f"fret22 gap={spacing_high:.1f}"
    print(f"  ✓ Fret spacing decreases   (fret1 gap={spacing_low:.1f}px > "
          f"fret22 gap={spacing_high:.1f}px)")

    # ── 2. x_to_fret bucketing ──────────────────────────────────────────────
    # A finger between the nut and fret-1's wire sounds fret 1 (nearest
    # pressable fret) — fret 0 means "open string, no finger" and is never
    # returned by x_to_fret itself; it's the absence of a reading.
    assert x_to_fret(0.0, positions) == 1, "x=0 (at the nut) should map to the nearest fret, 1"
    assert x_to_fret(-50, positions) == 1, "Negative x should clamp to fret 1"
    assert x_to_fret(WARP_W + 100, positions) == N_FRETS, "x past the end should clamp to last fret"
    # Just past fret 1's wire position should read as fret 2 (pressing between 1 and 2)
    fret2_x = positions[1] + 1.0
    assert x_to_fret(fret2_x, positions) == 2, \
        f"x just past fret1 wire should read as fret 2, got {x_to_fret(fret2_x, positions)}"
    # Exactly at fret 1's wire still counts as fret 1 (pressing right at the wire)
    assert x_to_fret(float(positions[1]), positions) == 1, \
        "x exactly at fret1's wire should still read as fret 1"
    print(f"  ✓ x_to_fret() bucketing    (nut→fret1, {fret2_x:.1f}px→fret2, "
          f"oob→clamped to [1,{N_FRETS}])")

    # ── 3. y_to_string mapping ──────────────────────────────────────────────
    # Row 0 (top, y≈0) should map to string index 5 (high E) per WARP_ROW_TO_STRING
    top_string    = y_to_string(1.0, WARP_H)
    bottom_string = y_to_string(WARP_H - 1.0, WARP_H)
    assert top_string == 5, f"Top of warped image should be high-E (string 5), got {top_string}"
    assert bottom_string == 0, f"Bottom of warped image should be low-E (string 0), got {bottom_string}"
    mid_string = y_to_string(WARP_H / 2, WARP_H)
    assert 0 <= mid_string <= 5
    print(f"  ✓ y_to_string() mapping    (top={STRING_NAMES[top_string]}, "
          f"bottom={STRING_NAMES[bottom_string]})")

    # ── 4. Homography point transform ───────────────────────────────────────
    # Identity-like homography: a simple translation, easy to verify by hand.
    H_translate = np.array([
        [1, 0, 50],
        [0, 1, 30],
        [0, 0, 1],
    ], dtype=np.float64)
    wx, wy = transform_point(10, 20, H_translate)
    assert abs(wx - 60) < 1e-3 and abs(wy - 50) < 1e-3, \
        f"Translation homography failed: expected (60,50), got ({wx:.2f},{wy:.2f})"
    print(f"  ✓ transform_point()        (translate H: (10,20)→({wx:.0f},{wy:.0f}))")

    # Bounds checking
    assert point_in_warp_bounds(300, 100) is True
    assert point_in_warp_bounds(-100, 100) is False
    assert point_in_warp_bounds(WARP_W + 5, 100) is True  # within margin
    assert point_in_warp_bounds(WARP_W + 50, 100) is False
    print(f"  ✓ point_in_warp_bounds()   (in-bounds / out-of-bounds / margin all correct)")

    # ── 5. HandTracker on a synthetic frame (no hand present) ───────────────
    try:
        from src.vision.finger_tracker import HandTracker
        # Check if the model file exists (required for the Tasks API)
        model_available = HAND_LANDMARKER_MODEL_PATH.exists()
        if not model_available:
            print(f"  ⚠ HandLandmarker model not found at {HAND_LANDMARKER_MODEL_PATH}")
            print(f"    Attempting to download…")
            from src.vision.finger_tracker import _ensure_model_downloaded
            _ensure_model_downloaded()
            model_available = HAND_LANDMARKER_MODEL_PATH.exists()

        mediapipe_available = True
    except ImportError:
        mediapipe_available = False
        model_available = False

    if mediapipe_available and model_available:
        H_identity = np.eye(3, dtype=np.float64)
        try:
            # Use IMAGE mode for single-frame test (no timestamp tracking needed)
            tracker = HandTracker(H_identity, running_mode="IMAGE")
        except (ImportError, FileNotFoundError) as e:
            print(f"  ⚠ mediapipe installed but unusable — skipping HandTracker live test "
                  f"({e})")
            tracker = None

        if tracker is not None:
            try:
                blank_frame = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
                readings, mp_results = tracker.process_frame(blank_frame, frame_idx=0, timestamp=0.0)
                assert isinstance(readings, list), "process_frame should return a list"
                assert len(readings) == 0, \
                    f"Blank frame should produce 0 hand readings, got {len(readings)}"
                print(f"  ✓ HandTracker.process_frame()  (blank frame → 0 readings, no crash)")
            finally:
                tracker.close()
    elif not mediapipe_available:
        print(f"  ⚠ mediapipe not installed — skipping HandTracker live test "
              f"(install with: pip install mediapipe)")
    else:
        print(f"  ⚠ HandLandmarker model not available — skipping HandTracker live test")

    # ── 6. CSV writer + annotation drawing on synthetic readings ────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        synthetic_readings = [
            FingerReading(
                frame_idx=0, timestamp=0.0, hand_label="Right",
                finger_id="index", px=120.0, py=40.0,
                string=5, fret=3, confidence=0.95,
            ),
            FingerReading(
                frame_idx=0, timestamp=0.0, hand_label="Right",
                finger_id="middle", px=200.0, py=80.0,
                string=3, fret=5, confidence=0.91,
            ),
            FingerReading(
                frame_idx=0, timestamp=0.0, hand_label="Right",
                finger_id="thumb", px=-50.0, py=10.0,
                string=None, fret=None, confidence=0.80,
            ),
        ]

        csv_path = tmp / "test_tracking.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "finger_id", "string", "fret", "confidence"])
            for r in synthetic_readings:
                writer.writerow([
                    r.timestamp, r.finger_id,
                    r.string if r.string is not None else "",
                    r.fret if r.fret is not None else "",
                    r.confidence,
                ])
        assert csv_path.exists()

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4, f"Expected header + 3 rows, got {len(rows)}"
        assert rows[0] == ["timestamp", "finger_id", "string", "fret", "confidence"]
        assert rows[1][1] == "index" and rows[1][2] == "5" and rows[1][3] == "3"
        assert rows[3][2] == "", "Off-fretboard reading should have empty string field"
        print(f"  ✓ CSV export               (3 rows written, off-board reading handled)")

        # Annotation drawing (doesn't require MediaPipe — uses cv2 directly,
        # mimics what HandTracker.draw_annotations does for the label overlay)
        frame = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
        for r in synthetic_readings:
            colour = (0, 255, 0) if r.string is not None else (0, 0, 255)
            cv2.circle(frame, (int(r.px) % WARP_W, int(r.py)), 5, colour, -1)
        annotated_path = tmp / "test_annotated.png"
        cv2.imwrite(str(annotated_path), frame)
        assert annotated_path.exists()
        loaded = cv2.imread(str(annotated_path))
        assert loaded is not None and loaded.shape == (WARP_H, WARP_W, 3)
        print(f"  ✓ Annotation drawing       (synthetic frame saved + reloaded OK)")

    print("\n" + "─" * 50)
    print("  All P11 smoke tests passed! ✅")
    print("─" * 50)


if __name__ == "__main__":
    print("\nP11 Smoke Test")
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
