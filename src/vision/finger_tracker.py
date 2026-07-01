"""
P11 — Finger Tracker + Fret Mapper
====================================
Tracks 21 hand landmarks per frame with MediaPipe Hands, transforms
fingertip pixel coordinates through the P9 homography into fretboard
space, then discretises that position into a (string, fret) grid cell
using the equal-temperament fret-spacing formula.

── Pipeline ──────────────────────────────────────────────────────────────────
  Video frame
      │
      ▼
  MediaPipe HandLandmarker  → 21 landmarks/hand, left/right handedness
      │
      ▼
  Extract fingertips        → landmarks 4, 8, 12, 16, 20 (thumb..pinky)
      │
      ▼
  Normalised → pixel coords → landmark.x * frame_w, landmark.y * frame_h
      │
      ▼
  cv2.perspectiveTransform  → maps pixel point through P9 homography.npy
                               into the flat 600×200 fretboard image space
      │
      ▼
  Fret-grid lookup           → equal-temperament boundaries (precomputed)
                               (x_fretboard, y_fretboard) → (string, fret)
      │
      ▼
  CSV row + annotated frame

── Usage ─────────────────────────────────────────────────────────────────────
    python -m src.vision.finger_tracker \\
        outputs/frames/my_video/frames/ \\
        outputs/frames/my_video/homography.npy \\
        --output_dir outputs/frames/my_video

Or run the full thing on a fresh video (calls P9 extract_frames first):
    python -m src.vision.finger_tracker --video my_video.mp4

── Outputs ───────────────────────────────────────────────────────────────────
<output_dir>/finger_tracking.csv        — timestamp, finger_id, string, fret, confidence
<output_dir>/annotated/frame_NNNNNN.png — hand skeleton + string/fret labels
<output_dir>/annotated_video.mp4        — stitched annotated video (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Guitar / fretboard constants ────────────────────────────────────────────
# Matches P9 warp output and the rest of the codebase's tuning convention.
WARP_W = 600   # px — matches src/vision/warp_fretboard.py WARP_W
WARP_H = 200   # px — matches src/vision/warp_fretboard.py WARP_H

N_STRINGS = 6
N_FRETS   = 22   # frets 0 (open) .. 22

# String order in the warped image, top row (y≈0) → bottom row (y≈WARP_H):
# P9 corner convention is TL=nut/high-E, BL=nut/low-E, so row 0 = high-E (string 5
# in GuitarSet's low-to-high indexing) down to row 5 = low-E (string 0).
# We expose both: WARP_ROW_TO_STRING converts a pixel row index -> GuitarSet string index.
WARP_ROW_TO_STRING = [5, 4, 3, 2, 1, 0]   # row 0..5 (high E..low E) -> string idx 5..0
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]  # index 0..5, GuitarSet convention

# MediaPipe fingertip landmark indices (thumb, index, middle, ring, pinky tips)
FINGERTIP_LANDMARKS = {
    4:  "thumb",
    8:  "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
MAX_NUM_HANDS            = 2

# ── MediaPipe model asset ────────────────────────────────────────────────────
# The new MediaPipe Tasks API (0.10.x+) requires a .task model file.
# We store it alongside other models in the models/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HAND_LANDMARKER_MODEL_PATH = _PROJECT_ROOT / "models" / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model_downloaded(model_path: Path = HAND_LANDMARKER_MODEL_PATH) -> Path:
    """Download the HandLandmarker .task model if it doesn't exist."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading MediaPipe HandLandmarker model to {model_path}…")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, str(model_path))
    print(f"   ✓ Downloaded ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return model_path


# ── Equal-temperament fret grid ──────────────────────────────────────────────

def compute_fret_boundaries(neck_length_px: float = WARP_W, n_frets: int = N_FRETS) -> np.ndarray:
    """
    Compute fret BOUNDARY x-positions (in warped-image pixels) using the
    standard luthier equal-temperament formula:

        distance_from_nut(N) = scale_length * (1 - 2^(-N/12))

    Since the warped fretboard image only shows the playable region (nut to
    ~22nd fret), we treat neck_length_px as the position of fret 22 and back
    out an equivalent "scale length" so fret 22 lands at the right edge of
    the image. This keeps the relative spacing physically correct (frets
    closer together near the body) without needing the guitar's real-world
    scale length.

    Returns:
        boundaries: array of shape (n_frets + 1,) — boundary positions for
                    frets 0..n_frets, i.e. boundaries[i] is the LEFT edge of
                    fret i's region, boundaries[i+1] is its right edge.
                    boundaries[0] = 0 (the nut).
                    boundaries[-1] = neck_length_px (right edge of image).
    """
    # Solve for scale_length such that position(n_frets) == neck_length_px
    # position(N) = scale * (1 - 2^(-N/12))  =>  scale = neck_length_px / (1 - 2^(-n_frets/12))
    denom = 1 - 2 ** (-n_frets / 12)
    scale_length = neck_length_px / denom

    # Fret N's wire sits at position(N). The boundary between fret N and N+1
    # is the midpoint convention we use here: we bucket a fingertip x-coord
    # into fret N if it falls in [position(N-1), position(N)) — i.e. between
    # the (N-1)th and Nth fret wires, which is where a finger physically
    # presses to sound fret N. Fret 0 (open string) covers [0, position(1)/2]
    # is NOT used — instead we special-case "before fret 1's wire" as fret 0.
    positions = np.array([
        scale_length * (1 - 2 ** (-n / 12)) for n in range(n_frets + 1)
    ])  # positions[0] = 0 (nut), positions[22] ≈ neck_length_px

    return positions  # use directly as boundaries; see x_to_fret()


def x_to_fret(x_fretboard: float, fret_positions: np.ndarray) -> int:
    """
    Map a warped-image x-coordinate to a fret number using the
    precomputed equal-temperament fret-wire positions.

    A fretting finger presses just BEHIND (toward the body from) the fret
    wire that sounds that note — i.e. to sound fret N you press between
    wire N-1 and wire N. So we find which inter-wire bucket x falls into:

        x in [fret_positions[N-1], fret_positions[N])  ->  fret N

    fret_positions[0] = 0 (the nut)
    fret_positions[k] = position of fret k's wire, k = 1..N_FRETS

    Note: this function only returns fretted positions (1..N_FRETS) — a
    finger anywhere between the nut and fret 1's wire is assigned to fret 1,
    since that's the nearest fret it could be pressing. Fret 0 (open string,
    no finger) is never returned here; it's the natural state of a string
    with no detected finger on it, handled by the caller simply not emitting
    a reading for that string.

    Returns fret number 1..N_FRETS (clamped at both ends).
    """
    x = max(0.0, min(x_fretboard, fret_positions[-1]))
    # side='left': x exactly at a fret wire's position rounds UP to that fret
    # (pressing right at the wire still sounds that fret, by luthier convention)
    fret = int(np.searchsorted(fret_positions, x, side="left"))
    return max(1, min(fret, N_FRETS))


def y_to_string(y_fretboard: float, warp_h: int = WARP_H) -> int:
    """
    Map a warped-image y-coordinate to a guitar string index (0=low E2, 5=high E4).

    The warped image has 6 implicit string rows evenly spaced top-to-bottom,
    with row 0 (y≈0) = high E (string idx 5) and row 5 (y≈warp_h) = low E
    (string idx 0), matching the P9 corner convention (TL/TR = high-E side).
    """
    y = max(0.0, min(y_fretboard, float(warp_h)))
    row = int(y / warp_h * N_STRINGS)
    row = max(0, min(row, N_STRINGS - 1))
    return WARP_ROW_TO_STRING[row]


# ── Homography point transform ───────────────────────────────────────────────

def load_homography(path: str | Path) -> np.ndarray:
    """Load the 3x3 homography matrix saved by P9 warp_fretboard.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Homography not found: {path}\n"
            "Run P9 first to generate homography.npy:\n"
            "  python -m src.vision.frame_detective <video>"
        )
    H = np.load(str(path))
    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography, got shape {H.shape}")
    return H


def transform_point(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """
    Map a single pixel point (px, py) in the ORIGINAL frame through the
    homography H into the WARPED fretboard image's coordinate space.

    Uses cv2.perspectiveTransform, which expects shape (1, 1, 2).
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    warped_pt = cv2.perspectiveTransform(pt, H)
    wx, wy = warped_pt[0, 0]
    return float(wx), float(wy)


def point_in_warp_bounds(wx: float, wy: float, margin: float = 10.0) -> bool:
    """Check whether a warped-space point lands within the fretboard rectangle
    (with a small margin to tolerate slight overshoot from imprecise homography)."""
    return (-margin <= wx <= WARP_W + margin) and (-margin <= wy <= WARP_H + margin)


# ── Per-frame tracking result ─────────────────────────────────────────────────

@dataclass
class FingerReading:
    frame_idx:   int
    timestamp:   float
    hand_label:  str     # "Left" | "Right"
    finger_id:   str     # "thumb" | "index" | "middle" | "ring" | "pinky"
    px:          float   # original-frame pixel x
    py:          float   # original-frame pixel y
    string:      Optional[int]  # 0-5, or None if off-fretboard
    fret:        Optional[int]  # 0-22, or None if off-fretboard
    confidence:  float   # MediaPipe landmark visibility/presence proxy


# ── MediaPipe wrapper ─────────────────────────────────────────────────────────

class HandTracker:
    """
    Thin wrapper around MediaPipe HandLandmarker (Tasks API, 0.10.x+) that
    extracts fingertip landmarks and maps them through a homography into
    (string, fret).

    The new Tasks API replaces the deprecated mp.solutions.hands interface.
    Key differences:
      - Requires a .task model file (auto-downloaded if missing)
      - Uses RunningMode.VIDEO with timestamp_ms for sequential frame processing
      - Landmarks are returned as lists of NormalizedLandmark objects
      - Drawing uses mp.tasks.vision.drawing_utils
    """

    def __init__(
        self,
        homography: np.ndarray,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float  = MIN_TRACKING_CONFIDENCE,
        model_path: str | Path | None = None,
        running_mode: str = "VIDEO",
    ):
        try:
            import mediapipe as mp
            # Verify the Tasks API is available (mp.tasks.vision.HandLandmarker)
            _ = mp.tasks.vision.HandLandmarker
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "mediapipe not installed (or installed incorrectly).\n"
                "Install with:  pip install mediapipe\n"
                f"(underlying error: {exc})"
            )

        self._mp = mp

        # Resolve model path
        if model_path is None:
            model_path = _ensure_model_downloaded()
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"HandLandmarker model not found: {model_path}\n"
                    "Download from: " + HAND_LANDMARKER_MODEL_URL
                )

        # Select running mode
        vision = mp.tasks.vision
        if running_mode == "IMAGE":
            rm = vision.RunningMode.IMAGE
        elif running_mode == "VIDEO":
            rm = vision.RunningMode.VIDEO
        else:
            rm = vision.RunningMode.IMAGE

        self._running_mode = rm

        # Create HandLandmarker with the new Tasks API
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=rm,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Store references for drawing
        self._drawing_utils = vision.drawing_utils
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

        self.H = homography
        self.fret_positions = compute_fret_boundaries(WARP_W, N_FRETS)

        # Track the last timestamp used (for VIDEO mode monotonicity)
        self._last_timestamp_ms = -1

    def close(self):
        self.landmarker.close()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> tuple[list[FingerReading], object]:
        """
        Run MediaPipe HandLandmarker on one BGR frame.

        Returns:
            (readings, mediapipe_result) — readings is a list of
            FingerReading, one per detected fingertip across all hands.
            mediapipe_result is the raw HandLandmarkerResult, useful for
            drawing the full hand skeleton.
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Run detection based on running mode
        import mediapipe as mp
        vision = mp.tasks.vision
        if self._running_mode == vision.RunningMode.VIDEO:
            timestamp_ms = max(int(timestamp * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        readings: list[FingerReading] = []
        if not result.hand_landmarks:
            return readings, result

        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            # Get handedness
            hand_label = "Unknown"
            hand_score = 0.0
            if result.handedness and hand_idx < len(result.handedness):
                # handedness is a list of lists of Category
                categories = result.handedness[hand_idx]
                if categories:
                    hand_label = categories[0].category_name  # "Left" | "Right"
                    hand_score = float(categories[0].score)

            for lm_idx, finger_name in FINGERTIP_LANDMARKS.items():
                lm = hand_lms[lm_idx]
                px, py = lm.x * w, lm.y * h

                string_idx: Optional[int] = None
                fret_idx:   Optional[int] = None

                wx, wy = transform_point(px, py, self.H)
                if point_in_warp_bounds(wx, wy):
                    string_idx = y_to_string(wy)
                    fret_idx   = x_to_fret(wx, self.fret_positions)

                # Use landmark visibility/presence if available, else
                # fall back to handedness classification score.
                visibility = getattr(lm, "visibility", None)
                presence = getattr(lm, "presence", None)
                if visibility is not None and visibility > 0:
                    confidence = float(visibility)
                elif presence is not None and presence > 0:
                    confidence = float(presence)
                else:
                    confidence = hand_score

                readings.append(FingerReading(
                    frame_idx=frame_idx,
                    timestamp=round(timestamp, 4),
                    hand_label=hand_label,
                    finger_id=finger_name,
                    px=round(px, 1),
                    py=round(py, 1),
                    string=string_idx,
                    fret=fret_idx,
                    confidence=round(confidence, 4),
                ))

        return readings, result

    def draw_annotations(
        self,
        frame: np.ndarray,
        readings: list[FingerReading],
        mp_result,
    ) -> np.ndarray:
        """
        Draw the full hand skeleton (MediaPipe connections) plus
        a string/fret text label at each tracked fingertip.
        """
        out = frame.copy()

        # Draw hand skeletons using the new Tasks drawing API
        if mp_result.hand_landmarks:
            for hand_lms in mp_result.hand_landmarks:
                self._drawing_utils.draw_landmarks(
                    out,
                    hand_lms,
                    self._hand_connections,
                )

        # Draw string/fret labels at each fingertip
        for r in readings:
            colour = (0, 255, 0) if r.string is not None else (0, 0, 255)
            cv2.circle(out, (int(r.px), int(r.py)), 5, colour, -1)

            if r.string is not None and r.fret is not None:
                label = f"{STRING_NAMES[r.string]} fr{r.fret}"
            else:
                label = "off-board"

            cv2.putText(
                out, label, (int(r.px) + 8, int(r.py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                out, label, (int(r.px) + 8, int(r.py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA,
            )

        return out


# ── Batch processing over a frame directory ───────────────────────────────────

def process_frame_directory(
    frames_dir:   str | Path,
    homography_path: str | Path,
    output_dir:   str | Path | None = None,
    fps:          float = 5.0,
    save_annotated: bool = True,
    make_video:   bool = False,
    verbose:      bool = True,
) -> dict:
    """
    Run finger tracking over every frame in a directory (output of P9
    extract_frames), writing a CSV + annotated frames.

    Args:
        frames_dir:       Directory of frame_*.png files.
        homography_path:  Path to P9's homography.npy.
        output_dir:       Where to write finger_tracking.csv + annotated/.
                          Defaults to frames_dir's parent.
        fps:              Frame rate used during P9 extraction — needed to
                          compute correct timestamps (frame_idx / fps).
        save_annotated:   Whether to save per-frame annotated PNGs.
        make_video:       Whether to stitch annotated frames into an .mp4.

    Returns:
        Summary dict with counts + output paths.
    """
    frames_dir = Path(frames_dir)
    if output_dir is None:
        output_dir = frames_dir.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated"
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    H = load_homography(homography_path)
    tracker = HandTracker(H)

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    csv_path = output_dir / "finger_tracking.csv"
    all_readings: list[FingerReading] = []
    n_frames_with_hand = 0

    if verbose:
        print(f"\n👆 P11 — Finger Tracking")
        print(f"   Frames:      {len(frames)} in {frames_dir}")
        print(f"   Homography:  {homography_path}")
        print(f"   FPS:         {fps}")
        print(f"   Output:      {output_dir}\n")

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "finger_id", "string", "fret", "confidence"])

            for i, fp in enumerate(frames):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue

                frame_idx = i
                timestamp = frame_idx / fps

                readings, mp_results = tracker.process_frame(frame, frame_idx, timestamp)

                if readings:
                    n_frames_with_hand += 1
                all_readings.extend(readings)

                for r in readings:
                    writer.writerow([
                        r.timestamp,
                        r.finger_id,
                        r.string if r.string is not None else "",
                        r.fret if r.fret is not None else "",
                        r.confidence,
                    ])

                if save_annotated:
                    annotated = tracker.draw_annotations(frame, readings, mp_results)
                    cv2.imwrite(str(annotated_dir / fp.name), annotated)

                if verbose and ((i + 1) % 10 == 0 or i == len(frames) - 1):
                    print(f"   [{i+1}/{len(frames)}] {fp.name} — "
                          f"{len(readings)} fingertips tracked")
    finally:
        tracker.close()

    # Optionally stitch annotated frames into a video
    video_path = None
    if make_video and save_annotated:
        video_path = output_dir / "annotated_video.mp4"
        _stitch_video(annotated_dir, video_path, fps=fps)

    n_on_board = sum(1 for r in all_readings if r.string is not None)
    pct_on_board = n_on_board / max(len(all_readings), 1) * 100

    summary = {
        "frames_processed":     len(frames),
        "frames_with_hand":     n_frames_with_hand,
        "pct_frames_with_hand": round(n_frames_with_hand / len(frames) * 100, 1),
        "total_fingertip_readings": len(all_readings),
        "readings_on_fretboard":    n_on_board,
        "pct_on_fretboard":         round(pct_on_board, 1),
        "csv_path":             str(csv_path),
        "annotated_dir":        str(annotated_dir) if save_annotated else None,
        "video_path":           str(video_path) if video_path else None,
    }

    summary_path = output_dir / "finger_tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  ✅ Finger tracking complete!")
        print(f"  Frames with a hand detected: {n_frames_with_hand}/{len(frames)} "
              f"({summary['pct_frames_with_hand']}%)")
        print(f"  Fingertip readings on fretboard: {n_on_board}/{len(all_readings)} "
              f"({pct_on_board:.0f}%)")
        print(f"  CSV:       {csv_path}")
        if save_annotated:
            print(f"  Annotated: {annotated_dir}")
        if video_path:
            print(f"  Video:     {video_path}")
        print(f"{'═'*55}\n")

    return summary


def _stitch_video(frames_dir: Path, out_path: Path, fps: float = 5.0):
    """Stitch annotated PNG frames into an .mp4 using OpenCV's VideoWriter."""
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for fp in frames:
        frame = cv2.imread(str(fp))
        writer.write(frame)
    writer.release()


# ── Full pipeline: video → P9 extract → P11 track ─────────────────────────────

def run_on_video(
    video_path: str | Path,
    output_root: str | Path = "outputs/frames",
    fps: float = 5.0,
    homography_path: str | Path | None = None,
    make_video: bool = True,
) -> dict:
    """
    Convenience wrapper: if a P9 job directory already exists for this video
    (frames/ + homography.npy), reuse it. Otherwise run P9 extraction first
    (homography still requires manual corner clicking from P9, or P10's
    auto-warp neck detector if trained).
    """
    from .extract_frames import extract_frames

    video_path = Path(video_path)
    stem       = video_path.stem
    job_dir    = Path(output_root) / stem
    frames_dir = job_dir / "frames"

    if homography_path is None:
        homography_path = job_dir / "homography.npy"
    homography_path = Path(homography_path)

    if not frames_dir.exists() or not any(frames_dir.glob("frame_*.png")):
        print(f"No extracted frames found at {frames_dir} — running P9 extraction…")
        extract_frames(video_path, output_root, fps=fps, overwrite=False)

    if not homography_path.exists():
        raise FileNotFoundError(
            f"No homography found at {homography_path}.\n"
            "Run P9's interactive corner picker first:\n"
            f"  python -m src.vision.frame_detective {video_path}\n"
            "or P10's auto-warp neck detector:\n"
            f"  python -m src.vision.neck_detector --auto_warp {video_path}"
        )

    return process_frame_directory(
        frames_dir, homography_path, output_dir=job_dir,
        fps=fps, save_annotated=True, make_video=make_video,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P11: Track fingers, map to (string, fret) via P9 homography.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", type=str, help="Run full pipeline on a video file")
    mode.add_argument("frames_dir", nargs="?", type=str, help="Directory of pre-extracted frames")

    parser.add_argument("homography", nargs="?", type=str,
                        help="Path to homography.npy (required with frames_dir)")
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--fps",         type=float, default=5.0)
    parser.add_argument("--no_annotate", action="store_true", help="Skip saving annotated frames")
    parser.add_argument("--make_video",  action="store_true", help="Stitch annotated frames into mp4")
    args = parser.parse_args()

    try:
        if args.video:
            run_on_video(args.video, fps=args.fps, make_video=True)
        else:
            if not args.frames_dir or not args.homography:
                print("❌ Both frames_dir and homography path are required "
                      "(or use --video instead).", file=sys.stderr)
                sys.exit(1)
            process_frame_directory(
                args.frames_dir, args.homography,
                output_dir=args.output_dir, fps=args.fps,
                save_annotated=not args.no_annotate,
                make_video=args.make_video,
            )
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
