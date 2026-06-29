"""
P9 — Frame Extractor
=====================
Extract video frames (and audio) from a guitar video using FFmpeg.

Usage:
    python -m src.vision.extract_frames <video_path> [--fps 5] [--output_dir outputs/frames]

Outputs (inside <output_dir>/<video_stem>/):
    frames/   — PNG frames at the requested fps
    audio.wav — Full audio track (for P7 pipeline input)
    meta.json — Width, height, fps, total_frames, duration_sec, source

FFmpeg is called via subprocess — no Python binding needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], description: str) -> subprocess.CompletedProcess:
    """Run an FFmpeg command, raise on non-zero exit."""
    print(f"  [{description}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"FFmpeg failed: {description}")
    return result


def probe_video(video_path: str) -> dict:
    """
    Use ffprobe to get video metadata.
    Returns dict with: width, height, fps, duration_sec, total_frames.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr}")

    info = json.loads(result.stdout)
    meta = {"width": None, "height": None, "fps": None, "duration_sec": None, "total_frames": None}

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and meta["width"] is None:
            meta["width"]  = stream.get("width")
            meta["height"] = stream.get("height")
            meta["duration_sec"] = float(stream.get("duration", 0) or 0)
            meta["total_frames"] = int(stream.get("nb_frames", 0) or 0)

            # Parse fps — stored as "30000/1001" or "30"
            raw_fps = stream.get("r_frame_rate", "0/1")
            try:
                num, den = raw_fps.split("/")
                meta["fps"] = round(int(num) / int(den), 3)
            except (ValueError, ZeroDivisionError):
                meta["fps"] = 0.0

    return meta


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 5.0,
    overwrite: bool = False,
) -> dict:
    """
    Extract frames from a video at `fps` using FFmpeg, and rip the audio track.

    Args:
        video_path:  Path to the input video file.
        output_dir:  Root output directory. Files go into <output_dir>/<video_stem>/
        fps:         Frames per second to extract (default: 5.0)
        overwrite:   Overwrite existing outputs if True.

    Returns:
        metadata dict saved to meta.json
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Output layout
    stem = video_path.stem
    job_dir    = Path(output_dir) / stem
    frames_dir = job_dir / "frames"
    audio_path = job_dir / "audio.wav"
    meta_path  = job_dir / "meta.json"

    if job_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {job_dir}. Use --overwrite to re-extract."
        )

    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬 Extracting from: {video_path}")
    print(f"   Output dir:       {job_dir}")
    print(f"   Frame rate:       {fps} fps")

    # 1. Probe video metadata
    print("\n📋 Probing video metadata…")
    meta = probe_video(video_path)
    meta["source"]     = str(video_path.resolve())
    meta["extract_fps"] = fps

    if meta["duration_sec"]:
        expected_frames = int(meta["duration_sec"] * fps)
        print(f"   Duration:         {meta['duration_sec']:.1f}s")
        print(f"   Expected frames:  ~{expected_frames}")
    if meta["width"]:
        print(f"   Resolution:       {meta['width']}×{meta['height']} @ {meta['fps']}fps")

    # 2. Extract frames
    print("\n🖼  Extracting frames…")
    frames_pattern = str(frames_dir / "frame_%06d.png")
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",            # high quality PNG
        frames_pattern,
    ], "extract frames")

    # Count extracted frames
    extracted = sorted(frames_dir.glob("frame_*.png"))
    meta["extracted_frames"] = len(extracted)
    print(f"   ✓ Extracted {len(extracted)} frames → {frames_dir}")

    # 3. Extract audio
    print("\n🎵 Extracting audio track…")
    _run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # WAV PCM
        "-ar", "22050",         # match GuitarAI sample rate
        "-ac", "1",             # mono
        str(audio_path),
    ], "extract audio")

    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Audio saved → {audio_path} ({size_mb:.1f} MB)")
        meta["audio_path"] = str(audio_path)

    # 4. Save metadata
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n📄 Metadata saved → {meta_path}")

    print(f"\n✅ Done! {len(extracted)} frames, audio at {audio_path}")
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P9: Extract frames + audio from a guitar video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video",       type=str,   help="Path to input video file")
    parser.add_argument("--fps",       type=float, default=5.0,   help="Frames per second to extract")
    parser.add_argument("--output_dir",type=str,   default="outputs/frames", help="Root output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extraction")
    args = parser.parse_args()

    try:
        meta = extract_frames(args.video, args.output_dir, fps=args.fps, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, RuntimeError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
