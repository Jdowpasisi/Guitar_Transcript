"""
P13: GuitarAI v1 — Transcription API
======================================
GuitarAI v1 upgrades the P7 API with three input modes:

  1. Audio only     → POST /transcribe           (P7 unchanged)
  2. Video upload   → POST /transcribe_video     (P13 new)
  3. YouTube URL    → POST /transcribe_url       (P13 new)

All three routes return a job_id immediately (async). Poll /status/{id}
and fetch the full result from /result/{id}.

Endpoints:
  POST /transcribe          — Upload audio, returns job_id immediately
  POST /transcribe_video    — Upload video (audio + vision + fusion)
  POST /transcribe_url      — YouTube URL → yt-dlp download + full pipeline
  GET  /status/{job_id}    — Poll job state (PENDING / STARTED / SUCCESS / FAILURE)
  GET  /result/{job_id}    — Full transcription JSON once SUCCESS
  GET  /models             — Versions of every loaded model
  GET  /health             — Liveness check
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .celery_app import celery_app
from .schemas import (
    JobSubmittedResponse,
    JobStatusResponse,
    TranscriptionResult,
    ModelsInfoResponse,
    HealthResponse,
    YouTubeDownloadRequest,
)
from .tasks import run_pipeline, run_pipeline_with_video, run_pipeline_from_url  # noqa: F401

# ─── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GuitarAI v1 Transcription API",
    description=(
        "GuitarAI v1: Full multimodal guitar transcription system. "
        "Accepts audio, video, or a YouTube URL. "
        "Routes through FusionModel (P12) when video is available, "
        "LSTM (P6) for audio-only. Returns chord chart + guitar tablature."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload staging directory (inside container)
UPLOAD_DIR = Path("/tmp/guitarai_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

MAX_AUDIO_SIZE_MB = 100
MAX_VIDEO_SIZE_MB = 500   # Videos are large


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness probe — returns 200 immediately."""
    return HealthResponse(status="ok", version=app.version)


@app.get("/models", response_model=ModelsInfoResponse, tags=["Meta"])
async def models_info():
    """
    Returns metadata for every model loaded in the GuitarAI v1 pipeline.
    """
    return ModelsInfoResponse(
        models={
            "stem_splitter": {
                "name": "Demucs htdemucs",
                "version": "4.0.1",
                "description": "P2: Separates guitar from mixed audio",
                "type": "engineering",
            },
            "chord_cnn": {
                "name": "ChordCNN",
                "version": "1.0.0",
                "checkpoint": "models/chord_cnn.pth",
                "num_classes": 51,
                "description": "P4: 3-block CNN classifies chords from CQT spectrograms",
                "type": "trained",
                "benchmark": "26.4% chord accuracy on GuitarSet test set",
            },
            "basic_pitch": {
                "name": "Basic Pitch (Spotify)",
                "version": "0.4.0",
                "backend": "ONNX",
                "description": "P5: Audio-to-MIDI note transcription",
                "type": "pretrained",
            },
            "voicing_lstm": {
                "name": "VoicingLSTM",
                "version": "1.0.0",
                "checkpoint": "models/voicing_lstm.pth",
                "output_classes": 138,
                "description": "P6: Bi-LSTM maps MIDI notes to (string, fret) positions",
                "type": "trained",
                "benchmark": "35.0% Tab Accuracy on GuitarSet test set",
            },
            "chord_shape_cnn": {
                "name": "ChordShapeCNN",
                "version": "1.0.0",
                "checkpoint": "models/chord_shape_cnn.pth",
                "num_classes": 7,
                "description": "P10: CNN classifies chord shapes from warped fretboard images",
                "type": "trained",
            },
            "fusion_model": {
                "name": "FusionModel (Cross-Attention Transformer)",
                "version": "1.0.0",
                "checkpoint": "models/fusion_model.pth",
                "parameters": "1.14M",
                "output_classes": 138,
                "description": "P12: Multimodal cross-attention model fusing audio (56d) + video (7d) features",
                "type": "trained",
                "benchmark": "83.8% Tab Accuracy (fused) vs 71.2% audio-only on GuitarSet test set",
            },
        }
    )


@app.post("/transcribe", response_model=JobSubmittedResponse, status_code=202, tags=["Transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Accept an audio upload, save to staging, enqueue audio-only pipeline.

    Pipeline: Demucs → Basic Pitch → ChordCNN → VoicingLSTM → ASCII tab
    Returns a job_id within ~100ms regardless of file size.
    """
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_AUDIO_SIZE_MB} MB",
        )

    job_id = str(uuid.uuid4())
    staging_path = UPLOAD_DIR / f"{job_id}{suffix}"
    staging_path.write_bytes(contents)

    run_pipeline.apply_async(
        args=[str(staging_path), job_id],
        task_id=job_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        status="PENDING",
        message="Audio job queued. Poll /status/{job_id} to track progress.",
        filename=file.filename,
        size_mb=round(size_mb, 2),
        has_video=False,
    )


@app.post("/transcribe_video", response_model=JobSubmittedResponse, status_code=202, tags=["Transcription"])
async def transcribe_video(file: UploadFile = File(...)):
    """
    Accept a video upload. Runs the full GuitarAI v1 multimodal pipeline:

    Audio branch (parallel):   Demucs → Basic Pitch → ChordCNN
    Vision branch (parallel):  P9 frames → P10 neck detect → P11 finger tracking
    Fusion:                    FusionModel (P12) combines audio + video → voicings
    Output:                    ASCII tab with per-note source badge (fusion/lstm/greedy)

    Returns a job_id within ~100ms. Video files up to 500 MB accepted.
    """
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum for video: {MAX_VIDEO_SIZE_MB} MB",
        )

    job_id = str(uuid.uuid4())

    # Save video to staging
    video_staging_path = UPLOAD_DIR / f"{job_id}_video{suffix}"
    video_staging_path.write_bytes(contents)

    # Audio will be extracted from video by the vision pipeline
    # Pass the video path as both audio_path (for initial load check) and video_path
    run_pipeline_with_video.apply_async(
        args=[str(video_staging_path), str(video_staging_path), job_id],
        kwargs={"video_source": "upload"},
        task_id=job_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        status="PENDING",
        message=(
            "Multimodal job queued. Audio + vision pipelines will run in parallel. "
            "Poll /status/{job_id} to track progress."
        ),
        filename=file.filename,
        size_mb=round(size_mb, 2),
        has_video=True,
    )


@app.post("/transcribe_url", response_model=JobSubmittedResponse, status_code=202, tags=["Transcription"])
async def transcribe_url(request: YouTubeDownloadRequest):
    """
    Accept a YouTube (or yt-dlp-compatible) URL. Downloads video via yt-dlp,
    then runs the full GuitarAI v1 multimodal pipeline.

    The entire pipeline runs asynchronously. Poll /status/{job_id}.
    Download + processing may take 2–5 minutes depending on video length.

    Example body:
        {"url": "https://www.youtube.com/watch?v=VIDEO_ID"}
    """
    url = str(request.url).strip()
    if not url:
        raise HTTPException(status_code=422, detail="url field is required")

    # Basic URL sanity check
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=422, detail="url must be a valid HTTP/HTTPS URL")

    job_id = str(uuid.uuid4())

    run_pipeline_from_url.apply_async(
        args=[url, job_id],
        task_id=job_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        status="PENDING",
        message=(
            f"YouTube download + transcription job queued. "
            f"Downloading: {url[:60]}{'...' if len(url) > 60 else ''}. "
            "Poll /status/{job_id} to track progress."
        ),
        filename=url,
        size_mb=None,
        has_video=True,
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Transcription"])
async def job_status(job_id: str):
    """
    Returns the current Celery task state.

    States: PENDING → STARTED → SUCCESS | FAILURE
    """
    result = celery_app.AsyncResult(job_id)

    state = result.state
    meta = {}

    if state == "STARTED":
        meta = result.info or {}
    elif state == "FAILURE":
        meta = {"error": str(result.result)}

    return JobStatusResponse(
        job_id=job_id,
        status=state,
        meta=meta,
    )


@app.get("/result/{job_id}", response_model=TranscriptionResult, tags=["Transcription"])
async def job_result(job_id: str):
    """
    Returns the full transcription once the job is SUCCESS.
    Returns 202 if still processing, 500 if the job failed.
    """
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        raise HTTPException(status_code=202, detail="Job is pending. Try again shortly.")
    if result.state == "STARTED":
        raise HTTPException(status_code=202, detail="Job is still processing.")
    if result.state == "FAILURE":
        raise HTTPException(status_code=500, detail=f"Job failed: {result.result}")
    if result.state != "SUCCESS":
        raise HTTPException(status_code=202, detail=f"Job state: {result.state}")

    return TranscriptionResult(**result.result)
