"""
P7: Transcription API — Main FastAPI Application
=================================================
Endpoints:
  POST /transcribe          — Upload audio, returns job_id immediately
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
)
from .tasks import run_pipeline  # noqa: F401  (must be imported so Celery discovers it)

# ─── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GuitarAI Transcription API",
    description=(
        "Wraps the full P1–P6 ML pipeline behind an async REST interface. "
        "Upload audio → get a job_id → poll until done → fetch chord + tab results."
    ),
    version="1.0.0",
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
MAX_FILE_SIZE_MB = 100


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness probe — returns 200 immediately."""
    return HealthResponse(status="ok", version=app.version)


@app.get("/models", response_model=ModelsInfoResponse, tags=["Meta"])
async def models_info():
    """
    Returns metadata for every model loaded in the pipeline.
    Versions are baked in at build time; checksum the .pth files for identity.
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
            },
        }
    )


@app.post("/transcribe", response_model=JobSubmittedResponse, status_code=202, tags=["Transcription"])
async def transcribe(file: UploadFile = File(...)):
    """
    Accept an audio upload, save to staging, enqueue Celery task.

    Returns a job_id within ~100ms regardless of file size.
    """
    # Extension check
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # Read and size-check
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB",
        )

    # Persist to staging with a unique name
    job_id = str(uuid.uuid4())
    staging_path = UPLOAD_DIR / f"{job_id}{suffix}"
    staging_path.write_bytes(contents)

    # Enqueue — fire and forget
    run_pipeline.apply_async(
        args=[str(staging_path), job_id],
        task_id=job_id,
    )

    return JobSubmittedResponse(
        job_id=job_id,
        status="PENDING",
        message="Job queued. Poll /status/{job_id} to track progress.",
        filename=file.filename,
        size_mb=round(size_mb, 2),
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
        # Worker publishes progress info in task meta
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
