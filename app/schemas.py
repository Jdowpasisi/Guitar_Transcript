"""
P7 API Schemas
==============
Pydantic v2 models for every request/response shape.
These also auto-generate the /docs Swagger UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Meta ──────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(json_schema_extra={"example": "ok"})
    version: str = Field(json_schema_extra={"example": "1.0.0"})


class ModelsInfoResponse(BaseModel):
    models: Dict[str, Dict[str, Any]] = Field(
        description="Keyed by internal model name; value contains metadata dict."
    )


# ─── Job lifecycle ─────────────────────────────────────────────────────────────

class JobSubmittedResponse(BaseModel):
    job_id: str = Field(json_schema_extra={"example": "3fa85f64-5717-4562-b3fc-2c963f66afa6"})
    status: str = Field(json_schema_extra={"example": "PENDING"})
    message: str
    filename: Optional[str] = None
    size_mb: Optional[float] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(
        description="One of: PENDING, STARTED, SUCCESS, FAILURE",
        json_schema_extra={"example": "STARTED"},
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Progress info (step, percent) while STARTED; error text on FAILURE.",
    )


# ─── Transcription result ──────────────────────────────────────────────────────

class ChordEvent(BaseModel):
    """One chord detection with timing."""
    start: float = Field(description="Start time in seconds", json_schema_extra={"example": 0.0})
    end: float = Field(description="End time in seconds", json_schema_extra={"example": 1.0})
    label: str = Field(description="Chord label e.g. 'G:maj'", json_schema_extra={"example": "G:maj"})
    confidence: float = Field(description="Model confidence [0, 1]", json_schema_extra={"example": 0.87})


class NoteEvent(BaseModel):
    """One transcribed note with guitar-specific voicing."""
    onset: float = Field(description="Note onset in seconds", json_schema_extra={"example": 0.12})
    offset: float = Field(description="Note offset in seconds", json_schema_extra={"example": 0.48})
    pitch: int = Field(description="MIDI pitch number", json_schema_extra={"example": 55})
    pitch_name: str = Field(description="Human-readable pitch", json_schema_extra={"example": "G3"})
    string: int = Field(description="Guitar string index (0=E2 … 5=E4)", json_schema_extra={"example": 3})
    fret: int = Field(description="Fret number (0–22)", json_schema_extra={"example": 0})
    string_name: str = Field(description="String note name", json_schema_extra={"example": "G3"})
    confidence: float = Field(description="Note detection confidence [0, 1]", json_schema_extra={"example": 0.91})
    voicing_source: str = Field(
        description="'lstm' | 'greedy' | 'heuristic'",
        json_schema_extra={"example": "lstm"},
    )


class PipelineInfo(BaseModel):
    """Which models ran and key performance stats."""
    stem_separation: bool = Field(description="Was Demucs stem separation applied?")
    models_used: List[str] = Field(
        description="Ordered list of models that ran",
        json_schema_extra={"example": ["Demucs htdemucs", "Basic Pitch ONNX", "ChordCNN", "VoicingLSTM"]},
    )
    audio_duration_sec: float
    processing_time_sec: float
    note_count: int
    chord_count: int


class TranscriptionResult(BaseModel):
    """
    Full pipeline output — returned by GET /result/{job_id}.
    """
    job_id: str

    # Core outputs
    chords: List[ChordEvent] = Field(
        description="Chord timeline — one entry per detected chord segment"
    )
    tab: str = Field(
        description=(
            "6-line ASCII guitar tablature string. "
            "Each line is one guitar string (high E at top, low E at bottom)."
        ),
        json_schema_extra={
            "example": (
                "e|--0---3---|\n"
                "B|--1---0---|\n"
                "G|--0---0---|\n"
                "D|--2---0---|\n"
                "A|--3---2---|\n"
                "E|--x---3---|"
            ),
        },
    )
    notes: List[NoteEvent] = Field(
        description="Per-note detail including MIDI pitch and (string, fret) assignment"
    )

    # Pipeline metadata
    pipeline: PipelineInfo
