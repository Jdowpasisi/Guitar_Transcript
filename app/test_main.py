"""
P7: Unit + Integration Tests for the Transcription API
=======================================================
Run with: pytest app/test_main.py -v

Tests that don't require a real Redis/Celery:
  - Health endpoint
  - Models endpoint
  - /transcribe validation (bad extension, file too large)
  - /status with a fake job id
  - /result 202 for pending job

Integration tests (require docker-compose to be running):
  - Full round-trip: upload → poll → result
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from .main import app, UPLOAD_DIR

client = TestClient(app, raise_server_exceptions=False)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_fake_audio(suffix: str = ".mp3", size_bytes: int = 1024) -> bytes:
    """Minimal fake audio bytes (not a real audio file — for upload validation only)."""
    return b"\x00" * size_bytes


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ─── Models ───────────────────────────────────────────────────────────────────

def test_models():
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert len(body["models"]) >= 4  # chord_cnn, voicing_lstm, basic_pitch, stem_splitter
    for key in ("chord_cnn", "voicing_lstm", "basic_pitch", "stem_splitter"):
        assert key in body["models"], f"Expected model '{key}' in /models response"


# ─── /transcribe validation ────────────────────────────────────────────────────

def test_transcribe_rejects_non_audio():
    """Only audio extensions allowed."""
    resp = client.post(
        "/transcribe",
        files={"file": ("document.pdf", _make_fake_audio(), "application/pdf")},
    )
    assert resp.status_code == 415
    assert "Unsupported" in resp.json()["detail"]


def test_transcribe_rejects_oversized_file():
    """Files > 100 MB must be rejected."""
    large = _make_fake_audio(size_bytes=101 * 1024 * 1024)
    resp = client.post(
        "/transcribe",
        files={"file": ("huge.mp3", large, "audio/mpeg")},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


@patch("app.tasks.run_pipeline.apply_async")
def test_transcribe_returns_job_id_fast(mock_apply_async):
    """
    POST /transcribe must return a job_id within << 1s regardless of file size.
    Celery task is mocked — we don't actually run the ML pipeline here.
    """
    mock_apply_async.return_value = MagicMock(id="fake-job-id")
    t0 = time.monotonic()
    resp = client.post(
        "/transcribe",
        files={"file": ("song.mp3", _make_fake_audio(size_bytes=4096), "audio/mpeg")},
    )
    elapsed = time.monotonic() - t0

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "PENDING"
    assert elapsed < 3.0, f"Upload took {elapsed:.2f}s — should be < 3s"


# ─── /status ─────────────────────────────────────────────────────────────────

def test_status_unknown_job():
    """A random UUID that was never submitted should return PENDING (Celery's default)."""
    resp = client.get("/status/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("PENDING", "FAILURE")


# ─── /result 202 for pending jobs ─────────────────────────────────────────────

def test_result_returns_202_for_pending_job():
    """Fetching /result for an unknown job returns 202 (not yet ready)."""
    resp = client.get("/result/00000000-0000-0000-0000-000000000001")
    # Should be 202 (pending) or 500 (failure) — never 200 for a job that doesn't exist
    assert resp.status_code in (202, 500)


# ─── CORS headers ─────────────────────────────────────────────────────────────

def test_cors_headers_present():
    """API must include CORS headers for the browser app (P8)."""
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" in resp.headers


# ─── Integration test (requires running docker-compose) ───────────────────────
# Mark with: pytest -m integration
# These are skipped in CI unless INTEGRATION=1 is set.

import os as _os

pytestmark_integration = pytest.mark.skipif(
    _os.getenv("INTEGRATION") != "1",
    reason="Set INTEGRATION=1 to run integration tests against a live stack",
)


@pytestmark_integration
def test_full_round_trip(tmp_path):
    """
    Full end-to-end test against a live docker-compose stack.
    Creates a minimal silent WAV and verifies the pipeline completes.
    """
    import struct
    import wave

    # Write a tiny 1-second silent WAV
    wav_path = tmp_path / "silent.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 22050)  # 1 second of silence

    base = _os.getenv("API_URL", "http://localhost:8000")
    import httpx

    with httpx.Client(base_url=base, timeout=30) as http:
        # Submit
        with open(wav_path, "rb") as f:
            resp = http.post("/transcribe", files={"file": ("silent.wav", f, "audio/wav")})
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        # Poll
        for _ in range(60):
            r = http.get(f"/status/{job_id}")
            state = r.json()["status"]
            if state in ("SUCCESS", "FAILURE"):
                break
            time.sleep(2)

        assert state == "SUCCESS", f"Job ended with state: {state}"

        # Fetch result
        r = http.get(f"/result/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()

        assert "chords" in body
        assert "tab" in body
        assert "notes" in body
        assert "pipeline" in body
        assert body["pipeline"]["audio_duration_sec"] == pytest.approx(1.0, abs=0.1)
