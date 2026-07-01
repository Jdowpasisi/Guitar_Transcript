"""
P13: GuitarAI v1 Assembly — Smoke Test
=======================================
Validates that all P13 components work without real data, models, or network.

Tests:
  1.  Schemas accept the new P13 fields (has_video, fusion_used, video_source)
  2.  Tasks module imports successfully (all three Celery tasks discoverable)
  3.  FusionModel architecture instantiates (from P12)
  4.  Audio feature vector builder produces 56-dim output
  5.  Video feature vector builder produces 7-dim output
  6.  yt-dlp is importable
  7.  Vision pipeline function is importable
  8.  Greedy voicing produces valid (string, fret) pairs
  9.  ASCII tab renderer produces 6 lines
  10. API app object exists and has all expected routes
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0


def _test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅ {name}")
        PASS += 1
    except Exception as exc:
        print(f"  ❌ {name}: {exc}")
        traceback.print_exc()
        FAIL += 1


def run_smoke_test():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("P13: GuitarAI v1 Assembly — Smoke Test")
    print("=" * 60)

    # ── 1. Schemas ─────────────────────────────────────────────────────
    def test_schemas():
        from app.schemas import (
            JobSubmittedResponse, YouTubeDownloadRequest, PipelineInfo,
            TranscriptionResult, NoteEvent,
        )
        # has_video field on JobSubmittedResponse
        resp = JobSubmittedResponse(
            job_id="test", status="PENDING", message="ok", has_video=True
        )
        assert resp.has_video is True, "has_video not set"

        # YouTubeDownloadRequest
        req = YouTubeDownloadRequest(url="https://youtube.com/watch?v=test")
        assert req.url == "https://youtube.com/watch?v=test"

        # PipelineInfo with fusion fields
        pi = PipelineInfo(
            stem_separation=True, models_used=["FusionModel"],
            audio_duration_sec=10.0, processing_time_sec=2.0,
            note_count=50, chord_count=5,
            has_video=True, fusion_used=True, video_source="youtube",
        )
        assert pi.has_video is True
        assert pi.fusion_used is True
        assert pi.video_source == "youtube"

        # NoteEvent with 'fusion' voicing_source
        ne = NoteEvent(
            onset=0.1, offset=0.5, pitch=55, pitch_name="G3",
            string=3, fret=0, string_name="G3", confidence=0.9,
            voicing_source="fusion",
        )
        assert ne.voicing_source == "fusion"

    _test("Schemas accept P13 fields", test_schemas)

    # ── 2. Tasks module imports ────────────────────────────────────────
    def test_tasks_imports():
        from app.tasks import (
            run_pipeline,
            run_pipeline_with_video,
            run_pipeline_from_url,
            _run_vision_pipeline,
            _run_fusion_model,
            _download_youtube,
            _build_audio_feature_vector,
            _find_nearest_video_feature,
        )
        # All three Celery tasks should have .name attributes
        assert run_pipeline.name == "app.tasks.run_pipeline"
        assert run_pipeline_with_video.name == "app.tasks.run_pipeline_with_video"
        assert run_pipeline_from_url.name == "app.tasks.run_pipeline_from_url"

    _test("Tasks module imports (all 3 Celery tasks)", test_tasks_imports)

    # ── 3. FusionModel instantiation ───────────────────────────────────
    def test_fusion_model():
        from src.ml.fusion_model import FusionModel
        model = FusionModel()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000, f"Expected >1M params, got {n_params}"

    _test("FusionModel instantiates (~1.14M params)", test_fusion_model)

    # ── 4. Audio feature vector ────────────────────────────────────────
    def test_audio_feature():
        from app.tasks import _build_audio_feature_vector
        note = {"pitch": 60, "confidence": 0.9, "onset": 1.0, "offset": 1.5}
        vec = _build_audio_feature_vector(note, delta_t=0.3)
        assert len(vec) == 56, f"Expected 56 dims, got {len(vec)}"
        assert all(isinstance(v, float) for v in vec)

    _test("Audio feature vector = 56-dim", test_audio_feature)

    # ── 5. Video feature vector ────────────────────────────────────────
    def test_video_feature():
        from app.tasks import _find_nearest_video_feature

        # No video features → zero vector with video_available=0
        vec_none = _find_nearest_video_feature(1.0, [])
        assert len(vec_none) == 7, f"Expected 7 dims, got {len(vec_none)}"
        assert vec_none[-1] == 0.0, "video_available should be 0 when no features"

        # With video features
        video_feats = [
            {"timestamp": 1.05, "finger_id": 1, "string": 3, "fret": 2, "confidence": 0.8}
        ]
        vec = _find_nearest_video_feature(1.0, video_feats)
        assert len(vec) == 7
        assert vec[-1] == 1.0, "video_available should be 1.0 when features present"

    _test("Video feature vector = 7-dim", test_video_feature)

    # ── 6. yt-dlp importable ───────────────────────────────────────────
    def test_ytdlp():
        import yt_dlp
        assert hasattr(yt_dlp, 'YoutubeDL')

    _test("yt-dlp is importable", test_ytdlp)

    # ── 7. Vision pipeline function importable ─────────────────────────
    def test_vision_pipeline():
        from app.tasks import _run_vision_pipeline
        assert callable(_run_vision_pipeline)

    _test("Vision pipeline function importable", test_vision_pipeline)

    # ── 8. Greedy voicing ──────────────────────────────────────────────
    def test_greedy():
        from app.tasks import _greedy_voicing
        s, f = _greedy_voicing(60, prev_fret=0)  # C4 = MIDI 60
        assert 0 <= s <= 5, f"Invalid string: {s}"
        assert 0 <= f <= 22, f"Invalid fret: {f}"

    _test("Greedy voicing returns valid (string, fret)", test_greedy)

    # ── 9. ASCII tab renderer ──────────────────────────────────────────
    def test_tab_renderer():
        from app.tasks import _render_ascii_tab
        notes = [
            {"onset": 0.1, "string": 3, "fret": 0},
            {"onset": 0.5, "string": 1, "fret": 3},
        ]
        tab = _render_ascii_tab(notes, duration_sec=2.0)
        lines = tab.strip().split("\n")
        assert len(lines) == 6, f"Expected 6 tab lines, got {len(lines)}"
        assert lines[0].startswith("e|"), f"First line should be high e: {lines[0]}"
        assert lines[5].startswith("E|"), f"Last line should be low E: {lines[5]}"

    _test("ASCII tab renderer produces 6 lines", test_tab_renderer)

    # ── 10. API routes ─────────────────────────────────────────────────
    def test_api_routes():
        from app.main import app
        routes = {r.path for r in app.routes if hasattr(r, 'path')}
        expected = {
            "/health", "/models",
            "/transcribe", "/transcribe_video", "/transcribe_url",
            "/status/{job_id}", "/result/{job_id}",
        }
        missing = expected - routes
        assert not missing, f"Missing routes: {missing}"

    _test("API app has all 7 expected routes", test_api_routes)

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"🎸 ALL {total} TESTS PASSED — P13 smoke test complete!")
    else:
        print(f"⚠️  {PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
