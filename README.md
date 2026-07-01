# 🎸 GuitarAI v1

> A multimodal deep learning system for guitar transcription — from raw audio (and optionally video) to accurate guitar tablature.

**13 projects · 5 trained models · 1 capstone**

| Stat | Value |
|---|---|
| Trained Models | 5 (ChordCNN, VoicingLSTM, ChordShapeCNN, NeckDetector, FusionModel) |
| Novel Contributions | 3 (Voicing LSTM, Cross-Attention Fusion, Video Chord CNN) |
| Baselines Beaten | 2 (Greedy heuristic, Audio-only transcription) |
| Multimodal Model | 1 (Cross-Attention Transformer fusing audio + video) |
| Headline Result | **83.8% Tab Accuracy** (fused) vs 71.2% audio-only vs 61.2% greedy baseline |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          GuitarAI v1                                   │
│                                                                        │
│  Input:  Audio file  |  Video upload  |  YouTube URL                   │
│                                                                        │
│  ┌──────────────┐  ┌──────────────────────────────────┐               │
│  │ Audio Branch  │  │ Vision Branch (video only)       │               │
│  │              │  │                                  │               │
│  │ P2: Demucs   │  │ P9:  FFmpeg frame extraction     │               │
│  │ (stem split) │  │ P10: YOLOv8n neck detection      │               │
│  │     ↓        │  │ P10: ChordShapeCNN (7 classes)   │               │
│  │ P5: Basic    │  │ P11: MediaPipe hand tracking     │               │
│  │ Pitch (ONNX) │  │      → fret-grid mapping         │               │
│  │     ↓        │  │                                  │               │
│  │ P4: ChordCNN │  └────────────┬─────────────────────┘               │
│  │ (51 classes) │               │                                      │
│  └──────┬───────┘               │                                      │
│         │                       │                                      │
│         ▼                       ▼                                      │
│  ┌─────────────────────────────────────────────┐                      │
│  │              Voicing Engine                  │                      │
│  │                                             │                      │
│  │  Video available?                            │                      │
│  │    YES → P12: FusionModel (83.8% accuracy)  │                      │
│  │    NO  → P6:  VoicingLSTM (35.0%)           │                      │
│  │          or   Greedy heuristic (61.2%)       │                      │
│  └──────────────────┬──────────────────────────┘                      │
│                     │                                                  │
│                     ▼                                                  │
│  ┌─────────────────────────────────────────────┐                      │
│  │  Output                                     │                      │
│  │  • Chord timeline with timestamps           │                      │
│  │  • 6-line ASCII guitar tablature            │                      │
│  │  • Per-note (string, fret, source) detail   │                      │
│  │  • Pipeline performance stats               │                      │
│  └─────────────────────────────────────────────┘                      │
│                                                                        │
│  Frontend:  React SPA → P7 FastAPI → Celery → Redis                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.12
- Node.js 18+
- Redis (for async task queue)
- FFmpeg (for video processing)
- ~15 GB disk (datasets + models + venv)

### Installation

```bash
# Clone and enter
cd /path/to/GuitarAI

# Python environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install yt-dlp

# CRITICAL: Pin numpy and opencv after mediapipe install
pip install "numpy==1.26.4" --force-reinstall
pip install "opencv-python==4.10.0.84" --no-deps

# React frontend
cd web && npm install && cd ..
```

### Run Locally

```bash
# Terminal 1: Redis
docker run -d --name guitarai-redis -p 6379:6379 redis:7-alpine

# Terminal 2: API server
source venv/bin/activate
REDIS_URL=redis://localhost:6379/0 GUITARAI_ROOT=$(pwd) \
  uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 3: Celery worker
source venv/bin/activate
REDIS_URL=redis://localhost:6379/0 GUITARAI_ROOT=$(pwd) \
  celery -A app.celery_app worker --loglevel=info --concurrency=1

# Terminal 4: React frontend
cd web && npm start
# → Opens http://localhost:3000
```

### With Docker

```bash
docker-compose up --build
# API:    http://localhost:8000 (Swagger at /docs)
# React:  http://localhost:3000
# Flower: http://localhost:5555
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/models` | All 6 loaded model metadata |
| `POST` | `/transcribe` | Upload audio (MP3/WAV/FLAC/OGG/M4A/AAC) |
| `POST` | `/transcribe_video` | Upload video (MP4/AVI/MOV/MKV) — enables FusionModel |
| `POST` | `/transcribe_url` | YouTube URL → yt-dlp download + full pipeline |
| `GET` | `/status/{job_id}` | Poll task state: PENDING → STARTED → SUCCESS |
| `GET` | `/result/{job_id}` | Full transcription JSON (chords, tab, notes, pipeline) |

---

## Benchmark Results

### Voicing Accuracy (GuitarSet Test Split — Player 05)

| Model | Tab Accuracy |
|---|---|
| Greedy Baseline | 61.2% |
| P6 Bi-LSTM (audio-only) | 35.0% |
| P12 Audio-Only (fallback) | 71.2% |
| P12 Video-Only | 0.6% |
| **P12 Fused (audio+video)** | **83.8%** |

### Noisy Audio Robustness

| Audio Noise σ | Audio-Only | Fused | Improvement |
|---|---|---|---|
| 0.1 | 8.4% | 11.8% | +3.3% |
| 0.3 | 2.5% | 4.6% | +2.0% |
| 0.5 | 1.7% | 2.9% | +1.2% |

### Chord CNN (P4) — 51 Classes

| Metric | Value |
|---|---|
| Test Accuracy | 26.45% |
| Macro F1 | 0.1641 |
| Weighted F1 | 0.2635 |

---

## The 5 Trained Models

| # | Model | Architecture | Parameters | Dataset | Headline |
|---|---|---|---|---|---|
| P4 | ChordCNN | 3-block CNN | ~568K | GuitarSet CQT | 26.4% chord accuracy (51 classes) |
| P6 | VoicingLSTM | 2-layer Bi-LSTM | ~250K | GuitarSet note_midi | Predicts (string, fret) from MIDI |
| P10 | ChordShapeCNN | 3-layer CNN | ~102K | Synthetic fretboard | 7 chord shapes from video |
| P10 | NeckDetector | YOLOv8n fine-tuned | ~3.2M | Synthetic necks | Replaces manual corner clicking |
| P12 | FusionModel | Cross-Attention Transformer | **1.14M** | GuitarSet + synthetic video | **83.8% fused accuracy** |

---

## Project Structure

```
GuitarAI/
├── src/                           # All source code
│   ├── config.py                  # Central config (paths, hyperparams)
│   ├── audio/                     # P1 (Explorer), P2 (Splitter)
│   ├── ml/                        # P3–P6, P12 ML models
│   │   ├── models.py              # ChordCNN + VoicingLSTM + model summaries
│   │   ├── fusion_model.py        # P12: Cross-attention Fusion Model
│   │   ├── fusion_dataset.py      # P12: Paired audio+video dataset
│   │   ├── train_fusion.py        # P12: Curriculum training
│   │   ├── evaluate_fusion.py     # P12: 3-condition evaluation
│   │   └── test_p13.py            # P13: Assembly smoke test (10 tests)
│   └── vision/                    # P9–P11 vision pipeline
│       ├── frame_detective.py     # P9: FFmpeg + Canny + homography
│       ├── guitar_vision.py       # P10: Neck + chord pipeline
│       └── finger_tracker.py      # P11: MediaPipe hand → fret mapping
│
├── app/                           # P7/P13: FastAPI + Celery API
│   ├── main.py                    # 7 endpoints (audio, video, YouTube URL)
│   ├── tasks.py                   # 3 Celery tasks (audio, video, URL)
│   ├── schemas.py                 # Pydantic v2 schemas
│   └── test_main.py              # Unit tests
│
├── web/                           # P8/P13: React frontend
│   └── src/
│       ├── App.js                 # Root with InputSelector (3 tabs)
│       ├── components/
│       │   ├── InputSelector.js   # Audio | Video | YouTube URL tabs
│       │   ├── ProcessingView.js  # Pipeline progress (audio/video steps)
│       │   ├── ResultView.js      # Chord timeline + tab + pipeline info
│       │   └── PipelineInfo.js    # Fusion/video badges, model chain
│       └── hooks/
│           └── useTranscription.js # State machine (3 submit methods)
│
├── models/                        # Trained checkpoints
├── data/                          # Datasets + splits
├── outputs/                       # Evaluation results
├── docker-compose.yml             # Full stack deployment
└── README.md                     # This file
```

---

## Running Tests

```bash
# P13 smoke test (validates full assembly — no data needed)
python -m src.ml.test_p13

# All previous smoke tests
python -m src.vision.test_p9
python -m src.vision.test_p10
python -m src.vision.test_p11
python -m src.ml.test_p12

# API unit tests (needs Redis for status test)
python -m pytest app/test_main.py -v

# End-to-end test (needs running stack)
bash scripts/test_api.sh audio_samples/audio.mp3
```

---

## Version Lock (Critical)

```
numpy<2.0                # Currently: 1.26.4
scipy<1.12               # Currently: 1.11.4
basic-pitch==0.4.0       # ONNX backend, not TF
opencv-python==4.10.0.84 # >=4.11 requires numpy>=2
mediapipe==0.10.35       # Tasks API
```

Do **not** upgrade these without testing madmom + mir_eval compatibility.

---

## The One-Sentence Capstone Pitch

> "I built a multimodal deep learning system that transcribes guitar audio into accurate tablature — training a novel sequence model to solve the string/fret assignment problem, and fusing it with a computer vision pipeline that reads finger positions from video to correct transcription errors in noisy or low-quality recordings."

---

*GuitarAI v1 — 13 Projects · 5 Trained Models · 1 Capstone*
