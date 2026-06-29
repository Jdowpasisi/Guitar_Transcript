# ── P7: GuitarAI API Dockerfile ─────────────────────────────────────────────
# Single image for both the API server and the Celery worker.
# Context is the GuitarAI project root (one level above app/).

FROM python:3.12-slim

# System deps: ffmpeg needed by librosa / Demucs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first for layer caching
COPY requirements_api.txt .
# --timeout 300 + --retries 5 for slow/flaky networks
# Install torch CPU-only first (saves ~500MB vs full CUDA wheel)
RUN pip install --no-cache-dir --timeout 300 --retries 5 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch' requirements_api.txt > /tmp/reqs_no_torch.txt \
    && pip install --no-cache-dir --timeout 300 --retries 5 \
        -r /tmp/reqs_no_torch.txt \
    && pip install --no-cache-dir --timeout 300 --retries 5 \
        --no-deps basic-pitch==0.4.0

# Application code
COPY . .

EXPOSE 8000

# Default: API server. Worker overrides this in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
