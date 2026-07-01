// src/utils/api.js
// All communication with the GuitarAI v1 FastAPI backend.
// The CRA proxy (package.json → "proxy": "http://localhost:8000") forwards
// /transcribe, /status, /result, etc. to the API during development.
// In production, set REACT_APP_API_URL to the deployed base URL.

const BASE_URL = process.env.REACT_APP_API_URL || '';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/**
 * POST /transcribe  (audio-only)
 * @param {File} file
 * @returns {Promise<{job_id, status, message, filename, size_mb, has_video}>}
 */
export async function submitTranscription(file) {
  const formData = new FormData();
  formData.append('file', file);
  return apiFetch('/transcribe', {
    method: 'POST',
    body: formData,
  });
}

/**
 * POST /transcribe_video  (video upload → multimodal fusion pipeline)
 * @param {File} file
 * @returns {Promise<{job_id, status, message, filename, size_mb, has_video}>}
 */
export async function submitVideoTranscription(file) {
  const formData = new FormData();
  formData.append('file', file);
  return apiFetch('/transcribe_video', {
    method: 'POST',
    body: formData,
  });
}

/**
 * POST /transcribe_url  (YouTube URL → yt-dlp → multimodal pipeline)
 * @param {string} url
 * @returns {Promise<{job_id, status, message, filename, has_video}>}
 */
export async function submitYouTubeUrl(url) {
  return apiFetch('/transcribe_url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
}

/**
 * GET /status/:jobId
 * @returns {Promise<{job_id, status, meta}>}
 */
export async function getJobStatus(jobId) {
  return apiFetch(`/status/${jobId}`);
}

/**
 * GET /result/:jobId
 * @returns {Promise<TranscriptionResult>}
 */
export async function getJobResult(jobId) {
  return apiFetch(`/result/${jobId}`);
}

/**
 * GET /health
 */
export async function getHealth() {
  return apiFetch('/health');
}

/**
 * GET /models
 */
export async function getModels() {
  return apiFetch('/models');
}
