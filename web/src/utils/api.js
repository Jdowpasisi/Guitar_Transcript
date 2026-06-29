// src/utils/api.js
// All communication with the P7 FastAPI backend.
// The CRA proxy (package.json → "proxy": "http://localhost:8000") forwards
// /transcribe, /status, /result, etc. to the API during development.
// In production, set REACT_APP_API_URL to the deployed base URL.

const BASE_URL = process.env.REACT_APP_API_URL || '';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    // Try to pull a detail message from FastAPI's error body
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
 * POST /transcribe
 * @param {File} file
 * @returns {Promise<{job_id, status, message, filename, size_mb}>}
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
