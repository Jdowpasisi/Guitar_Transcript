// src/hooks/useTranscription.js
// State machine for the transcription job lifecycle.
// Supports three input modes: audio file, video file, YouTube URL
//
//   IDLE → UPLOADING → PENDING → PROCESSING → DONE | ERROR

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  submitTranscription,
  submitVideoTranscription,
  submitYouTubeUrl,
  getJobStatus,
  getJobResult,
} from '../utils/api';

export const AppState = {
  IDLE:       'IDLE',
  UPLOADING:  'UPLOADING',
  PENDING:    'PENDING',
  PROCESSING: 'PROCESSING',
  DONE:       'DONE',
  ERROR:      'ERROR',
};

const POLL_INTERVAL_MS = 2000;

export function useTranscription() {
  const [appState, setAppState]   = useState(AppState.IDLE);
  const [jobId, setJobId]         = useState(null);
  const [progress, setProgress]   = useState({ step: '', percent: 0 });
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [fileName, setFileName]   = useState(null);
  const [hasVideo, setHasVideo]   = useState(false);

  const pollRef = useRef(null);

  // Clear polling on unmount
  useEffect(() => () => clearInterval(pollRef.current), []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback((id) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(id);

        if (status.status === 'STARTED') {
          setAppState(AppState.PROCESSING);
          setProgress({
            step:    status.meta?.step    || 'Processing…',
            percent: status.meta?.percent || 0,
          });
        } else if (status.status === 'PENDING') {
          setAppState(AppState.PENDING);
          setProgress({ step: 'Waiting in queue…', percent: 0 });
        } else if (status.status === 'SUCCESS') {
          stopPolling();
          try {
            const res = await getJobResult(id);
            setResult(res);
            setAppState(AppState.DONE);
          } catch (e) {
            setError(e.message || 'Failed to fetch result');
            setAppState(AppState.ERROR);
          }
        } else if (status.status === 'FAILURE') {
          stopPolling();
          setError(status.meta?.error || 'The transcription job failed on the server.');
          setAppState(AppState.ERROR);
        }
      } catch (e) {
        stopPolling();
        setError(e.message || 'Lost connection to server.');
        setAppState(AppState.ERROR);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling]);

  /** Submit an audio file (audio-only pipeline) */
  const submit = useCallback(async (file) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setHasVideo(false);
    setAppState(AppState.UPLOADING);
    setProgress({ step: 'Uploading audio…', percent: 0 });

    try {
      const res = await submitTranscription(file);
      setJobId(res.job_id);
      setAppState(AppState.PENDING);
      setProgress({ step: 'Waiting in queue…', percent: 0 });
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message || 'Upload failed.');
      setAppState(AppState.ERROR);
    }
  }, [startPolling]);

  /** Submit a video file (multimodal fusion pipeline) */
  const submitVideo = useCallback(async (file) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setHasVideo(true);
    setAppState(AppState.UPLOADING);
    setProgress({ step: 'Uploading video…', percent: 0 });

    try {
      const res = await submitVideoTranscription(file);
      setJobId(res.job_id);
      setAppState(AppState.PENDING);
      setProgress({ step: 'Waiting in queue…', percent: 0 });
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message || 'Video upload failed.');
      setAppState(AppState.ERROR);
    }
  }, [startPolling]);

  /** Submit a YouTube URL (yt-dlp download + multimodal pipeline) */
  const submitUrl = useCallback(async (url) => {
    setError(null);
    setResult(null);
    setFileName(url);
    setHasVideo(true);
    setAppState(AppState.UPLOADING);
    setProgress({ step: 'Sending URL to server…', percent: 0 });

    try {
      const res = await submitYouTubeUrl(url);
      setJobId(res.job_id);
      setAppState(AppState.PENDING);
      setProgress({ step: 'Downloading video (yt-dlp)…', percent: 0 });
      startPolling(res.job_id);
    } catch (e) {
      setError(e.message || 'URL submission failed.');
      setAppState(AppState.ERROR);
    }
  }, [startPolling]);

  const reset = useCallback(() => {
    stopPolling();
    setAppState(AppState.IDLE);
    setJobId(null);
    setProgress({ step: '', percent: 0 });
    setResult(null);
    setError(null);
    setFileName(null);
    setHasVideo(false);
  }, [stopPolling]);

  return {
    appState, jobId, progress, result, error, fileName, hasVideo,
    submit, submitVideo, submitUrl, reset,
  };
}
