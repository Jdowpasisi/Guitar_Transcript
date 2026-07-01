// src/components/ProcessingView.js
// Shows pipeline progress for both audio-only and multimodal (video) pipelines.
import React from 'react';
import './ProcessingView.css';

const AUDIO_STEPS = [
  { label: 'Loading audio',               icon: '♪',  hint: 'librosa load + normalize' },
  { label: 'Separating guitar stem',      icon: '✂',  hint: 'Demucs htdemucs (P2)' },
  { label: 'Transcribing notes',          icon: '𝄞',  hint: 'Basic Pitch ONNX (P5)' },
  { label: 'Detecting chords',            icon: '⬡',  hint: 'ChordCNN (P4)' },
  { label: 'Assigning voicings',          icon: '⚡',  hint: 'VoicingLSTM (P6)' },
  { label: 'Rendering tablature',         icon: '≡',  hint: 'ASCII tab output' },
];

const VIDEO_STEPS = [
  { label: 'Loading audio + video',           icon: '♪',  hint: 'Preparing both pipelines' },
  { label: 'Separating guitar stem',          icon: '✂',  hint: 'Demucs htdemucs (P2)' },
  { label: 'Extracting video frames',         icon: '🎬', hint: 'FFmpeg 5fps (P9)' },
  { label: 'Transcribing notes (parallel)',   icon: '𝄞',  hint: 'Basic Pitch ONNX (P5)' },
  { label: 'Tracking finger positions',       icon: '👆', hint: 'MediaPipe HandLandmarker (P11)' },
  { label: 'Detecting chords',               icon: '⬡',  hint: 'ChordCNN (P4)' },
  { label: 'Fusing audio + video',            icon: '🔀', hint: 'Cross-Attention FusionModel (P12)' },
  { label: 'Rendering tablature',             icon: '≡',  hint: 'ASCII tab output' },
];

function stepFromPercent(percent, steps) {
  const n = steps.length;
  const idx = Math.floor((percent / 100) * n);
  return Math.min(idx, n - 1);
}

export default function ProcessingView({ appState, progress, fileName, hasVideo = false }) {
  const isUploading = appState === 'UPLOADING';
  const isPending   = appState === 'PENDING';
  const steps       = hasVideo ? VIDEO_STEPS : AUDIO_STEPS;
  const activeStep  = isUploading || isPending ? -1 : stepFromPercent(progress.percent, steps);
  const displayPct  = isUploading ? 0 : isPending ? 2 : progress.percent;

  const uploaderLabel = hasVideo
    ? 'Uploading video…'
    : 'Uploading audio…';

  const processingLabel = hasVideo
    ? 'Transcribing (audio + vision)…'
    : 'Transcribing…';

  return (
    <div className="processing">
      <div className="processing__header">
        <div className="processing__spinner" aria-hidden="true">
          <div className="processing__spinner-ring" />
          <span className="processing__spinner-icon">{hasVideo ? '🎬' : '🎸'}</span>
        </div>
        <div className="processing__title-group">
          <h2 className="processing__title">
            {isUploading ? uploaderLabel : isPending ? 'In queue…' : processingLabel}
          </h2>
          {fileName && (
            <p className="processing__filename">{fileName.length > 60 ? fileName.slice(0, 57) + '…' : fileName}</p>
          )}
          {hasVideo && !isUploading && !isPending && (
            <div className="processing__fusion-badge">
              🔀 FusionModel (P12) — audio + video
            </div>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="processing__bar-track"
        role="progressbar"
        aria-valuenow={displayPct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="processing__bar-fill"
          style={{ width: `${Math.max(displayPct, isUploading ? 8 : isPending ? 4 : 8)}%` }}
        />
      </div>

      {/* Step list */}
      {!isUploading && !isPending && (
        <ol className="processing__steps" aria-label="Pipeline progress">
          {steps.map((s, i) => {
            const done   = i < activeStep;
            const active = i === activeStep;
            return (
              <li
                key={s.label}
                className={`processing__step ${done ? 'done' : active ? 'active' : 'waiting'}`}
                aria-current={active ? 'step' : undefined}
              >
                <span className="processing__step-indicator" aria-hidden="true">
                  {done ? (
                    <svg viewBox="0 0 16 16" fill="none">
                      <circle cx="8" cy="8" r="7" fill="var(--green)" opacity="0.2"/>
                      <path d="M5 8L7 10L11 6" stroke="var(--green)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  ) : active ? (
                    <span className="processing__step-pulse" />
                  ) : (
                    <span className="processing__step-dot" />
                  )}
                </span>
                <div className="processing__step-text">
                  <span className="processing__step-label">{s.label}</span>
                  {active && s.hint && (
                    <span className="processing__step-hint">{s.hint}</span>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}

      {(isUploading || isPending) && (
        <p className="processing__status-text">
          {isUploading
            ? `Sending ${hasVideo ? 'video' : 'file'} to server…`
            : 'Waiting for a worker to pick up the job…'
          }
        </p>
      )}
    </div>
  );
}
