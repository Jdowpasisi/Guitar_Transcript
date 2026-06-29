// src/components/ProcessingView.js
import React from 'react';
import './ProcessingView.css';

const STEPS = [
  { label: 'Loading audio',               icon: '♪' },
  { label: 'Separating guitar stem',      icon: '✂' },
  { label: 'Transcribing notes',          icon: '𝄞' },
  { label: 'Detecting chords',            icon: '⬡' },
  { label: 'Assigning voicings',          icon: '⬡' },
  { label: 'Rendering tablature',         icon: '≡' },
];

function stepFromPercent(percent) {
  if (percent <= 5)  return 0;
  if (percent <= 15) return 1;
  if (percent <= 35) return 2;
  if (percent <= 55) return 3;
  if (percent <= 75) return 4;
  return 5;
}

export default function ProcessingView({ appState, progress, fileName }) {
  const isUploading = appState === 'UPLOADING';
  const isPending   = appState === 'PENDING';
  const activeStep  = isUploading || isPending ? -1 : stepFromPercent(progress.percent);
  const displayPct  = isUploading ? 0 : isPending ? 2 : progress.percent;

  return (
    <div className="processing">
      <div className="processing__header">
        <div className="processing__spinner" aria-hidden="true">
          <div className="processing__spinner-ring" />
          <span className="processing__spinner-icon">🎸</span>
        </div>
        <div className="processing__title-group">
          <h2 className="processing__title">
            {isUploading ? 'Uploading…' : isPending ? 'In queue…' : 'Transcribing…'}
          </h2>
          {fileName && (
            <p className="processing__filename">{fileName}</p>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="processing__bar-track" role="progressbar" aria-valuenow={displayPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="processing__bar-fill" style={{ width: `${Math.max(displayPct, isUploading ? 8 : isPending ? 4 : 8)}%` }} />
      </div>

      {/* Step list */}
      {!isUploading && !isPending && (
        <ol className="processing__steps" aria-label="Pipeline progress">
          {STEPS.map((s, i) => {
            const done    = i < activeStep;
            const active  = i === activeStep;
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
                <span className="processing__step-label">{s.label}</span>
              </li>
            );
          })}
        </ol>
      )}

      {(isUploading || isPending) && (
        <p className="processing__status-text">
          {isUploading ? 'Sending file to server…' : 'Waiting for a worker to pick up the job…'}
        </p>
      )}
    </div>
  );
}
