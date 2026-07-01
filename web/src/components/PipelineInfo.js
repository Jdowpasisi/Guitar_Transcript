// src/components/PipelineInfo.js
import React, { useState } from 'react';
import './PipelineInfo.css';

export default function PipelineInfo({ pipeline }) {
  const [expanded, setExpanded] = useState(false);
  if (!pipeline) return null;

  const {
    models_used = [], audio_duration_sec, processing_time_sec,
    note_count, chord_count, stem_separation,
    has_video, fusion_used, video_source,
  } = pipeline;

  const speedup = audio_duration_sec > 0
    ? (audio_duration_sec / processing_time_sec).toFixed(1)
    : null;

  const summaryLabel = fusion_used
    ? `FusionModel (P12) + ${models_used.slice(0, 2).join(' + ')}`
    : `${models_used.slice(0, 2).join(' → ')}${models_used.length > 2 ? ` +${models_used.length - 2} more` : ''}`;

  return (
    <section className="pipeline-info">
      <button
        className="pipeline-info__toggle"
        onClick={() => setExpanded(v => !v)}
        aria-expanded={expanded}
        aria-controls="pipeline-detail"
      >
        <span className="pipeline-info__toggle-label">
          Pipeline: {summaryLabel}
        </span>
        <svg
          className={`pipeline-info__chevron ${expanded ? 'open' : ''}`}
          viewBox="0 0 16 16" fill="none" aria-hidden="true"
        >
          <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {expanded && (
        <div className="pipeline-info__detail" id="pipeline-detail">
          {/* Fusion / video badges */}
          {(has_video || fusion_used) && (
            <div className="pipeline-badges">
              {has_video && (
                <span className="pipeline-badge pipeline-badge--video">
                  🎬 Video {video_source === 'youtube' ? '(YouTube)' : '(Upload)'}
                </span>
              )}
              {fusion_used && (
                <span className="pipeline-badge pipeline-badge--fusion">
                  🔀 FusionModel — 83.8% Tab Accuracy
                </span>
              )}
              {!fusion_used && has_video && (
                <span className="pipeline-badge pipeline-badge--fallback">
                  ⚡ LSTM fallback (vision unavailable)
                </span>
              )}
            </div>
          )}

          <div className="pipeline-stats">
            <div className="pipeline-stat">
              <span className="pipeline-stat__value">{note_count}</span>
              <span className="pipeline-stat__label">notes</span>
            </div>
            <div className="pipeline-stat">
              <span className="pipeline-stat__value">{chord_count}</span>
              <span className="pipeline-stat__label">chords</span>
            </div>
            <div className="pipeline-stat">
              <span className="pipeline-stat__value">{audio_duration_sec?.toFixed(1)}s</span>
              <span className="pipeline-stat__label">audio</span>
            </div>
            <div className="pipeline-stat">
              <span className="pipeline-stat__value">{processing_time_sec?.toFixed(1)}s</span>
              <span className="pipeline-stat__label">processing</span>
            </div>
            {speedup && (
              <div className="pipeline-stat">
                <span className="pipeline-stat__value">{speedup}×</span>
                <span className="pipeline-stat__label">faster than realtime</span>
              </div>
            )}
          </div>

          <div className="pipeline-model-chain" aria-label="Model pipeline chain">
            {models_used.map((name, i) => (
              <React.Fragment key={name}>
                <span className={`pipeline-model ${name.includes('Fusion') ? 'pipeline-model--fusion' : ''}`}>
                  {name}
                </span>
                {i < models_used.length - 1 && (
                  <span className="pipeline-arrow" aria-hidden="true">→</span>
                )}
              </React.Fragment>
            ))}
          </div>

          {stem_separation && (
            <p className="pipeline-note">
              ✦ Demucs stem separation was applied — the guitar stem was extracted before transcription.
            </p>
          )}
        </div>
      )}
    </section>
  );

}
