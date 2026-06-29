// src/components/ChordTimeline.js
import React, { useRef } from 'react';
import './ChordTimeline.css';

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function confidenceClass(conf) {
  if (conf >= 0.7) return 'high';
  if (conf >= 0.4) return 'medium';
  return 'low';
}

/** Split e.g. "G:maj" → { root: "G", quality: "maj" } */
function parseChord(label) {
  if (label === 'N') return { root: '—', quality: 'no chord' };
  const colon = label.indexOf(':');
  if (colon === -1) return { root: label, quality: '' };
  return { root: label.slice(0, colon), quality: label.slice(colon + 1) };
}

export default function ChordTimeline({ chords }) {
  const scrollRef = useRef(null);

  if (!chords || chords.length === 0) {
    return (
      <section className="chord-timeline chord-timeline--empty">
        <h3 className="chord-timeline__label">Chord Timeline</h3>
        <p className="chord-timeline__empty-msg">No chords detected in this recording.</p>
      </section>
    );
  }

  return (
    <section className="chord-timeline">
      <h3 className="chord-timeline__label">
        Chord Timeline
        <span className="chord-timeline__count">{chords.length} segments</span>
      </h3>

      <div className="chord-timeline__scroll" ref={scrollRef}>
        <div className="chord-timeline__track">
          {chords.map((chord, i) => {
            const { root, quality } = parseChord(chord.label);
            const conf = confidenceClass(chord.confidence);

            return (
              <div
                key={i}
                className={`chord-card chord-card--${conf}`}
                title={`${chord.label} — ${formatTime(chord.start)} → ${formatTime(chord.end)} (${Math.round(chord.confidence * 100)}% confidence)`}
              >
                <div className="chord-card__timestamp">{formatTime(chord.start)}</div>
                <div className="chord-card__name">
                  <span className="chord-card__root">{root}</span>
                  {quality && <span className="chord-card__quality">{quality}</span>}
                </div>
                <div className={`chord-card__badge chord-card__badge--${conf}`}>
                  {Math.round(chord.confidence * 100)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="chord-timeline__legend" aria-label="Confidence legend">
        <span className="legend-item legend-item--high">High ≥70%</span>
        <span className="legend-item legend-item--medium">Medium ≥40%</span>
        <span className="legend-item legend-item--low">Low &lt;40%</span>
      </div>
    </section>
  );
}
