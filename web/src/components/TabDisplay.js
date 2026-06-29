// src/components/TabDisplay.js
import React, { useCallback } from 'react';
import './TabDisplay.css';

const STRING_LABELS  = ['e', 'B', 'G', 'D', 'A', 'E'];
const STRING_PITCHES = ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'];

function sourceLabel(src) {
  switch (src) {
    case 'lstm':     return { text: 'LSTM',    cls: 'lstm' };
    case 'greedy':   return { text: 'Greedy',  cls: 'greedy' };
    case 'heuristic':return { text: 'Heuristic', cls: 'greedy' };
    default:         return { text: src || '?', cls: 'greedy' };
  }
}

export default function TabDisplay({ tab, notes, fileName }) {
  // Parse the raw ASCII tab string from the API
  const tabLines = (tab || '').split('\n');

  // Count source types for the info bar
  const sourceCounts = (notes || []).reduce((acc, n) => {
    const s = n.voicing_source || 'unknown';
    acc[s] = (acc[s] || 0) + 1;
    return acc;
  }, {});

  const handleExport = useCallback(() => {
    const header = `GuitarAI Transcription\nFile: ${fileName || 'audio'}\n${'─'.repeat(50)}\n\n`;
    const blob = new Blob([header + tab], { type: 'text/plain' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `${(fileName || 'transcription').replace(/\.[^.]+$/, '')}_tab.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [tab, fileName]);

  if (!tab || tabLines.length < 6) {
    return (
      <section className="tab-display tab-display--empty">
        <h3 className="tab-display__label">Guitar Tab</h3>
        <p className="tab-display__empty-msg">No tablature generated.</p>
      </section>
    );
  }

  return (
    <section className="tab-display">
      <div className="tab-display__header">
        <h3 className="tab-display__label">
          Guitar Tab
          {notes?.length > 0 && (
            <span className="tab-display__note-count">{notes.length} notes</span>
          )}
        </h3>
        <button className="tab-export-btn" onClick={handleExport} title="Download as .txt">
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 2V10M8 10L5 7M8 10L11 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M3 13H13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          Export .txt
        </button>
      </div>

      {/* Source info bar */}
      {notes?.length > 0 && (
        <div className="tab-display__source-bar" aria-label="Voicing source breakdown">
          {Object.entries(sourceCounts).map(([src, count]) => {
            const { text, cls } = sourceLabel(src);
            return (
              <span key={src} className={`source-badge source-badge--${cls}`}>
                {text}: {count}
              </span>
            );
          })}
          <span className="source-legend">
            voicing source per note
          </span>
        </div>
      )}

      {/* The actual tab — fretboard inlay dots above as the signature element */}
      <div className="tab-display__frame">
        <div className="tab-display__inlay" aria-hidden="true">
          {[3,5,7,9,12,15,17,19].map(fret => (
            <span key={fret} className="inlay-dot" style={{ left: `calc(${fret * 4.3}% + 28px)` }} />
          ))}
        </div>
        <div className="tab-scroll">
          <pre className="tab-display__pre" role="region" aria-label="Guitar tablature">
            {tabLines.map((line, i) => (
              <div key={i} className="tab-line">
                <span className="tab-string-label" aria-label={STRING_PITCHES[i] || ''}>
                  {line[0] || STRING_LABELS[i]}
                </span>
                <span className="tab-string-content">{line.slice(1)}</span>
              </div>
            ))}
          </pre>
        </div>
      </div>
    </section>
  );
}
