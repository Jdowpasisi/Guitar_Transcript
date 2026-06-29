// src/components/ResultView.js
import React from 'react';
import ChordTimeline from './ChordTimeline';
import TabDisplay from './TabDisplay';
import PipelineInfo from './PipelineInfo';
import './ResultView.css';

export default function ResultView({ result, fileName, onReset }) {
  if (!result) return null;

  return (
    <div className="result-view">
      <div className="result-view__topbar">
        <div className="result-view__file">
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 2H9L13 6V14H3V2Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
            <path d="M9 2V6H13" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
          </svg>
          <span>{fileName}</span>
        </div>
        <button className="result-view__reset-btn" onClick={onReset}>
          Transcribe another file
        </button>
      </div>

      <div className="result-view__sections">
        <ChordTimeline chords={result.chords} />
        <TabDisplay tab={result.tab} notes={result.notes} fileName={fileName} />
        <PipelineInfo pipeline={result.pipeline} />
      </div>
    </div>
  );
}
