// src/App.js — GuitarAI v1
import React from 'react';
import { useTranscription, AppState } from './hooks/useTranscription';
import InputSelector from './components/InputSelector';
import ProcessingView from './components/ProcessingView';
import ResultView from './components/ResultView';
import './App.css';

// All 5 trained models in the pipeline
const PIPELINE_MODELS = [
  { label: 'Demucs',      desc: 'Stem separation',          icon: '✂️' },
  { label: 'Basic Pitch', desc: 'Note detection',           icon: '🎵' },
  { label: 'ChordCNN',    desc: 'Chord classification',     icon: '🧠' },
  { label: 'VoicingLSTM',desc: 'String/fret assignment',   icon: '⚡' },
  { label: 'FusionModel', desc: 'Multimodal fusion (video)',icon: '🔀' },
];

export default function App() {
  const {
    appState, progress, result, error, fileName, hasVideo,
    submit, submitVideo, submitUrl, reset,
  } = useTranscription();

  const isIdle       = appState === AppState.IDLE;
  const isProcessing = [AppState.UPLOADING, AppState.PENDING, AppState.PROCESSING].includes(appState);
  const isDone       = appState === AppState.DONE;
  const isError      = appState === AppState.ERROR;

  return (
    <div className="app">
      <header className="app-header">
        <a href="/" className="app-logo" aria-label="GuitarAI v1 home">
          <span className="app-logo__icon" aria-hidden="true">🎸</span>
          <span className="app-logo__word">Guitar</span>
          <span className="app-logo__word app-logo__word--accent">AI</span>
          <span className="app-logo__version">v1</span>
        </a>
        <nav className="app-nav">
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className="app-nav__link">
            API docs
          </a>
          <a href="http://localhost:5555" target="_blank" rel="noopener noreferrer" className="app-nav__link">
            Workers
          </a>
        </nav>
      </header>

      <main className="app-main">
        {/* Hero — only shown when idle */}
        {isIdle && (
          <div className="hero">
            <div className="hero__eyebrow">Audio · Video · YouTube → Tablature</div>
            <h1 className="hero__title">
              Drop in a recording.<br />
              <span className="hero__title--accent">Get the tab.</span>
            </h1>
            <p className="hero__sub">
              GuitarAI v1 runs your audio through 5 trained models: stem separator,
              chord classifier, note transcriber, Bi-LSTM voicing, and a multimodal
              cross-attention Fusion Model that fuses audio + video for maximum accuracy.
            </p>
          </div>
        )}

        {/* Content card */}
        <div className="app-card">
          {isIdle && (
            <InputSelector
              onAudio={submit}
              onVideo={submitVideo}
              onUrl={submitUrl}
            />
          )}

          {isProcessing && (
            <ProcessingView
              appState={appState}
              progress={progress}
              fileName={fileName}
              hasVideo={hasVideo}
            />
          )}

          {isError && (
            <div className="app-error" role="alert">
              <div className="app-error__icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5"/>
                  <path d="M12 7V12.5M12 16H12.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <div className="app-error__body">
                <h2 className="app-error__title">Transcription failed</h2>
                <p className="app-error__msg">{error}</p>
              </div>
              <button className="app-error__retry" onClick={reset}>
                Try again
              </button>
            </div>
          )}

          {isDone && (
            <ResultView
              result={result}
              fileName={fileName}
              onReset={reset}
            />
          )}
        </div>

        {/* Idle footer — all 5 model badges */}
        {isIdle && (
          <div className="model-badges" aria-label="Models used in the pipeline">
            {PIPELINE_MODELS.map(m => (
              <div key={m.label} className={`model-badge ${m.label === 'FusionModel' ? 'model-badge--fusion' : ''}`}>
                <span className="model-badge__icon" aria-hidden="true">{m.icon}</span>
                <span className="model-badge__label">{m.label}</span>
                <span className="model-badge__desc">{m.desc}</span>
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="app-footer">
        GuitarAI v1 — AI/ML Capstone · 5 Trained Models · Multimodal Fusion
      </footer>
    </div>
  );
}
