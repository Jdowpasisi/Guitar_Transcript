// src/App.js
import React from 'react';
import { useTranscription, AppState } from './hooks/useTranscription';
import DropZone from './components/DropZone';
import ProcessingView from './components/ProcessingView';
import ResultView from './components/ResultView';
import './App.css';

export default function App() {
  const { appState, progress, result, error, fileName, submit, reset } = useTranscription();

  const isIdle       = appState === AppState.IDLE;
  const isProcessing = [AppState.UPLOADING, AppState.PENDING, AppState.PROCESSING].includes(appState);
  const isDone       = appState === AppState.DONE;
  const isError      = appState === AppState.ERROR;

  return (
    <div className="app">
      <header className="app-header">
        <a href="/" className="app-logo" aria-label="GuitarAI home">
          <span className="app-logo__icon" aria-hidden="true">🎸</span>
          <span className="app-logo__word">Guitar</span>
          <span className="app-logo__word app-logo__word--accent">AI</span>
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
            <div className="hero__eyebrow">Audio → Tablature</div>
            <h1 className="hero__title">
              Drop in a recording.<br />
              <span className="hero__title--accent">Get the tab.</span>
            </h1>
            <p className="hero__sub">
              GuitarAI runs your audio through a stem separator, chord classifier,
              note transcriber, and a trained Bi-LSTM that assigns each note to a
              specific string and fret.
            </p>
          </div>
        )}

        {/* Content card */}
        <div className="app-card">
          {isIdle && (
            <DropZone onFile={submit} />
          )}

          {isProcessing && (
            <ProcessingView
              appState={appState}
              progress={progress}
              fileName={fileName}
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

        {/* Idle footer — model badges */}
        {isIdle && (
          <div className="model-badges" aria-label="Models used in the pipeline">
            {[
              { label: 'Demucs', desc: 'Stem separation' },
              { label: 'Basic Pitch', desc: 'Note detection' },
              { label: 'ChordCNN', desc: 'Chord classification' },
              { label: 'VoicingLSTM', desc: 'String/fret assignment' },
            ].map(m => (
              <div key={m.label} className="model-badge">
                <span className="model-badge__label">{m.label}</span>
                <span className="model-badge__desc">{m.desc}</span>
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="app-footer">
        GuitarAI — AI/ML Capstone · P8 Upload App
      </footer>
    </div>
  );
}
