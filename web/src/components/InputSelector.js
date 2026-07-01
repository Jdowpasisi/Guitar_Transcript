// src/components/InputSelector.js
// GuitarAI v1 — three-mode input selector:
//   🎵 Audio  |  🎬 Video  |  ▶️ YouTube URL

import React, { useState, useRef } from 'react';
import './InputSelector.css';

const AUDIO_EXTS  = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'];
const VIDEO_EXTS  = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'];
const MAX_AUDIO_MB = 100;
const MAX_VIDEO_MB = 500;

function fileSizeMB(file) {
  return file.size / (1024 * 1024);
}

export default function InputSelector({ onAudio, onVideo, onUrl }) {
  const [mode, setMode]         = useState('audio'); // 'audio' | 'video' | 'youtube'
  const [urlValue, setUrlValue] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [urlError, setUrlError]  = useState('');
  const audioInputRef = useRef();
  const videoInputRef = useRef();

  // ── Drag & drop handlers ────────────────────────────────────────────────
  const handleDragEnter = (e) => { e.preventDefault(); setDragActive(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setDragActive(false); };
  const handleDragOver  = (e) => { e.preventDefault(); };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) _handleFile(file);
  };

  // ── File validation + dispatch ──────────────────────────────────────────
  const _handleFile = (file) => {
    const name = file.name.toLowerCase();
    const isAudio = AUDIO_EXTS.some(ext => name.endsWith(ext));
    const isVideo = VIDEO_EXTS.some(ext => name.endsWith(ext));

    if (mode === 'audio') {
      if (!isAudio) {
        alert(`Please drop an audio file.\nSupported: ${AUDIO_EXTS.join(', ')}`);
        return;
      }
      if (fileSizeMB(file) > MAX_AUDIO_MB) {
        alert(`File too large (${fileSizeMB(file).toFixed(1)} MB). Max: ${MAX_AUDIO_MB} MB`);
        return;
      }
      onAudio(file);
    } else if (mode === 'video') {
      if (!isAudio && !isVideo) {
        alert(`Please drop a video file.\nSupported: ${VIDEO_EXTS.join(', ')}`);
        return;
      }
      if (fileSizeMB(file) > MAX_VIDEO_MB) {
        alert(`File too large (${fileSizeMB(file).toFixed(1)} MB). Max: ${MAX_VIDEO_MB} MB`);
        return;
      }
      onVideo(file);
    }
  };

  // ── YouTube URL submit ──────────────────────────────────────────────────
  const handleUrlSubmit = (e) => {
    e.preventDefault();
    const url = urlValue.trim();
    setUrlError('');
    if (!url) { setUrlError('Please enter a YouTube URL'); return; }
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      setUrlError('URL must start with http:// or https://');
      return;
    }
    onUrl(url);
  };

  const currentExts = mode === 'audio' ? AUDIO_EXTS : VIDEO_EXTS;
  const currentMaxMB = mode === 'audio' ? MAX_AUDIO_MB : MAX_VIDEO_MB;

  return (
    <div className="input-selector">
      {/* ── Mode tabs ─────────────────────────────────────────────────── */}
      <div className="input-tabs" role="tablist" aria-label="Input mode">
        {[
          { id: 'audio',   icon: '🎵', label: 'Audio',   desc: 'MP3, WAV, FLAC…' },
          { id: 'video',   icon: '🎬', label: 'Video',   desc: 'MP4, MOV, AVI…' },
          { id: 'youtube', icon: '▶️', label: 'YouTube', desc: 'Paste a URL' },
        ].map(tab => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            role="tab"
            aria-selected={mode === tab.id}
            aria-controls={`panel-${tab.id}`}
            className={`input-tab ${mode === tab.id ? 'input-tab--active' : ''}`}
            onClick={() => { setMode(tab.id); setUrlError(''); }}
          >
            <span className="input-tab__icon">{tab.icon}</span>
            <span className="input-tab__label">{tab.label}</span>
            <span className="input-tab__desc">{tab.desc}</span>
          </button>
        ))}
      </div>

      {/* ── File drop zone (audio or video mode) ──────────────────────── */}
      {(mode === 'audio' || mode === 'video') && (
        <div
          id={`panel-${mode}`}
          role="tabpanel"
          aria-labelledby={`tab-${mode}`}
          className={`drop-zone ${dragActive ? 'drop-zone--active' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleFileDrop}
          onClick={() => mode === 'audio' ? audioInputRef.current.click() : videoInputRef.current.click()}
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && (mode === 'audio' ? audioInputRef.current.click() : videoInputRef.current.click())}
          aria-label={`Drop ${mode} file here or click to browse`}
        >
          <div className="drop-zone__icon" aria-hidden="true">
            {mode === 'audio' ? '🎸' : '🎬'}
          </div>
          <div className="drop-zone__text">
            <span className="drop-zone__primary">
              Drop your {mode} file here
            </span>
            <span className="drop-zone__secondary">
              or <span className="drop-zone__link">click to browse</span>
            </span>
          </div>
          <div className="drop-zone__formats">
            {currentExts.join(' · ')} · max {currentMaxMB} MB
          </div>
          {mode === 'video' && (
            <div className="drop-zone__badge drop-zone__badge--fusion">
              ✨ Enables FusionModel (P12) for multimodal transcription
            </div>
          )}
          {/* Hidden file inputs */}
          <input
            ref={audioInputRef}
            type="file"
            accept={AUDIO_EXTS.join(',')}
            style={{ display: 'none' }}
            onChange={e => e.target.files[0] && _handleFile(e.target.files[0])}
            aria-hidden="true"
          />
          <input
            ref={videoInputRef}
            type="file"
            accept={VIDEO_EXTS.join(',')}
            style={{ display: 'none' }}
            onChange={e => e.target.files[0] && _handleFile(e.target.files[0])}
            aria-hidden="true"
          />
        </div>
      )}

      {/* ── YouTube URL panel ──────────────────────────────────────────── */}
      {mode === 'youtube' && (
        <div
          id="panel-youtube"
          role="tabpanel"
          aria-labelledby="tab-youtube"
          className="yt-panel"
        >
          <div className="yt-panel__icon" aria-hidden="true">▶️</div>
          <p className="yt-panel__intro">
            Paste a YouTube (or any yt-dlp compatible) URL.<br />
            We'll download the video and run the full multimodal pipeline.
          </p>
          <form className="yt-panel__form" onSubmit={handleUrlSubmit}>
            <div className="yt-panel__input-wrap">
              <input
                id="yt-url-input"
                type="url"
                className={`yt-panel__input ${urlError ? 'yt-panel__input--error' : ''}`}
                placeholder="https://www.youtube.com/watch?v=..."
                value={urlValue}
                onChange={e => { setUrlValue(e.target.value); setUrlError(''); }}
                autoComplete="off"
                spellCheck={false}
              />
              {urlError && (
                <span className="yt-panel__error" role="alert">{urlError}</span>
              )}
            </div>
            <button
              type="submit"
              className="yt-panel__submit"
              disabled={!urlValue.trim()}
              id="yt-transcribe-btn"
            >
              <span>Transcribe</span>
              <span className="yt-panel__submit-icon" aria-hidden="true">→</span>
            </button>
          </form>
          <div className="yt-panel__badge">
            ✨ Downloads via yt-dlp · FusionModel (P12) used when video available
          </div>
          <p className="yt-panel__note">
            ⏱ Download + processing may take 2–5 minutes depending on video length.
          </p>
        </div>
      )}
    </div>
  );
}
