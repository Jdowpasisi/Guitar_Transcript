// src/components/DropZone.js
import React, { useState, useRef, useCallback } from 'react';
import './DropZone.css';

const ALLOWED = new Set(['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']);

function getExtension(filename) {
  return filename.slice(filename.lastIndexOf('.')).toLowerCase();
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DropZone({ onFile }) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError]   = useState(null);
  const inputRef = useRef(null);

  const validate = useCallback((file) => {
    const ext = getExtension(file.name);
    if (!ALLOWED.has(ext)) {
      return `${ext} files aren't supported. Use MP3, WAV, FLAC, OGG, M4A, or AAC.`;
    }
    if (file.size > 100 * 1024 * 1024) {
      return `${file.name} is ${formatSize(file.size)} — maximum is 100 MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback((file) => {
    setFileError(null);
    const err = validate(file);
    if (err) { setFileError(err); return; }
    onFile(file);
  }, [validate, onFile]);

  // Drag handlers
  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = (e) => {
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const onInputChange = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  return (
    <div
      className={`dropzone ${isDragging ? 'dropzone--dragging' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Drop audio file or click to browse"
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.ogg,.m4a,.aac,audio/*"
        onChange={onInputChange}
        aria-hidden="true"
        tabIndex={-1}
      />

      <div className="dropzone__icon" aria-hidden="true">
        {isDragging ? (
          <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M24 8L24 32M24 8L16 16M24 8L32 16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M8 36H40V40H8V36Z" fill="currentColor" opacity="0.3"/>
            <path d="M8 36H40" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="6" y="14" width="36" height="26" rx="3" stroke="currentColor" strokeWidth="2" opacity="0.4"/>
            <path d="M16 14V11C16 9.34 17.34 8 19 8H29C30.66 8 32 9.34 32 11V14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="30" cy="25" r="5" stroke="currentColor" strokeWidth="2"/>
            <path d="M30 20V25L33 28" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M13 22H20M13 27H18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.5"/>
          </svg>
        )}
      </div>

      <div className="dropzone__text">
        {isDragging ? (
          <span className="dropzone__text--drop">Release to transcribe</span>
        ) : (
          <>
            <span className="dropzone__text--primary">Drop a guitar recording here</span>
            <span className="dropzone__text--secondary">or click to browse — MP3, WAV, FLAC up to 100 MB</span>
          </>
        )}
      </div>

      {fileError && (
        <div className="dropzone__error" role="alert" onClick={(e) => e.stopPropagation()}>
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M8 5V8.5M8 11H8.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          {fileError}
        </div>
      )}
    </div>
  );
}
