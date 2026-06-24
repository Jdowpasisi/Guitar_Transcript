"""
GuitarAI — Audio Explorer (P1)
===============================
Load any audio file. Visualise it. Understand what the models will see.

Usage:
    python -m src.audio.explorer path/to/audio.wav
    python -m src.audio.explorer path/to/audio.wav --output-dir outputs/my_run

Programmatic:
    from src.audio.explorer import process_audio
    result = process_audio("song.wav")
    print(result["sr"], result["duration"])
    mel_db = result["mel_db"]  # reuse without recomputing
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
import os
from pathlib import Path

# Try to import shared config; fall back to defaults if run standalone
try:
    from src.config import OUTPUTS_DIR, SR_RAW
except ImportError:
    OUTPUTS_DIR = Path("outputs")
    SR_RAW = 44100


def process_audio(file_path, output_dir=None, sr=None, save_plots=True,
                  save_stems=True):
    """
    Analyse an audio file: compute stats, HPSS separation, and spectrograms.

    Parameters
    ----------
    file_path : str or Path
        Path to the audio file (WAV, MP3, FLAC, etc.)
    output_dir : str or Path, optional
        Directory for output files. Defaults to outputs/explorer/.
        A subdirectory named after the input file is created automatically.
    sr : int, optional
        Force a specific sample rate. None = native sample rate.
    save_plots : bool
        Whether to save visualization PNGs.
    save_stems : bool
        Whether to save harmonic/percussive WAV stems.

    Returns
    -------
    dict
        Keys: sr, channels, duration, max_amp, min_amp, mean_amp,
              harmonic, percussive, stft_db, mel_db, analysis_audio,
              output_dir (Path to where files were saved)
    """
    file_path = Path(file_path)

    # Resolve output directory
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "explorer"
    output_dir = Path(output_dir) / file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Audio (mono=False to check for channels)
    # Note: librosa usually converts to mono by default unless specified
    audio, native_sr = librosa.load(str(file_path), sr=sr, mono=False)

    # Handle channel count
    channels = 1 if len(audio.shape) == 1 else audio.shape[0]
    duration = librosa.get_duration(y=audio, sr=native_sr)

    # For stats, if stereo, we'll just look at the first channel
    analysis_audio = audio if channels == 1 else audio[0]

    print(f"\n--- 📊 Audio Stats: {file_path.name} ---")
    print(f"Sample Rate: {native_sr} Hz")
    print(f"Channels:    {channels}")
    print(f"Duration:    {duration:.2f} seconds")
    print(f"Max Amp:     {np.max(analysis_audio):.4f}")
    print(f"Min Amp:     {np.min(analysis_audio):.4f}")
    print(f"Mean Amp:    {np.mean(analysis_audio):.4f}")

    # 2. HPSS (Harmonic-Percussive Source Separation)
    # This separates the 'notes' from the 'percussion/pick noise'
    print("Applying HPSS separation...")
    harmonic, percussive = librosa.effects.hpss(analysis_audio)

    if save_stems:
        harmonic_path = output_dir / "harmonic_stem.wav"
        percussive_path = output_dir / "percussive_stem.wav"
        sf.write(str(harmonic_path), harmonic, native_sr)
        sf.write(str(percussive_path), percussive, native_sr)
        print(f"Saved: {harmonic_path}")
        print(f"Saved: {percussive_path}")

    # 3. Compute spectrograms (always — even if we don't save plots)
    stft = np.abs(librosa.stft(analysis_audio))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)

    mel = librosa.feature.melspectrogram(y=analysis_audio, sr=native_sr,
                                         n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 4. Visualizations
    if save_plots:
        plt.figure(figsize=(15, 10))

        # --- Plot 1: Waveform ---
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(analysis_audio, sr=native_sr)
        plt.title("Waveform")

        # --- Plot 2: Linear Spectrogram (STFT) ---
        plt.subplot(3, 1, 2)
        librosa.display.specshow(stft_db, sr=native_sr, x_axis='time',
                                 y_axis='log')
        plt.title("Standard Spectrogram (Log Frequency)")
        plt.colorbar(format='%+2.0f dB')

        # --- Plot 3: Mel Spectrogram ---
        plt.subplot(3, 1, 3)
        librosa.display.specshow(mel_db, sr=native_sr, x_axis='time',
                                 y_axis='mel')
        plt.title("Mel Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plot_path = output_dir / "audio_analysis.png"
        plt.savefig(str(plot_path))
        plt.close()
        print(f"Saved: {plot_path}")

    # 5. Return everything for downstream reuse
    return {
        "sr":             native_sr,
        "channels":       channels,
        "duration":       duration,
        "max_amp":        float(np.max(analysis_audio)),
        "min_amp":        float(np.min(analysis_audio)),
        "mean_amp":       float(np.mean(analysis_audio)),
        "analysis_audio": analysis_audio,
        "harmonic":       harmonic,
        "percussive":     percussive,
        "stft_db":        stft_db,
        "mel_db":         mel_db,
        "output_dir":     output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuitarAI Audio Explorer")
    parser.add_argument("path", help="Path to the audio file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/explorer/)")
    args = parser.parse_args()

    if os.path.exists(args.path):
        process_audio(args.path, output_dir=args.output_dir)
    else:
        print(f"Error: File {args.path} not found.")