"""
GuitarAI — Stem Splitter (P2)
==============================
Separate a noisy mix into clean stems using Demucs.
The guitar stem feeds directly into P3, P4, P5, and P6.

Usage:
    python -m src.audio.splitter path/to/mixed_audio.wav
    python -m src.audio.splitter path/to/mixed_audio.wav --output-dir outputs/my_run

Programmatic:
    from src.audio.splitter import run_splitter
    result = run_splitter("song.wav")
    guitar_audio = result["guitar_mono"]
    print(result["energy_ratio"])

Batch:
    from src.audio.splitter import batch_separate
    batch_separate(["song1.wav", "song2.wav"], output_dir="data/processed/stems")
"""

import io
import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from demucs.api import Separator

# Try to import shared config; fall back to defaults if run standalone
try:
    from src.config import OUTPUTS_DIR
except ImportError:
    OUTPUTS_DIR = Path("outputs")


def calculate_energy(audio_array):
    """Calculates the Root Mean Square (RMS) energy of an audio signal."""
    return np.sqrt(np.mean(audio_array**2))


def run_splitter(input_path, output_dir=None, model="htdemucs",
                 save_plot=True):
    """
    Run Demucs source separation on an audio file and extract the guitar stem.

    Parameters
    ----------
    input_path : str or Path
        Path to the mixed audio file.
    output_dir : str or Path, optional
        Directory for output files. Defaults to outputs/splitter/.
        A subdirectory named after the input file is created automatically.
    model : str
        Demucs model name. 'htdemucs' is the standard 4-stem model.
        Use 'htdemucs_ft' for higher quality at the cost of speed.
    save_plot : bool
        Whether to save the comparison spectrogram.

    Returns
    -------
    dict
        Keys: guitar_mono (np.ndarray), guitar_path (Path),
              samplerate (int), energy_ratio (float), output_dir (Path)
    """
    input_path = Path(input_path)

    # Resolve output directory
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "splitter"
    output_dir = Path(output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize the Separator
    # 'htdemucs' is the standard 4-stem model (drums, bass, vocals, other)
    print(f"--- 🚀 Initializing Demucs ({model}) ---")
    separator = Separator(model=model)

    # 2. Load and Separate
    print(f"Separating: {input_path.name}...")
    # This returns a dictionary of {stem_name: torch.Tensor}
    origin, separated = separator.separate_audio_file(str(input_path))

    # In 'htdemucs', the guitar usually lives in the 'other' stem
    guitar_tensor = separated['other']

    # Convert torch tensor back to numpy for processing
    # Demucs output is usually (Channels, Samples)
    guitar_np = guitar_tensor.cpu().numpy()

    # If stereo, mix to mono for our later models
    if guitar_np.shape[0] > 1:
        guitar_mono = librosa.to_mono(guitar_np)
    else:
        guitar_mono = guitar_np.squeeze()

    # 3. Calculate Separation Quality (Energy Ratio)
    # We compare the energy of the 'other' stem to the original mix
    mix_audio, _ = librosa.load(str(input_path), sr=separator.samplerate)
    energy_mix = calculate_energy(mix_audio)
    energy_guitar = calculate_energy(guitar_mono)

    ratio = (energy_guitar / energy_mix) * 100 if energy_mix > 0 else 0.0
    print(f"\n--- 📈 Separation Stats ---")
    print(f"Original Mix Energy: {energy_mix:.4f}")
    print(f"Guitar Stem Energy:  {energy_guitar:.4f}")
    print(f"Energy Ratio:        {ratio:.2f}% (Percentage of total mix)")

    # 4. Save the Result
    guitar_path = output_dir / "guitar_stem.wav"
    sf.write(str(guitar_path), guitar_mono, separator.samplerate)
    print(f"Saved cleaned guitar to: {guitar_path}")

    # 5. Side-by-Side Visualization
    if save_plot:
        plot_comparison(mix_audio, guitar_mono, separator.samplerate,
                        output_dir)

    return {
        "guitar_mono":  guitar_mono,
        "guitar_path":  guitar_path,
        "samplerate":   separator.samplerate,
        "energy_ratio": ratio,
        "output_dir":   output_dir,
    }


def plot_comparison(mix, guitar, sr, output_dir):
    """Save a side-by-side mel spectrogram comparing original mix vs guitar stem."""
    plt.figure(figsize=(14, 6))

    # Left: Original Mix Spectrogram
    plt.subplot(1, 2, 1)
    S_mix = librosa.feature.melspectrogram(y=mix, sr=sr)
    S_mix_db = librosa.power_to_db(S_mix, ref=np.max)
    librosa.display.specshow(S_mix_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title("Original Mix (Messy)")
    plt.colorbar(format='%+2.0f dB')

    # Right: Guitar Stem Spectrogram
    plt.subplot(1, 2, 2)
    S_guitar = librosa.feature.melspectrogram(y=guitar, sr=sr)
    S_guitar_db = librosa.power_to_db(S_guitar, ref=np.max)
    librosa.display.specshow(S_guitar_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title("Isolated 'Other' Stem (Clean)")
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plot_path = Path(output_dir) / "separation_comparison.png"
    plt.savefig(str(plot_path))
    plt.close()
    print(f"Saved comparison plot: {plot_path}")


def batch_separate(file_list, output_dir=None, model="htdemucs"):
    """
    Run Demucs on a list of audio files. Useful for preprocessing
    an entire dataset before feeding into the ML pipeline.

    Parameters
    ----------
    file_list : list of str or Path
        Audio files to separate.
    output_dir : str or Path, optional
        Base output directory. Each file gets its own subdirectory.
    model : str
        Demucs model name.

    Returns
    -------
    list of dict
        Results from run_splitter() for each file.
    """
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "splitter"

    results = []
    for i, fpath in enumerate(file_list):
        print(f"\n[{i+1}/{len(file_list)}] {Path(fpath).name}")
        try:
            result = run_splitter(fpath, output_dir=output_dir, model=model,
                                  save_plot=False)
            results.append(result)
        except Exception as e:
            print(f"  ⚠️  Failed: {e}")
            results.append(None)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="GuitarAI Stem Splitter (Demucs)")
    parser.add_argument("path", help="Path to mixed audio file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/splitter/)")
    parser.add_argument("--model", default="htdemucs",
                        help="Demucs model (default: htdemucs)")
    args = parser.parse_args()

    if os.path.exists(args.path):
        run_splitter(args.path, output_dir=args.output_dir, model=args.model)
    else:
        print("File not found.")