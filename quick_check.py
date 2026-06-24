# quick_check.py — run this once
import librosa, soundfile as sf, tempfile, os
from basic_pitch.inference import predict as bp_predict

# grab any one test file from your splits
TEST_WAV = "data/raw/guitarset/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav"
audio, sr = librosa.load(TEST_WAV, sr=22050, mono=True)

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio, sr)
    tmp = f.name

_, _, note_events = bp_predict(tmp)
os.unlink(tmp)

print(f"Notes detected: {len(note_events)}")
print("First 5:", note_events[:5])
# Expected: ~50-200 notes, each like (0.12, 0.45, 52, 0.8, False)