import os
import librosa
import numpy as np

TARGET_SR = 16000


def collect_wav_files(base_dir):
    wav_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    return wav_files


def load_audio(path, sr=TARGET_SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio


def load_dataset(data_root):
    marine_dir = os.path.join(data_root, "marine_life")
    ship_dir = os.path.join(data_root, "ships")

    marine_files = collect_wav_files(marine_dir)
    ship_files = collect_wav_files(ship_dir)

    print(f"[INFO] Marine files: {len(marine_files)}")
    print(f"[INFO] Ship files: {len(ship_files)}")

    marine_audio = [load_audio(f) for f in marine_files]
    ship_audio = [load_audio(f) for f in ship_files]

    return marine_audio, ship_audio
