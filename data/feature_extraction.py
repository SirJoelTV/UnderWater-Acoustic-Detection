# data/feature_extraction.py

import librosa
import numpy as np
from scipy import signal

def extract_mel_spectrogram(audio, sr, n_mels=128, window_size=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=window_size
    )
    return librosa.power_to_db(mel_spec, ref=np.max)

def extract_mfcc(audio, sr, n_mfcc=13, window_size=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=window_size
    )
    return mfcc

def extract_spectral_features(audio, sr, hop_length=512):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)
    zero_crossing = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    return {
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zero_crossing
    }

def extract_lofar(audio, window_size=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)
    magnitude = np.abs(stft)
    lofar = np.log10(magnitude + 1e-10)
    return lofar

def butter_highpass(audio, sr, cutoff=50, order=4):
    b, a = signal.butter(order, cutoff, btype='high', fs=sr)
    return signal.filtfilt(b, a, audio)

def extract_all_features(audio, sr, config=None):
    if config is None:
        config = {
            "n_mels": 128,
            "n_mfcc": 13,
            "window_size": 2048,
            "hop_length": 512
        }
    # Highpass for underwater noise
    audio = butter_highpass(audio, sr)
    # Normalize
    audio = librosa.util.normalize(audio)
    # Feature extraction
    features = {
        "mel_spec": extract_mel_spectrogram(audio, sr, config["n_mels"], config["window_size"], config["hop_length"]),
        "mfcc": extract_mfcc(audio, sr, config["n_mfcc"], config["window_size"], config["hop_length"]),
        "lofar": extract_lofar(audio, config["window_size"], config["hop_length"])
    }
    features.update(extract_spectral_features(audio, sr, config["hop_length"]))
    return features

# Example usage
if __name__ == "__main__":
    import soundfile as sf
    # Replace with your audio file path
    audio_path = "sample.wav"
    audio, sr = sf.read(audio_path)
    features = extract_all_features(audio, sr)
    print("Extracted Features:")
    for key in features:
        print(f"{key}: shape {features[key].shape}")

