"""
Test Audio for Anomaly Detection

Takes a user-provided audio file, applies the full pipeline
(load → normalize → segment → extract features → detect anomaly)
and classifies each segment as Normal or Anomaly.

Usage:
    python -m anomaly_detection.test_audio
    python -m anomaly_detection.test_audio --audio "path/to/file.wav"
"""

import numpy as np
import joblib
import librosa
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.config import TARGET_SR, WINDOW_SECONDS, OVERLAP
from anomaly_detection.features.logmel import logmel_features


def load_and_normalize(audio_path, sr=TARGET_SR):
    """Load audio file, resample, and normalize."""
    print(f"\n[1] Loading audio: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return None

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Normalize to [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    duration = len(audio) / sr
    print(f"    Sample rate: {sr} Hz")
    print(f"    Duration: {duration:.2f}s")
    print(f"    Samples: {len(audio)}")

    return audio


def segment_audio(audio, sr=TARGET_SR):
    """Split audio into fixed-length overlapping segments."""
    window_size = int(sr * WINDOW_SECONDS)
    hop_size = int(window_size * (1 - OVERLAP))

    print(f"\n[2] Segmenting audio")
    print(f"    Window: {WINDOW_SECONDS}s ({window_size} samples)")
    print(f"    Overlap: {OVERLAP * 100:.0f}%")
    print(f"    Hop: {hop_size} samples")

    segments = []
    if len(audio) < window_size:
        print(f"    [WARNING] Audio too short ({len(audio)} < {window_size}), padding with zeros")
        padded = np.zeros(window_size)
        padded[:len(audio)] = audio
        segments.append(padded)
    else:
        for start in range(0, len(audio) - window_size + 1, hop_size):
            segments.append(audio[start:start + window_size])

    segments = np.array(segments)
    print(f"    Segments created: {len(segments)}")

    return segments


def extract_features(segments):
    """Extract log-mel features from each segment."""
    print(f"\n[3] Extracting log-mel features")

    features = np.array([logmel_features(seg) for seg in segments], dtype=np.float32)

    print(f"    Feature shape: {features.shape}")
    print(f"    Features per segment: {features.shape[1]}")

    return features


def detect_anomalies(features):
    """
    Hybrid anomaly detection: absolute + relative scoring.

    - Absolute: Uses training threshold (catches uniform anomaly files)
    - Relative: Uses IQR within file (catches outliers in mixed files)
    - A segment is anomaly if it fails EITHER check
    """
    print(f"\n[4] Running anomaly detection (hybrid scoring)")

    # Load model and scaler
    autoencoder = joblib.load("autoencoder.pkl")
    scaler = joblib.load("scaler.pkl")

    # Load training threshold
    try:
        abs_threshold = joblib.load("anomaly_threshold.pkl")
    except FileNotFoundError:
        abs_threshold = None

    # Scale features
    features_scaled = scaler.transform(features)

    # Compute reconstruction error
    reconstructed = autoencoder.predict(features_scaled)
    errors = np.mean((features_scaled - reconstructed) ** 2, axis=1)

    # Relative threshold using IQR
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr = q3 - q1
    rel_threshold = q3 + 1.5 * iqr

    if iqr < 0.001:
        rel_threshold = np.median(errors) * 1.5

    # Determine which check to use per segment
    # anomaly = fails absolute OR relative
    is_anomaly = np.zeros(len(errors), dtype=bool)

    if abs_threshold is not None:
        is_anomaly |= (errors > abs_threshold)
    is_anomaly |= (errors > rel_threshold)

    print(f"    Error stats:")
    print(f"      Min: {np.min(errors):.6f}")
    print(f"      Median: {np.median(errors):.6f}")
    print(f"      Max: {np.max(errors):.6f}")
    if abs_threshold is not None:
        print(f"    Absolute threshold (training): {abs_threshold:.6f}")
    print(f"    Relative threshold (IQR): {rel_threshold:.6f}")

    return errors, is_anomaly


def display_results(segments, errors, is_anomaly, sr=TARGET_SR):
    """Display per-segment anomaly detection results."""
    hop_size = int(sr * WINDOW_SECONDS * (1 - OVERLAP))
    n_anomalies = 0

    print(f"\n{'=' * 70}")
    print(f"{'Seg':>4} | {'Start':>7} | {'End':>7} | {'Error':>10} | {'Result'}")
    print(f"{'-' * 70}")

    for i, error in enumerate(errors):
        start_time = i * hop_size / sr
        end_time = start_time + WINDOW_SECONDS

        if is_anomaly[i]:
            n_anomalies += 1
            label = "** ANOMALY **"
        else:
            label = "   Normal"

        print(f"{i:4d} | {start_time:6.2f}s | {end_time:6.2f}s | {error:10.6f} | {label}")

    print(f"{'=' * 70}")
    print(f"\nTotal segments: {len(errors)}")
    print(f"Anomalies: {n_anomalies} ({100 * n_anomalies / len(errors):.1f}%)")
    print(f"Normal: {len(errors) - n_anomalies} ({100 * (len(errors) - n_anomalies) / len(errors):.1f}%)")

    return n_anomalies


def test_audio(audio_path):
    """
    Full pipeline: Load -> Normalize -> Segment -> Features -> Detect

    Args:
        audio_path: Path to the .wav audio file
    """
    print("\n" + "=" * 70)
    print("   UNDERWATER ACOUSTIC ANOMALY DETECTION - AUDIO TEST")
    print("=" * 70)

    # Step 1: Load and normalize
    audio = load_and_normalize(audio_path)
    if audio is None:
        return

    # Step 2: Segment
    segments = segment_audio(audio)

    # Step 3: Extract features
    features = extract_features(segments)

    # Step 4: Detect anomalies
    errors, threshold = detect_anomalies(features)

    # Step 5: Display results
    n_anomalies = display_results(segments, errors, threshold)

    return n_anomalies


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test audio file for anomaly detection")
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="Path to audio file (.wav)")
    args = parser.parse_args()

    # If no audio path provided, ask user
    audio_path = args.audio
    if audio_path is None:
        audio_path = input("\nEnter path to audio file (.wav): ").strip().strip('"')

    test_audio(audio_path)

