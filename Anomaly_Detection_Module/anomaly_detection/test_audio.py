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
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Anomaly_Detection_Module.anomaly_detection.AE_config import TARGET_SR, WINDOW_SECONDS, OVERLAP, AUTOENCODER_PATH, SCALER_PATH, THRESHOLD_PATH, OUTPUT_DIR
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
    autoencoder = joblib.load(AUTOENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load training threshold
    try:
        abs_threshold = joblib.load(THRESHOLD_PATH)
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


# -----------------------------------------------------------------------
# NEW: Returns anomalous audio as a numpy array instead of saving to disk.
# This is used by the LangChain pipeline to avoid disk I/O between models.
# -----------------------------------------------------------------------
def get_anomalous_audio(audio, is_anomaly, sr=TARGET_SR):
    """
    Extract anomalous regions from the original audio and return as a
    numpy array — no file I/O, no temp files.

    Args:
        audio:      Full original audio array (numpy float32)
        is_anomaly: Boolean mask per segment (True = anomaly)
        sr:         Sample rate

    Returns:
        numpy array of anomalous audio, or full audio if none detected
    """
    n_anomaly = int(np.sum(is_anomaly))

    if n_anomaly == 0:
        print("[INFO] No anomalies detected — returning full audio.")
        return audio

    if n_anomaly == len(is_anomaly):
        print("[INFO] Entire file is anomalous — returning full audio.")
        return audio

    window_size = int(sr * WINDOW_SECONDS)
    hop_size    = int(window_size * (1 - OVERLAP))

    # Build time ranges for each anomalous segment
    anomaly_ranges = []
    for i, flag in enumerate(is_anomaly):
        if flag:
            start = i * hop_size
            end   = start + window_size
            anomaly_ranges.append((start, end))

    # Merge overlapping/adjacent ranges to avoid duplicating audio
    merged = [anomaly_ranges[0]]
    for start, end in anomaly_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Extract and concatenate unique chunks
    chunks   = [audio[start : min(end, len(audio))] for start, end in merged]
    combined = np.concatenate(chunks)

    if np.max(np.abs(combined)) > 0:
        combined = combined / np.max(np.abs(combined))

    print(f"[INFO] Anomalous audio extracted: {len(combined)/sr:.2f}s "
          f"from {len(merged)} region(s)")
    return combined.astype(np.float32)


def save_anomalous_wav(audio, is_anomaly, audio_path, sr=TARGET_SR):
    """
    Extract anomalous regions from the ORIGINAL audio (no duplicates).
    (Kept for standalone use — pipeline.py uses get_anomalous_audio instead.)
    """
    n_anomaly = int(np.sum(is_anomaly))
    n_total = len(is_anomaly)

    if n_anomaly == 0:
        print("\n[5] No anomalous segments to export.")
        return None

    if n_anomaly == n_total:
        print(f"\n[5] Entire file is anomalous ({n_total}/{n_total} segments).")
        print(f"    No export needed - pass the original file directly to classification:")
        print(f"    -> {audio_path}")
        return audio_path

    combined = get_anomalous_audio(audio, is_anomaly, sr)
    # Save to output/ folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name   = os.path.splitext(os.path.basename(audio_path))[0]
    short_name  = base_name[:8].rstrip(" _-")
    output_path = os.path.join(OUTPUT_DIR, f"{short_name}_anomaly.wav")

    sf.write(output_path, combined, sr)

    original_duration = len(audio) / sr
    anomaly_duration  = len(combined) / sr
    print(f"\n[5] Anomalous audio exported (deduplicated)!")
    print(f"    Original duration: {original_duration:.2f}s -> Anomaly duration: {anomaly_duration:.2f}s")
    print(f"    Saved to: {output_path}")

    return output_path


def test_audio(audio_path):
    """
    Full pipeline: Load -> Normalize -> Segment -> Features -> Detect -> Export
    """
    print("\n" + "=" * 70)
    print("   UNDERWATER ACOUSTIC ANOMALY DETECTION - AUDIO TEST")
    print("=" * 70)

    audio = load_and_normalize(audio_path)
    if audio is None:
        return None, None

    segments             = segment_audio(audio)
    features             = extract_features(segments)
    errors, is_anomaly   = detect_anomalies(features)
    n_anomalies          = display_results(segments, errors, is_anomaly)
    anomaly_path         = save_anomalous_wav(audio, is_anomaly, audio_path)

    return n_anomalies, anomaly_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test audio file for anomaly detection")
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="Path to audio file (.wav)")
    args = parser.parse_args()

    audio_path = args.audio
    if audio_path is None:
        audio_path = input("\nEnter path to audio file (.wav): ").strip().strip('"')

    test_audio(audio_path)