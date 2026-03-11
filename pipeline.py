"""
pipeline.py — Chain autoencoder (anomaly detection) → CNN (classification)

Flow:
    .wav path
      → Autoencoder: load → segment → features → detect anomalies
                     → { audio: np.ndarray, anomaly_ratio: float }
      → Resample 16 kHz → 32 kHz
                     → { audio: np.ndarray, anomaly_ratio: float }
      → CNN: audio → mel spectrogram → classify
      → { predicted_class, confidence, probabilities, anomaly_ratio }

Usage:
    python pipeline.py --audio path/to/audio.wav
"""

import sys
import os

# ---------------------------------------------------------------------------
# Add project root to path — all imports use full dotted module paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import numpy as np
import librosa
import joblib
import torch
import torch.nn.functional as F
from langchain_core.runnables import RunnableLambda

from Classification_Module.Model.CNN_model   import SimpleCNN
import Classification_Module.Model.CNN_config as cnn_config
import Anomaly_Detection_Module.anomaly_detection.AE_config as ae_config

from Anomaly_Detection_Module.anomaly_detection.test_audio import (
    load_and_normalize,
    segment_audio,
    extract_features,
    detect_anomalies,
    get_anomalous_audio,
)

# ---------------------------------------------------------------------------
# Settings — pulled from each module's own config
# ---------------------------------------------------------------------------
AE_SR               = ae_config.TARGET_SR
CNN_SR              = cnn_config.SAMPLE_RATE
CNN_N_MELS          = cnn_config.N_MELS
CNN_HOP_LENGTH      = cnn_config.HOP_LENGTH
CNN_CHUNK_SECS      = cnn_config.CHUNK_DURATION
CNN_TIME_STEPS      = (CNN_SR * CNN_CHUNK_SECS) // CNN_HOP_LENGTH + 1
CONFIDENCE_THRESHOLD = cnn_config.CONFIDENCE_THRESHOLD  # e.g. 60.0

AE_MODEL_PATH    = ae_config.AUTOENCODER_PATH
AE_SCALER_PATH   = ae_config.SCALER_PATH
AE_THRESH_PATH   = ae_config.THRESHOLD_PATH
CNN_MODEL_PATH   = cnn_config.CNN_MODEL_PATH
CNN_CLASSES_PATH = cnn_config.CLASSES_PATH

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
print("[pipeline] Loading autoencoder...")
_ae_model  = joblib.load(AE_MODEL_PATH)
_ae_scaler = joblib.load(AE_SCALER_PATH)
_ae_thresh = joblib.load(AE_THRESH_PATH)

print("[pipeline] Loading CNN...")
with open(CNN_CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

_cnn_model = SimpleCNN(
    num_classes=len(CLASS_NAMES),
    n_mels=CNN_N_MELS,
    time_steps=CNN_TIME_STEPS
)
_cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
_cnn_model.eval()
print(f"[pipeline] Ready — {len(CLASS_NAMES)} classes loaded.\n")


# ---------------------------------------------------------------------------
# Step 1 — Autoencoder: wav path → dict { audio, anomaly_ratio }
# ---------------------------------------------------------------------------
def run_autoencoder(audio_path: str) -> dict:
    audio = load_and_normalize(audio_path)
    if audio is None:
        raise ValueError(f"Could not load audio: {audio_path}")

    segments           = segment_audio(audio)
    features           = extract_features(segments)
    errors, is_anomaly = detect_anomalies(features)

    anomaly_ratio   = float(np.sum(is_anomaly)) / len(is_anomaly)
    anomalous_audio = get_anomalous_audio(audio, is_anomaly)

    print(f"[autoencoder] Anomaly ratio: {anomaly_ratio:.1%}")

    return {
        "audio":         anomalous_audio,  # numpy float32 @ AE_SR (16 kHz)
        "anomaly_ratio": anomaly_ratio,
    }


# ---------------------------------------------------------------------------
# Step 2 — Resample: AE_SR (16 kHz) → CNN_SR (32 kHz)
# ---------------------------------------------------------------------------
def resample_audio(data: dict) -> dict:
    audio     = data["audio"]
    resampled = librosa.resample(audio, orig_sr=AE_SR, target_sr=CNN_SR)
    print(f"[resample] {AE_SR} Hz → {CNN_SR} Hz "
          f"({len(audio)} → {len(resampled)} samples)")

    return {
        "audio":         resampled,
        "anomaly_ratio": data["anomaly_ratio"],
    }


# ---------------------------------------------------------------------------
# Step 3 — CNN: audio array → class label + probabilities
# ---------------------------------------------------------------------------
def run_cnn(data: dict) -> dict:
    audio     = data["audio"]
    chunk_len = CNN_SR * CNN_CHUNK_SECS

    # Pad or trim to exactly one chunk
    if len(audio) < chunk_len:
        audio_chunk = np.pad(audio, (0, chunk_len - len(audio)))
    else:
        audio_chunk = audio[:chunk_len]

    # Audio → mel spectrogram (identical settings to CNN training)
    mel    = librosa.feature.melspectrogram(
                y=audio_chunk,
                sr=CNN_SR,
                n_mels=CNN_N_MELS,
                hop_length=CNN_HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    # Shape: (1, 1, N_MELS, time_steps)
    tensor = torch.tensor(
        mel_db[np.newaxis, np.newaxis, :, :],
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = _cnn_model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100  # as percentage

    # Apply confidence threshold — same logic as predict.py
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown"
    else:
        predicted_class = CLASS_NAMES[pred_idx]

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 2),
        "probabilities":   {
            name: round(probs[i].item() * 100, 2)
            for i, name in enumerate(CLASS_NAMES)
        },
        "anomaly_ratio":   data["anomaly_ratio"],
    }


# ---------------------------------------------------------------------------
# Build the LangChain pipeline
# ---------------------------------------------------------------------------
pipeline = (
    RunnableLambda(run_autoencoder)
    | RunnableLambda(resample_audio)
    | RunnableLambda(run_cnn)
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomaly detection → classification pipeline")
    parser.add_argument("--audio", "-a", required=True, help="Path to .wav file")
    args = parser.parse_args()

    result = pipeline.invoke(args.audio)

    print("\n" + "=" * 50)
    print("  PIPELINE RESULT")
    print("=" * 50)
    print(f"  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.1f}%")
    print(f"  Anomaly ratio   : {result['anomaly_ratio']:.1%}")
    print("\n  All probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 0.4)  # scaled to match predict.py display
        print(f"  {cls:<55}: {prob:>5.1f}%  {bar}")