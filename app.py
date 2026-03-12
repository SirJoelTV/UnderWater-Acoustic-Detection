"""
app.py — Flask REST API for the underwater acoustic detection pipeline

Endpoints:
    GET  /           — health check
    POST /predict    — upload a .wav file, returns full analysis result

Response shape:
    {
      "predicted_class": "Ships",
      "confidence":       92.3,
      "anomaly_ratio":    75.0,
      "probabilities":    { "Ships": 92.3, ... },
      "anomalies": [
        { "id":1, "start":1.24, "end":3.87, "duration":2.63,
          "peak_db":-18.4, "confidence":87, "class":"Ships" }
      ],
      "features": {
        "duration":2.63, "sample_rate":16000, "channels":1,
        "rms_db":-18.4, "spectral_centroid":2340.5,
        "spectral_rolloff":4200.3, "zcr":0.045, "bandwidth":1200.3
      }
    }

Usage:
    python app.py
"""

import os
import sys
import tempfile
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import webbrowser
import threading

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# Import pipeline and anomaly helpers
# ---------------------------------------------------------------------------
from pipeline import pipeline, AE_SR, CNN_SR

from Anomaly_Detection_Module.anomaly_detection.test_audio import (
    load_and_normalize,
    segment_audio,
    extract_features,
    detect_anomalies,
    get_anomalous_audio,
)
import Anomaly_Detection_Module.anomaly_detection.AE_config as ae_config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"wav"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_anomaly_segments(audio, is_anomaly, errors, sr):
    """
    Build a list of anomaly dicts with timestamps and peak energy.
    """
    window_size = int(sr * ae_config.WINDOW_SECONDS)
    hop_size    = int(window_size * (1 - ae_config.OVERLAP))

    segments = []
    for i, flag in enumerate(is_anomaly):
        if not flag:
            continue
        start_s  = round(i * hop_size / sr, 3)
        end_s    = round(start_s + ae_config.WINDOW_SECONDS, 3)
        duration = round(end_s - start_s, 3)

        # Peak energy of this segment (dB)
        seg_start = i * hop_size
        seg_end   = seg_start + window_size
        segment   = audio[seg_start : min(seg_end, len(audio))]
        rms       = float(np.sqrt(np.mean(segment ** 2) + 1e-9))
        peak_db   = round(20 * np.log10(rms), 1)

        # Use reconstruction error as a proxy for confidence (0–100)
        max_err    = float(np.max(errors)) if np.max(errors) > 0 else 1.0
        confidence = round(float(errors[i]) / max_err * 100, 1)

        segments.append({
            "id":         len(segments) + 1,
            "start":      start_s,
            "end":        end_s,
            "duration":   duration,
            "peak_db":    peak_db,
            "confidence": confidence,
            "class":      "—",  # filled in after CNN classification
        })

    return segments


def compute_features(audio, sr):
    """Compute acoustic features for the uploaded audio."""
    duration   = round(len(audio) / sr, 3)
    rms        = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    rms_db     = round(20 * np.log10(rms), 1)

    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=audio,  sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(audio)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

    return {
        "duration":          duration,
        "sample_rate":       sr,
        "channels":          1,
        "rms_db":            rms_db,
        "spectral_centroid": round(float(np.mean(centroid)), 1),
        "spectral_rolloff":  round(float(np.mean(rolloff)),  1),
        "zcr":               round(float(np.mean(zcr)),      4),
        "bandwidth":         round(float(np.mean(bandwidth)),1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Underwater Acoustic Detection API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    # --- Validate ---
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a .wav file with key 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files are supported."}), 400

    # --- Save to temp file ---
    try:
        suffix   = "_" + secure_filename(file.filename)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(tmp_file.name)
        tmp_path = tmp_file.name
        tmp_file.close()
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500

    try:
        # --- Run anomaly detection separately to get timestamps ---
        audio = load_and_normalize(tmp_path)
        if audio is None:
            return jsonify({"error": "Could not load audio file."}), 500

        segments           = segment_audio(audio)
        feat_vectors       = extract_features(segments)
        errors, is_anomaly = detect_anomalies(feat_vectors)

        # Build anomaly segment list with timestamps
        anomaly_list = get_anomaly_segments(audio, is_anomaly, errors, ae_config.TARGET_SR)

        # Compute acoustic features of the full audio
        features = compute_features(audio, ae_config.TARGET_SR)

        # --- Run full pipeline for classification ---
        result = pipeline.invoke(tmp_path)

        # Attach predicted class to each anomaly row
        for a in anomaly_list:
            a["class"] = result["predicted_class"]

    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500
    finally:
        os.unlink(tmp_path)

    # --- Build response ---
    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence":      result["confidence"],
        "anomaly_ratio":   round(result["anomaly_ratio"] * 100, 2),
        "probabilities":   result["probabilities"],
        "anomalies":       anomaly_list,
        "features":        features,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("file:///d:/Main Project/UnderWater-Acoustic-Detection/Front end/home.html")).start()
    app.run(debug=False, host="0.0.0.0", port=5000)