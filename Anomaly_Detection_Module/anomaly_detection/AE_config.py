# ===================================================
# Anomaly Detection Module - Central Configuration
# ===================================================
# All paths, model settings, and audio parameters
# are defined here. Update this file to reconfigure
# the entire pipeline.
# ===================================================

import os

# --- Project Root ---
# All paths are relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Audio Settings ---
TARGET_SR = 16000          # Sample rate (Hz) — all audio resampled to this
WINDOW_SECONDS = 3.0       # Segment length (seconds)
OVERLAP = 0.5              # Segment overlap (0.5 = 50%)

# --- Feature Extraction ---
N_MFCC = 40                # MFCC coefficients (if used)
N_FFT = 1024               # FFT window size
HOP_LENGTH = 512           # FFT hop length
N_MELS = 128               # Number of mel bands for log-mel spectrogram

# --- Dataset Paths ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MARINE_DIR = os.path.join(DATA_ROOT, "marine_life")
SHIPS_DIR = os.path.join(DATA_ROOT, "ships")
AMBIENT_DIR = os.path.join(DATA_ROOT, "ambient")

# --- Output Path ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# --- Model Paths ---
MODEL_DIR = PROJECT_ROOT  # Models saved in project root
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "anomaly_threshold.pkl")

# --- Training Defaults ---
AMBIENT_PERCENTILE = 10    # Bottom X% energy = ambient (default for training)
HIDDEN_LAYERS = (128, 64, 32, 64, 128)  # Autoencoder architecture
MAX_ITER = 300             # Max training iterations
