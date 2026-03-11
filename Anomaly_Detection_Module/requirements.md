# Requirements & Setup Guide

## Project: UnderWater Acoustic Detection

### Quick Overview
This project detects **anomalies** in underwater acoustic recordings using an **Autoencoder** trained on ambient ocean noise. Any sound that deviates from ambient (ships, marine life) is flagged as an anomaly.

---

## Python Dependencies

```bash
pip install numpy scikit-learn librosa joblib soundfile
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, feature processing |
| `scikit-learn` | Autoencoder (MLPRegressor), StandardScaler, metrics |
| `librosa` | Audio loading, resampling, mel-spectrogram |
| `joblib` | Model saving/loading (.pkl files) |
| `soundfile` | Audio file I/O |

---

## Dataset Setup

### Primary Dataset (`data/`)

The dataset must be placed in the project root under a `data/` folder. 
**Directory Structure:**

```
data/
├── marine_life/       # Marine animal recordings (anomalies)
│   ├── Beluga, White Whale/
│   ├── Bottlenose Dolphin/
│   ├── Common Dolphin/
│   ├── ...
├── ships/             # Vessel recordings (anomalies)
│   ├── cargo/
│   ├── passengership/
│   ├── tanker/
│   └── tug/
└── ambient/           # Pure ambient recordings (used to train baseline)
    └── dragon-studio-underwater-ambience-376890.wav
```

> **Note:** The autoencoder trains by extracting the absolute quietest background noise from both the marine and ships folders, combined with the pure recordings in the `ambient/` folder.

---

## Anomaly Detection Module API

### How to use (for Classification Module teammate)

If you are building the classification module, you can import functions directly from the anomaly detection preprocessing pipeline.

**Example: Loading the dataset**
```python
from anomaly_detection.preprocessing.load_audio import load_dataset

# Loads all .wav files from data/marine_life and data/ships
# Returns lists of loaded, normalized audio arrays (sample rate 16000)
marine_audio_list, ship_audio_list = load_dataset("data")

print(f"Loaded {len(marine_audio_list)} marine files and {len(ship_audio_list)} ship files.")
```

**Example: Segmenting Audio (3s windows, 50% overlap)**
```python
from anomaly_detection.preprocessing.segment import segment_dataset

# Chops long audio files into standardized 3-second segments
# Input: List of audio arrays
# Output: 2D numpy array of segments (N_segments, segment_length)
marine_segments = segment_dataset(marine_audio_list)
ship_segments = segment_dataset(ship_audio_list)
```

**Example: Extracting Features (Log-Mel Spectrograms)**
```python
from anomaly_detection.features.logmel import logmel_dataset

# Converts raw audio segments into 128-dimensional frequency features
# Output: 2D numpy array (N_segments, 128)
marine_features = logmel_dataset(marine_segments)
```

---

## Running the Pipeline

### 1. Training the Autoencoder
To train the model on your local machine:
```bash
# --percentile 10 means train ONLY on the quietest 10% of dataset segments
python -m anomaly_detection.models.train_autoencoder_sklearn --percentile 10
```
This generates three files in the root directory:
- `autoencoder.pkl` (The trained neural network)
- `scaler.pkl` (Feature normalization weights)
- `anomaly_threshold.pkl` (The exact error threshold separating Normal from Anomaly)

### 2. Testing a single file
To run the full pipeline (load → segment → extract features → classify → export) on a specific audio file:
```bash
python -m anomaly_detection.test_audio --audio "path/to/your/audio.wav"
```

**What happens after detection:**

| Scenario | Output |
|----------|--------|
| **No anomalies** (pure ambient) | No file exported |
| **Partial anomaly** (mixed) | Anomalous regions extracted, deduplicated, saved to `output/<filename>_anomalies.wav` |
| **100% anomaly** (pure whale/ship) | No duplicate created — use original file directly |

**For the Classification Module teammate:**

After running `test_audio`, grab the exported `.wav` from the `output/` folder:
```python
from anomaly_detection.test_audio import test_audio

n_anomalies, anomaly_wav_path = test_audio("path/to/recording.wav")
# anomaly_wav_path → "output/recording_anomalies.wav"  (or original path if 100% anomaly)
# Feed anomaly_wav_path into your classification pipeline
```

> **Note:** The exported audio is **deduplicated** — overlapping segments (50% overlap) are merged into contiguous regions, so there is no redundant audio data.

### 3. Evaluating Accuracy
To test the model's performance (Precision, Recall, F1-Score) across the entire dataset:
```bash
# Tests the quietest 10% (known normal) against the loudest 30% (known anomaly)
python -m anomaly_detection.evaluation.evaluate_anomaly --ambient 10 --anomaly 30
```

---

## Project Structure

```
UnderWater-Acoustic-Detection/
├── data/                          # Dataset (must be present to train)
├── output/                        # Anomaly exports (auto-created by test_audio)
├── anomaly_detection/
│   ├── config.py                  # Standard config (Sample Rate: 16k, Segment: 3s)
│   ├── test_audio.py              # CLI script to test single files + export anomaly wav
│   ├── preprocessing/
│   │   ├── load_audio.py          # Data loading logic
│   │   ├── segment.py             # Audio segmentation logic
│   │   └── energy.py              # Spectral energy analysis
│   ├── features/
│   │   └── logmel.py              # Log-mel spectrogram extraction
│   ├── models/
│   │   ├── train_autoencoder_sklearn.py  # Primary autoencoder training
│   │   └── train_isolation_forest.py     # Alternative model (deprecated)
│   └── evaluation/
│       ├── evaluate_anomaly.py           # Autoencoder evaluation
│       └── evaluate_isolation_forest.py
├── autoencoder.pkl                # Trained model output
├── scaler.pkl                     # Scaler output
├── anomaly_threshold.pkl          # Detection threshold output
├── requirements.md                # ← This file
└── history.md                     # Development history
```
