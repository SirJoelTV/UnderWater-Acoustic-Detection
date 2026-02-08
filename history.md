Phase 2: Shift to Anomaly Detection (MFCC + Autoencoder)

What we did:

Reframed the problem as anomaly detection.

Built a full pipeline:

Audio loading

Segmentation

MFCC feature extraction

Autoencoder training

Threshold-based anomaly detection

Why we changed approach:

Anomaly detection works with limited labels.

More realistic for underwater monitoring scenarios.

🔹 Phase 3: Energy-Based Normal Data Refinement

What we did:

Observed poor anomaly separation when training on all marine sounds.

Introduced energy-based filtering.

Selected low-energy segments as a proxy for ambient underwater noise.

Retrained autoencoder only on ambient segments.

Outcome:

Training was stable.

Reconstruction loss converged.

However, ship sounds were still reconstructed well.

🔹 Phase 4: Evaluation and Failure Analysis

What we observed:

Very low anomaly recall for ship sounds.

High overlap in reconstruction error between ambient and ship sounds.

Why it failed:

MFCC features compress spectral energy information.

Ship noise and ambient noise overlap significantly in MFCC space.

Autoencoder generalized too well and reconstructed anomalies.

Key Insight:

The limitation lies in feature representation, not implementation.

🔹 Current Conclusion

MFCC + autoencoder is insufficient for separating ship noise from ambient underwater noise in an unsupervised setting.

The pipeline is correct, but the representation is not discriminative enough.

🔹 Planned Next Steps

Replace MFCC with log-mel spectrogram energy features.

Use distance-based anomaly models (Isolation Forest / One-Class SVM).

Preserve energy distribution instead of compressing it.

****
### Segment Duration Experiment (MFCC + Autoencoder)

We evaluated the impact of segment duration on anomaly detection performance
by modifying WINDOW_SECONDS in the segmentation module.

Tested segment durations:
- 1 second (baseline)
- 3 seconds
- 5 seconds

Observations:
- Increasing segment duration reduced variance in reconstruction error.
- Longer segments provided better temporal context for stationary sounds.
- However, MFCC features still showed overlap between ambient noise and ship noise.

Inference:
Segment duration alone does not resolve the anomaly detection failure.
Feature representation remains the primary bottleneck.

Decision:
Proceed to log-mel energy features while keeping longer segments.

---

## Phase 5: Log-Mel Autoencoder & Isolation Forest Comparison (2026-02-08)

### Changes Made

**1. Switched to Log-Mel Spectrogram Features**
- Replaced MFCC with log-mel spectrogram (64 mel bands)
- Output: 128 features per segment (mean + std over time)
- Better preservation of energy distribution

**2. True Ambient Noise Training**
- Train on quietest 10% of ALL data (both marine + ships)
- This represents true ambient ocean background
- Both ships AND marine life now detected as anomalies

**3. Random Sampling for Evaluation**
- Instead of top X% energy, randomly sample from non-ambient segments
- More realistic test scenario

---

### Autoencoder Results

**Configuration:**
- Ambient: Bottom 10% energy segments (643 samples)
- Anomaly: Random 30% sample from non-ambient (1735 samples)
- Architecture: 128-64-32-64-128
- Features: Log-mel spectrogram (128 features)

**Confusion Matrix:**
```
              Predicted
              Ambient    Anomaly
Actual Ambient    578         65
       Anomaly     25        618
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 93.00% |
| Precision | 90.48% |
| Recall | 96.11% |
| F1-Score | 93.21% |
| ROC-AUC | 0.9779 |
| Separation Ratio | 7.57x |

---

### Isolation Forest Results

**Configuration:**
- Same ambient/anomaly split as autoencoder
- n_estimators: 200
- contamination: 0.05

**Confusion Matrix:**
```
              Predicted
              Ambient    Anomaly
Actual Ambient    610         33
       Anomaly    537        106
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 55.68% |
| Precision | 76.26% |
| Recall | 16.49% |
| F1-Score | 27.11% |
| ROC-AUC | 0.6554 |

---

### Model Comparison

| Metric | Autoencoder | Isolation Forest | Winner |
|--------|-------------|------------------|--------|
| Accuracy | 93.00% | 55.68% | ✅ Autoencoder |
| Precision | 90.48% | 76.26% | ✅ Autoencoder |
| Recall | 96.11% | 16.49% | ✅ Autoencoder |
| F1-Score | 93.21% | 27.11% | ✅ Autoencoder |
| ROC-AUC | 0.978 | 0.655 | ✅ Autoencoder |

**Conclusion:** Autoencoder with log-mel features significantly outperforms Isolation Forest for underwater acoustic anomaly detection.

---

### Files Modified/Created

| File | Description |
|------|-------------|
| `anomaly_detection/models/train_autoencoder_sklearn.py` | Train on true ambient noise |
| `anomaly_detection/models/train_isolation_forest.py` | Train Isolation Forest on ambient |
| `anomaly_detection/evaluation/evaluate_anomaly.py` | Metrics with confusion matrix |
| `anomaly_detection/evaluation/evaluate_isolation_forest.py` | Isolation Forest evaluation |

### Commands

```bash
# Train autoencoder
python -m anomaly_detection.models.train_autoencoder_sklearn --percentile 10

# Evaluate autoencoder
python -m anomaly_detection.evaluation.evaluate_anomaly --ambient 10 --anomaly 30

# Train Isolation Forest
python -m anomaly_detection.models.train_isolation_forest --percentile 10

# Evaluate Isolation Forest
python -m anomaly_detection.evaluation.evaluate_isolation_forest --ambient 10 --anomaly 30
```
