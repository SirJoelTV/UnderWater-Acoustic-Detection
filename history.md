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


