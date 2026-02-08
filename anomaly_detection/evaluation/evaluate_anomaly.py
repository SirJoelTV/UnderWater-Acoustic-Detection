"""
Anomaly Detection Evaluation

Tests the autoencoder against:
- Ambient noise (TRUE baseline - should have LOW error)
- Ships (ANOMALY - should have HIGH error)
- Marine life (ANOMALY - should have HIGH error)
"""

import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.logmel import logmel_dataset
from anomaly_detection.preprocessing.energy import segment_energy


def reconstruction_error(model, X):
    """Compute reconstruction error for each sample."""
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def print_confusion_matrix(cm, labels=["Normal", "Anomaly"]):
    """Pretty print confusion matrix."""
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"\n{'':15} Predicted")
    print(f"{'':15} {labels[0]:>10} {labels[1]:>10}")
    print("-" * 40)
    print(f"Actual {labels[0]:>8} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"       {labels[1]:>8} {cm[1,0]:>10} {cm[1,1]:>10}")
    print("-" * 40)
    
    print(f"\nTrue Negatives (TN):  {cm[0,0]:>5} - Correctly identified as Normal")
    print(f"False Positives (FP): {cm[0,1]:>5} - Normal misclassified as Anomaly")
    print(f"False Negatives (FN): {cm[1,0]:>5} - Anomaly misclassified as Normal")
    print(f"True Positives (TP):  {cm[1,1]:>5} - Correctly identified as Anomaly")


def evaluate_anomaly_detection(threshold_percentile=90, ambient_percentile=10, anomaly_percentile=30, balanced=True):
    """
    Evaluate anomaly detection.
    
    Normal = Ambient noise (low-energy segments)
    Anomaly = Ships AND marine life (high-energy segments with activity)
    
    Args:
        threshold_percentile: Threshold on ambient errors
        ambient_percentile: What defines ambient (bottom X% energy) - default 10%
        anomaly_percentile: What defines anomaly (top X% energy) - default 30%
        balanced: Balance dataset for fair evaluation
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 60)
    print("\nExpected result:")
    print("  - Ambient noise: LOW reconstruction error (Normal)")
    print("  - Ships: HIGH reconstruction error (Anomaly)")
    print("  - Marine life: HIGH reconstruction error (Anomaly)")
    
    # Load trained model and scaler
    print("\n[INFO] Loading trained autoencoder...")
    autoencoder = joblib.load("autoencoder.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Load dataset
    print("[INFO] Loading dataset...")
    marine, ships = load_dataset("data")
    
    # Segment
    print("[INFO] Segmenting audio...")
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)
    
    all_segments = np.vstack([marine_segments, ship_segments])
    print(f"[INFO] Total segments: {len(all_segments)}")
    
    # Separate into ambient vs non-ambient based on energy
    all_energies = segment_energy(all_segments)
    ambient_threshold = np.percentile(all_energies, ambient_percentile)
    
    # Ambient = quietest segments (for comparison)
    ambient_mask = all_energies <= ambient_threshold
    ambient_segments = all_segments[ambient_mask]
    
    # Non-ambient = everything else (potential anomalies)
    non_ambient_mask = all_energies > ambient_threshold
    non_ambient_segments = all_segments[non_ambient_mask]
    
    # Randomly sample from non-ambient for testing
    rng = np.random.default_rng(seed=42)
    n_anomaly_samples = int(len(non_ambient_segments) * (anomaly_percentile / 100))
    sample_idx = rng.choice(len(non_ambient_segments), size=n_anomaly_samples, replace=False)
    anomaly_segments = non_ambient_segments[sample_idx]
    
    print(f"\n[INFO] Ambient segments (bottom {ambient_percentile}% energy): {len(ambient_segments)}")
    print(f"[INFO] Non-ambient segments available: {len(non_ambient_segments)}")
    print(f"[INFO] Anomaly segments (random {anomaly_percentile}% sample): {len(anomaly_segments)}")
    
    # Extract features
    print("[INFO] Extracting features...")
    ambient_features = logmel_dataset(ambient_segments)
    anomaly_features = logmel_dataset(anomaly_segments)
    
    # Scale
    ambient_scaled = scaler.transform(ambient_features)
    anomaly_scaled = scaler.transform(anomaly_features)
    
    # Compute reconstruction errors
    print("[INFO] Computing reconstruction errors...")
    ambient_error = reconstruction_error(autoencoder, ambient_scaled)
    anomaly_error = reconstruction_error(autoencoder, anomaly_scaled)
    
    print(f"\n[INFO] Reconstruction Errors:")
    print(f"  Ambient - Mean: {np.mean(ambient_error):.4f}, Std: {np.std(ambient_error):.4f}")
    print(f"  Anomaly - Mean: {np.mean(anomaly_error):.4f}, Std: {np.std(anomaly_error):.4f}")
    
    # Check separation
    separation = np.mean(anomaly_error) / np.mean(ambient_error)
    print(f"\n[INFO] Separation ratio: {separation:.2f}x")
    if separation > 1.5:
        print("       GOOD: Anomalies have higher error than ambient")
    else:
        print("       WARNING: Poor separation between ambient and anomaly")
    
    # Balance if needed
    if balanced:
        min_size = min(len(ambient_error), len(anomaly_error))
        rng = np.random.default_rng(seed=42)
        
        idx_ambient = rng.choice(len(ambient_error), size=min_size, replace=False)
        idx_anomaly = rng.choice(len(anomaly_error), size=min_size, replace=False)
        
        ambient_error = ambient_error[idx_ambient]
        anomaly_error = anomaly_error[idx_anomaly]
        print(f"[INFO] Balanced: {min_size} samples each")
    
    # Threshold from ambient data
    threshold = np.percentile(ambient_error, threshold_percentile)
    print(f"\n[INFO] Threshold ({threshold_percentile}th percentile of ambient): {threshold:.6f}")
    
    # Create labels and predictions
    y_true = np.concatenate([
        np.zeros(len(ambient_error)),   # 0 = Normal (ambient)
        np.ones(len(anomaly_error))     # 1 = Anomaly (ships + marine)
    ])
    
    y_pred = np.concatenate([
        (ambient_error > threshold).astype(int),
        (anomaly_error > threshold).astype(int)
    ])
    
    y_scores = np.concatenate([ambient_error, anomaly_error])
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm, labels=["Ambient", "Anomaly"])
    
    # Metrics
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = 0.0
    
    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%  (When predicted anomaly, how often correct)")
    print(f"Recall:    {recall * 100:.2f}%  (How many actual anomalies detected)")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Classification Report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(
        y_true, y_pred, 
        target_names=["Ambient", "Anomaly"],
        zero_division=0
    ))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "threshold": threshold
    }


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection")
    parser.add_argument("--threshold", type=float, default=90,
                        help="Threshold percentile on ambient errors (default: 90)")
    parser.add_argument("--ambient", type=float, default=10,
                        help="Ambient percentile - bottom X%% energy (default: 10)")
    parser.add_argument("--anomaly", type=float, default=30,
                        help="Anomaly percentile - top X%% energy (default: 30)")
    parser.add_argument("--unbalanced", action="store_true",
                        help="Use unbalanced dataset")
    args = parser.parse_args()
    
    evaluate_anomaly_detection(
        threshold_percentile=args.threshold,
        ambient_percentile=args.ambient,
        anomaly_percentile=args.anomaly,
        balanced=not args.unbalanced
    )
