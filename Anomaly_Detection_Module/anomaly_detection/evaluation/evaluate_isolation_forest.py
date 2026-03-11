"""
Isolation Forest Evaluation

Same evaluation approach as autoencoder for fair comparison.
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
from Anomaly_Detection_Module.anomaly_detection.AE_config import DATA_ROOT


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


def evaluate_isolation_forest(ambient_percentile=10, anomaly_percentile=30, balanced=True):
    """
    Evaluate Isolation Forest.
    
    Args:
        ambient_percentile: Bottom X% energy as ambient
        anomaly_percentile: Random X% sample from non-ambient as anomaly
        balanced: Balance dataset
    """
    print("\n" + "=" * 60)
    print("ISOLATION FOREST EVALUATION")
    print("=" * 60)
    
    # Load model
    print("\n[INFO] Loading Isolation Forest model...")
    try:
        iso = joblib.load("isolation_forest.pkl")
        scaler = joblib.load("scaler_isolation_forest.pkl")
    except FileNotFoundError:
        print("[ERROR] Model not found! Run training first:")
        print("  python -m anomaly_detection.models.train_isolation_forest")
        return None
    
    # Load data
    print("[INFO] Loading dataset...")
    marine, ships = load_dataset(DATA_ROOT)
    
    # Segment
    print("[INFO] Segmenting audio...")
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)
    all_segments = np.vstack([marine_segments, ship_segments])
    print(f"[INFO] Total segments: {len(all_segments)}")
    
    # Separate ambient vs non-ambient
    all_energies = segment_energy(all_segments)
    ambient_threshold = np.percentile(all_energies, ambient_percentile)
    
    ambient_mask = all_energies <= ambient_threshold
    ambient_segments = all_segments[ambient_mask]
    
    non_ambient_mask = all_energies > ambient_threshold
    non_ambient_segments = all_segments[non_ambient_mask]
    
    # Random sample from non-ambient
    rng = np.random.default_rng(seed=42)
    n_samples = int(len(non_ambient_segments) * (anomaly_percentile / 100))
    sample_idx = rng.choice(len(non_ambient_segments), size=n_samples, replace=False)
    anomaly_segments = non_ambient_segments[sample_idx]
    
    print(f"\n[INFO] Ambient segments: {len(ambient_segments)}")
    print(f"[INFO] Anomaly segments (random {anomaly_percentile}%): {len(anomaly_segments)}")
    
    # Extract features
    print("[INFO] Extracting features...")
    ambient_features = logmel_dataset(ambient_segments)
    anomaly_features = logmel_dataset(anomaly_segments)
    
    ambient_scaled = scaler.transform(ambient_features)
    anomaly_scaled = scaler.transform(anomaly_features)
    
    # Get decision scores (for ROC-AUC)
    ambient_scores = iso.decision_function(ambient_scaled)
    anomaly_scores = iso.decision_function(anomaly_scaled)
    
    print(f"\n[INFO] Decision Scores (negative = anomaly):")
    print(f"  Ambient - Mean: {np.mean(ambient_scores):.4f}, Std: {np.std(ambient_scores):.4f}")
    print(f"  Anomaly - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
    
    # Predict (-1 = anomaly, 1 = normal)
    ambient_pred = iso.predict(ambient_scaled)
    anomaly_pred = iso.predict(anomaly_scaled)
    
    # Balance if needed
    if balanced:
        min_size = min(len(ambient_pred), len(anomaly_pred))
        idx_a = rng.choice(len(ambient_pred), size=min_size, replace=False)
        idx_b = rng.choice(len(anomaly_pred), size=min_size, replace=False)
        
        ambient_pred = ambient_pred[idx_a]
        anomaly_pred = anomaly_pred[idx_b]
        ambient_scores = ambient_scores[idx_a]
        anomaly_scores = anomaly_scores[idx_b]
        print(f"[INFO] Balanced: {min_size} samples each")
    
    # Create labels
    # y_true: 0 = ambient (normal), 1 = anomaly
    # y_pred: -1 from model = anomaly (1), 1 from model = normal (0)
    y_true = np.concatenate([
        np.zeros(len(ambient_pred)),   # ambient = normal = 0
        np.ones(len(anomaly_pred))     # anomaly = 1
    ])
    
    y_pred = np.concatenate([
        (ambient_pred == -1).astype(int),   # -1 = anomaly = 1
        (anomaly_pred == -1).astype(int)
    ])
    
    # Scores for ROC (invert because lower score = more anomalous)
    y_scores = -np.concatenate([ambient_scores, anomaly_scores])
    
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
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
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
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest")
    parser.add_argument("--ambient", type=float, default=10,
                        help="Ambient percentile (default: 10)")
    parser.add_argument("--anomaly", type=float, default=30,
                        help="Anomaly sample percentile (default: 30)")
    parser.add_argument("--unbalanced", action="store_true",
                        help="Use unbalanced dataset")
    args = parser.parse_args()
    
    evaluate_isolation_forest(
        ambient_percentile=args.ambient,
        anomaly_percentile=args.anomaly,
        balanced=not args.unbalanced
    )
