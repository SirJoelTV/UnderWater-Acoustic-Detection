"""
Train Isolation Forest on Ambient Noise

Same approach as autoencoder:
- Train on low-energy ambient segments
- Both ships AND marine life will be detected as anomalies
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.logmel import logmel_dataset
from anomaly_detection.preprocessing.energy import segment_energy


def select_ambient_segments(segments, percentile=10):
    """Select quietest segments as ambient noise."""
    energies = segment_energy(segments)
    threshold = np.percentile(energies, percentile)
    mask = energies <= threshold
    return segments[mask]


def train_isolation_forest(
    ambient_percentile=10,
    n_estimators=200,
    contamination=0.05,
    save_path="."
):
    """
    Train Isolation Forest on ambient noise.
    
    Args:
        ambient_percentile: Use bottom X% energy as ambient
        n_estimators: Number of trees
        contamination: Expected anomaly ratio in training data
        save_path: Where to save model
    """
    print("\n" + "=" * 60)
    print("ISOLATION FOREST TRAINING ON AMBIENT NOISE")
    print("=" * 60)
    
    # Load ALL data
    print("\n[INFO] Loading dataset...")
    marine, ships = load_dataset("data")
    
    # Segment
    print("[INFO] Segmenting audio...")
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)
    
    all_segments = np.vstack([marine_segments, ship_segments])
    print(f"[INFO] Total segments: {len(all_segments)}")
    
    # Select ambient only
    print(f"[INFO] Selecting ambient noise (bottom {ambient_percentile}% energy)...")
    ambient_segments = select_ambient_segments(all_segments, percentile=ambient_percentile)
    print(f"[INFO] Ambient segments: {len(ambient_segments)}")
    
    # Extract features
    print("[INFO] Extracting features...")
    ambient_features = logmel_dataset(ambient_segments)
    print(f"[INFO] Feature shape: {ambient_features.shape}")
    
    # Scale
    print("[INFO] Fitting scaler...")
    scaler = StandardScaler()
    X = scaler.fit_transform(ambient_features)
    
    # Train Isolation Forest
    print(f"\n[INFO] Training Isolation Forest...")
    print(f"[INFO] n_estimators: {n_estimators}")
    print(f"[INFO] contamination: {contamination}")
    
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    iso.fit(X)
    
    # Compute training scores
    train_scores = iso.decision_function(X)
    train_pred = iso.predict(X)
    
    n_normal = np.sum(train_pred == 1)
    n_anomaly = np.sum(train_pred == -1)
    
    print(f"\n[INFO] Training completed!")
    print(f"[INFO] Training score - Mean: {np.mean(train_scores):.4f}, Std: {np.std(train_scores):.4f}")
    print(f"[INFO] Training predictions: {n_normal} normal, {n_anomaly} anomaly")
    
    # Test on non-ambient
    print("\n[INFO] Verifying on non-ambient segments...")
    all_energies = segment_energy(all_segments)
    high_energy_mask = all_energies > np.percentile(all_energies, 90)
    non_ambient = all_segments[high_energy_mask][:500]
    
    if len(non_ambient) > 0:
        non_ambient_features = logmel_dataset(non_ambient)
        non_ambient_scaled = scaler.transform(non_ambient_features)
        non_ambient_scores = iso.decision_function(non_ambient_scaled)
        non_ambient_pred = iso.predict(non_ambient_scaled)
        
        n_detected = np.sum(non_ambient_pred == -1)
        print(f"[INFO] Non-ambient score - Mean: {np.mean(non_ambient_scores):.4f}")
        print(f"[INFO] Anomalies detected: {n_detected}/{len(non_ambient)} ({100*n_detected/len(non_ambient):.1f}%)")
    
    # Save
    iso_path = os.path.join(save_path, "isolation_forest.pkl")
    scaler_iso_path = os.path.join(save_path, "scaler_isolation_forest.pkl")
    
    joblib.dump(iso, iso_path)
    joblib.dump(scaler, scaler_iso_path)
    
    print(f"\n[INFO] Models saved:")
    print(f"  - {iso_path}")
    print(f"  - {scaler_iso_path}")
    
    return iso, scaler


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Isolation Forest")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Ambient percentile (default: 10)")
    parser.add_argument("--estimators", type=int, default=200,
                        help="Number of trees (default: 200)")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Contamination ratio (default: 0.05)")
    args = parser.parse_args()
    
    train_isolation_forest(
        ambient_percentile=args.percentile,
        n_estimators=args.estimators,
        contamination=args.contamination
    )
