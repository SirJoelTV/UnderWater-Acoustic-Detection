"""
Train Autoencoder on TRUE Ambient Noise

Ambient noise = quiet ocean background with NO activity (no ships, no marine life)
Any sound with activity (ships, whales, dolphins) = ANOMALY

Training approach:
1. Use ONLY the lowest-energy segments (bottom 10%)
2. These represent true ambient ocean noise
3. Both ships AND marine sounds will have HIGH reconstruction error = anomalies
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.logmel import logmel_dataset
from anomaly_detection.preprocessing.energy import segment_energy
from Anomaly_Detection_Module.anomaly_detection.AE_config import (DATA_ROOT, AMBIENT_DIR, AUTOENCODER_PATH,
                                       SCALER_PATH, THRESHOLD_PATH,
                                       AMBIENT_PERCENTILE, HIDDEN_LAYERS, MAX_ITER)


def select_ambient_segments(segments, percentile=10):
    """
    Select TRUE ambient noise segments (quietest segments only).
    
    Args:
        segments: Array of audio segments
        percentile: Energy percentile (lower = quieter = more ambient)
                   10 = bottom 10% quietest segments
    
    Returns:
        Array of ambient-only segments
    """
    energies = segment_energy(segments)
    threshold = np.percentile(energies, percentile)
    
    # Select only the quietest segments
    mask = energies <= threshold
    ambient = segments[mask]
    
    print(f"[INFO] Energy threshold: {threshold:.6f}")
    print(f"[INFO] Selected {len(ambient)} ambient segments (bottom {percentile}%)")
    
    return ambient


def train_on_ambient(
    ambient_percentile=AMBIENT_PERCENTILE,
    hidden_layers=HIDDEN_LAYERS,
    max_iter=MAX_ITER,
    save_path=None
):
    """
    Train autoencoder on true ambient noise.
    
    After training:
    - Low reconstruction error = ambient noise (normal)
    - High reconstruction error = ANY sound activity (ships, marine life = anomaly)
    
    Args:
        ambient_percentile: Use only bottom X% energy segments as ambient
        hidden_layers: Autoencoder architecture
        max_iter: Training iterations
        save_path: Where to save models
    """
    print("\n" + "=" * 60)
    print("AUTOENCODER TRAINING ON TRUE AMBIENT NOISE")
    print("=" * 60)
    print("\nGoal: Learn ambient ocean background")
    print("      Ships AND marine life = ANOMALIES (high error)")
    
    # Load ALL data
    print("\n[INFO] Loading dataset...")
    marine, ships = load_dataset(DATA_ROOT)
    
    # Segment ALL audio (both marine and ships)
    print("[INFO] Segmenting all audio...")
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)
    
    # Combine all segments to find true ambient
    all_segments = np.vstack([marine_segments, ship_segments])
    print(f"[INFO] Total segments: {len(all_segments)}")
    
    # Load external ambient recordings from ambient dir
    external_ambient = []
    if os.path.exists(AMBIENT_DIR):
        import librosa
        from Anomaly_Detection_Module.anomaly_detection.AE_config import TARGET_SR, WINDOW_SECONDS, OVERLAP
        
        ambient_files = [f for f in os.listdir(AMBIENT_DIR) if f.endswith('.wav')]
        print(f"\n[INFO] Loading {len(ambient_files)} external ambient file(s)...")
        
        for f in ambient_files:
            filepath = os.path.join(AMBIENT_DIR, f)
            audio, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Segment
            window_size = int(TARGET_SR * WINDOW_SECONDS)
            hop_size = int(window_size * (1 - OVERLAP))
            for start in range(0, len(audio) - window_size + 1, hop_size):
                external_ambient.append(audio[start:start + window_size])
            
            print(f"  - {f}: {len(audio)/TARGET_SR:.1f}s")
        
        if external_ambient:
            external_ambient = np.array(external_ambient)
            print(f"[INFO] External ambient segments: {len(external_ambient)}")
    
    
    # Select ONLY the quietest segments as ambient
    print(f"\n[INFO] Selecting ambient noise (bottom {ambient_percentile}% energy)...")
    ambient_segments = select_ambient_segments(all_segments, percentile=ambient_percentile)
    
    if len(ambient_segments) < 50:
        print("[WARNING] Very few ambient segments! Consider increasing percentile.")
    
    # Combine with external ambient recordings
    if external_ambient is not None and len(external_ambient) > 0:
        ambient_segments = np.vstack([ambient_segments, external_ambient])
        print(f"[INFO] Combined ambient segments: {len(ambient_segments)}")
    
    # Extract features from ambient only
    print("\n[INFO] Extracting features from ambient segments...")
    ambient_features = logmel_dataset(ambient_segments)
    print(f"[INFO] Feature shape: {ambient_features.shape}")
    
    # Scale features
    print("[INFO] Fitting scaler on ambient data...")
    scaler = StandardScaler()
    X = scaler.fit_transform(ambient_features)
    
    # Train autoencoder
    print(f"\n[INFO] Training autoencoder...")
    print(f"[INFO] Architecture: {hidden_layers}")
    print(f"[INFO] Max iterations: {max_iter}")
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True
    )
    
    autoencoder.fit(X, X)
    
    # Compute training reconstruction error
    X_pred = autoencoder.predict(X)
    train_error = np.mean((X - X_pred) ** 2, axis=1)
    
    # Compute anomaly threshold from ambient data
    # Using mean + 3*std is more robust for unseen ambient recordings
    ambient_mean = np.mean(train_error)
    ambient_std = np.std(train_error)
    anomaly_threshold = ambient_mean + 3 * ambient_std
    
    print(f"\n[INFO] Training completed!")
    print(f"[INFO] Final loss: {autoencoder.loss_:.6f}")
    print(f"[INFO] Ambient reconstruction error:")
    print(f"       Mean: {ambient_mean:.4f}, Std: {ambient_std:.4f}")
    print(f"[INFO] Anomaly threshold (mean + 3*std): {anomaly_threshold:.6f}")
    
    # Test on non-ambient data to verify
    print("\n[INFO] Verifying on non-ambient segments...")
    
    # Get high-energy segments (non-ambient)
    all_energies = segment_energy(all_segments)
    high_energy_mask = all_energies > np.percentile(all_energies, 90 - ambient_percentile)
    non_ambient_segments = all_segments[high_energy_mask][:500]  # Sample
    
    if len(non_ambient_segments) > 0:
        non_ambient_features = logmel_dataset(non_ambient_segments)
        non_ambient_scaled = scaler.transform(non_ambient_features)
        non_ambient_pred = autoencoder.predict(non_ambient_scaled)
        non_ambient_error = np.mean((non_ambient_scaled - non_ambient_pred) ** 2, axis=1)
        
        print(f"[INFO] Non-ambient (anomaly) reconstruction error:")
        print(f"       Mean: {np.mean(non_ambient_error):.4f}, Std: {np.std(non_ambient_error):.4f}")
        
        # Compute separation
        separation = np.mean(non_ambient_error) / np.mean(train_error)
        print(f"\n[INFO] Separation ratio: {separation:.2f}x")
        print(f"       (Higher = better anomaly detection)")
    
    # Save models and threshold
    joblib.dump(autoencoder, AUTOENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(anomaly_threshold, THRESHOLD_PATH)
    
    print(f"\n[INFO] Models saved:")
    print(f"  - {AUTOENCODER_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {THRESHOLD_PATH}")
    
    return autoencoder, scaler


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train autoencoder on ambient noise")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Ambient energy percentile (default: 10 = quietest 10%%)")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Max training iterations (default: 300)")
    args = parser.parse_args()
    
    train_on_ambient(
        ambient_percentile=args.percentile,
        max_iter=args.iterations
    )
