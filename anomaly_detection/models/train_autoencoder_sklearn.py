import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.mfcc import mfcc_dataset
from anomaly_detection.preprocessing.energy import select_low_energy_segments



def train():
    # Load data
    marine, ships = load_dataset("data")

    # Segment marine audio
    marine_segments = segment_dataset(marine)

    # Select low-energy (ambient) segments
    ambient_segments = select_low_energy_segments(
    marine_segments,
    percentile=30
)

    print(f"[INFO] Ambient segments selected: {len(ambient_segments)}")

    # Extract MFCC features from ambient-only data
    marine_features = mfcc_dataset(ambient_segments)


    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(marine_features)

    # Autoencoder model
    autoencoder = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16, 32, 64),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42,
    verbose=True
)


    autoencoder.fit(X, X)

    # Save trained model
    joblib.dump(autoencoder, "autoencoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Autoencoder and scaler saved successfully.")


if __name__ == "__main__":
    train()
