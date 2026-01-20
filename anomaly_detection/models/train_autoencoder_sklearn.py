import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.mfcc import mfcc_dataset


def train():
    # Load data
    marine, _ = load_dataset("data")

    marine_segments = segment_dataset(marine)
    marine_features = mfcc_dataset(marine_segments)

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
