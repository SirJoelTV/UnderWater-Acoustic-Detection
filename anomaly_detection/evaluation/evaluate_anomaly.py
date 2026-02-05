
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.mfcc import mfcc_dataset


def reconstruction_error(model, X):
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def main():
    # Load trained model and scaler
    autoencoder = joblib.load("autoencoder.pkl")
    scaler = joblib.load("scaler.pkl")

    # Load dataset
    marine, ships = load_dataset("data")

    # Feature extraction
    marine_features = mfcc_dataset(segment_dataset(marine))
    ship_features = mfcc_dataset(segment_dataset(ships))

    marine_features = scaler.transform(marine_features)
    ship_features = scaler.transform(ship_features)

    # Reconstruction errors
    marine_error = reconstruction_error(autoencoder, marine_features)
    ship_error = reconstruction_error(autoencoder, ship_features)

       # Threshold from normal data
    threshold = np.percentile(marine_error, 80)
    print(f"Anomaly threshold: {threshold:.6f}")
    # ---------------- BALANCED EVALUATION ----------------

    # Number of normal samples
    n_marine = len(marine_error)

    # Randomly sample ship errors to match marine count
    rng = np.random.default_rng(seed=42)
    ship_idx = rng.choice(len(ship_error), size=n_marine, replace=False)

    balanced_ship_error = ship_error[ship_idx]

    # Create balanced labels
    y_true = np.concatenate([
        np.zeros(n_marine),          # normal
        np.ones(n_marine)            # anomaly
    ])

    y_pred = np.concatenate([
        marine_error > threshold,
        balanced_ship_error > threshold
    ])


     

    # Labels
   # y_true = np.concatenate([
        #np.zeros(len(marine_error)),  # normal
        #np.ones(len(ship_error))      # anomaly
    #])

    #y_pred = np.concatenate([
        #marine_error > threshold,
        #ship_error > threshold
    #])
    

    # Evaluation
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=["Normal", "Anomaly"]
    ))


if __name__ == "__main__":
    main()

