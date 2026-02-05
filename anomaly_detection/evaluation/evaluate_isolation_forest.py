import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.logmel import logmel_dataset


def main():
    # Load model and scaler
    iso = joblib.load("isolation_forest.pkl")
    scaler = joblib.load("scaler.pkl")

    # Load data
    marine, ships = load_dataset("data")

    # Segment
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)

    # Features
    marine_features = logmel_dataset(marine_segments)
    ship_features = logmel_dataset(ship_segments)

    marine_features = scaler.transform(marine_features)
    ship_features = scaler.transform(ship_features)

    # Predict (-1 = anomaly, 1 = normal)
    marine_pred = iso.predict(marine_features)
    ship_pred = iso.predict(ship_features)

    # Convert to labels
    y_true = np.concatenate([
        np.zeros(len(marine_pred)),
        np.ones(len(ship_pred))
    ])

    y_pred = np.concatenate([
        marine_pred == -1,
        ship_pred == -1
    ])

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, target_names=["Normal", "Anomaly"]
    ))


if __name__ == "__main__":
    main()
