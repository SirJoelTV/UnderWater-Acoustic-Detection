import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.logmel import logmel_dataset


def main():
    # Load dataset
    marine, ships = load_dataset("data")

    # Segment audio (use same WINDOW_SECONDS from config)
    marine_segments = segment_dataset(marine)
    ship_segments = segment_dataset(ships)

    # Extract log-mel features
    marine_features = logmel_dataset(marine_segments)
    ship_features = logmel_dataset(ship_segments)

    # Scale features (FIT ONLY ON NORMAL DATA)
    scaler = StandardScaler()
    marine_features = scaler.fit_transform(marine_features)
    ship_features = scaler.transform(ship_features)

    # Train Isolation Forest on NORMAL data only
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42
    )
    iso.fit(marine_features)

    # Save model and scaler
    joblib.dump(iso, "isolation_forest.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Isolation Forest trained and saved successfully.")


if __name__ == "__main__":
    main()
