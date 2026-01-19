from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.features.mfcc import mfcc_dataset

marine, ships = load_dataset("data")

marine_segments = segment_dataset(marine)
ship_segments = segment_dataset(ships)

marine_features = mfcc_dataset(marine_segments)
ship_features = mfcc_dataset(ship_segments)

print("Marine MFCC shape:", marine_features.shape)
print("Ship MFCC shape:", ship_features.shape)
print("Feature dimension:", marine_features.shape[1])
