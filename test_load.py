from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset

marine, ships = load_dataset("data")

marine_segments = segment_dataset(marine)
ship_segments = segment_dataset(ships)

print("Marine segments shape:", marine_segments.shape)
print("Ship segments shape:", ship_segments.shape)
