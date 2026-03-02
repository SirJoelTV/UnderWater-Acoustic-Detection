from anomaly_detection.preprocessing.load_audio import load_dataset
from anomaly_detection.preprocessing.segment import segment_dataset
from anomaly_detection.preprocessing.energy import select_low_energy_segments

marine, _ = load_dataset("data")
segments = segment_dataset(marine)

ambient = select_low_energy_segments(segments, percentile=30)

print("Total segments:", len(segments))
print("Ambient segments:", len(ambient))
