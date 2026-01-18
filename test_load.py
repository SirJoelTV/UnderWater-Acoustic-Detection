print("### TEST_LOAD.PY IS RUNNING ###")

from anomaly_detection.preprocessing.load_audio import load_dataset

print("Starting data loading test...")

marine, ships = load_dataset("data")

print("Marine samples:", len(marine))
print("Ship samples:", len(ships))

if len(marine) > 0:
    print("Example marine length:", len(marine[0]))

if len(ships) > 0:
    print("Example ship length:", len(ships[0]))
