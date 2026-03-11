from collections import Counter
from preprocessing import get_train_val_test_datasets
import config

train_ds, _, _, encoder = get_train_val_test_datasets(config.DATA_DIR)

counts = Counter(train_ds.encoded_labels)
classes = list(encoder.classes_)

ship_total = 0
marine_total = 0

print("Chunks per class:")
for i, cls in enumerate(classes):
    count = counts.get(i, 0)
    print(f"  {cls:<50}: {count}")
    if cls.startswith("Vessels"):
        ship_total += count
    else:
        marine_total += count

print(f"\nTotal ship chunks   : {ship_total}")
print(f"Total marine chunks : {marine_total}")

'''I already suspect what this will show:
```
Marine life: 33 classes × ~40 files × 1 chunk  = ~1300 chunks
Ships:        4 classes × 10 chunks per file   = varies'''
