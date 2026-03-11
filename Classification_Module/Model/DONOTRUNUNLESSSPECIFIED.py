#Dont run this code pls. It's a last stage code to get model perf.

import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import CNN_config as config
from CNN_model import SimpleCNN
from CNN_preprocessing import get_train_val_test_datasets

def get_time_steps():
    return (config.SAMPLE_RATE * config.CHUNK_DURATION) // config.HOP_LENGTH + 1

# Load datasets (we only need the test split)
_, _, test_ds, encoder = get_train_val_test_datasets(config.DATA_DIR)
classes    = list(encoder.classes_)
time_steps = get_time_steps()

# Load the best saved model
model = SimpleCNN(num_classes=len(classes), n_mels=config.N_MELS, time_steps=time_steps)
model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()

# Run predictions on test set
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
all_preds, all_labels = [], []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)
        preds = model(X).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print("\n=== FINAL TEST RESULTS (never seen during training) ===\n")
print(classification_report(all_labels, all_preds, target_names=classes))