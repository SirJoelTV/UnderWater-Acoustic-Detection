import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import config as config
from model import SimpleCNN
from preprocessing import get_train_val_test_datasets

def get_time_steps():
    return (config.SAMPLE_RATE * config.CHUNK_DURATION) // config.HOP_LENGTH + 1

_, val_ds, _, encoder = get_train_val_test_datasets(config.DATA_DIR)
classes    = list(encoder.classes_)
time_steps = get_time_steps()

model = SimpleCNN(num_classes=len(classes), n_mels=config.N_MELS, time_steps=time_steps)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()

all_preds, all_labels = [], []
loader = DataLoader(val_ds, batch_size=32, shuffle=False)

with torch.no_grad():
    for X, y in loader:
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)
        preds = model(X).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=classes))