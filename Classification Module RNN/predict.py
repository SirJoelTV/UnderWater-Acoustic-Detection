import torch
import json
import os
import config
from model import EnhancedBiLSTM
from preprocessing import process_audio

def predict_single_file():
    device = config.get_device()

    # 1. Load Classes
    if not os.path.exists(config.CLASSES_PATH):
        print("Error: classes.json not found. Run train.py first.")
        return
    with open(config.CLASSES_PATH, 'r') as f:
        labels = json.load(f)

    # 2. Load Model
    model = EnhancedBiLSTM(num_classes=len(labels)).to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found. Run train.py first.")
        return
    model.eval()

    # 3. Inference Loop
    print("\n--- Audio Classifier ---")
    while True:
        path = input("Enter audio file path (or 'exit'): ").strip().strip('"') # Strip quotes if dragged in
        if path.lower() == 'exit': break
        
        if not os.path.exists(path):
            print("File not found.")
            continue

        # Process and Predict
        features = process_audio(path)
        if features is None: continue

        features = features.unsqueeze(0).to(device) # Add batch dim
        
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
        print(f"Result: {labels[idx.item()]} (Confidence: {conf.item():.2f})")

if __name__ == "__main__":
    predict_single_file()