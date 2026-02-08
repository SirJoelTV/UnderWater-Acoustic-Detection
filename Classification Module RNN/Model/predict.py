import torch
import json
import os
import librosa
import numpy as np
import config
from model import EnhancedBiLSTM
from preprocessing import process_audio_to_spectrogram, load_global_statistics

def predict_single_file():
    device = config.get_device()
    
    # Load global normalization statistics
    load_global_statistics(config.GLOBAL_STATS_PATH)

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

        # Load and Process Audio
        try:
            audio, sr = librosa.load(path, sr=config.SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue
        
        # Convert to spectrogram
        mel_db = process_audio_to_spectrogram(audio, sr)
        features = torch.tensor(mel_db.T).float()  # [Time, Freq]
        features = features.unsqueeze(0).to(device) # Add batch dim
        
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
        result_label = labels[idx.item()]
        confidence = conf.item()
        
        # Show top 3 predictions
        top3_probs, top3_indices = torch.topk(probs, min(3, len(labels)), dim=1)
        print(f"\nResult: {result_label} (Confidence: {confidence:.2%})")
        print("Top predictions:")
        for i, (prob, label_idx) in enumerate(zip(top3_probs[0], top3_indices[0])):
            print(f"  {i+1}. {labels[label_idx.item()]} - {prob.item():.2%}")
        
        if confidence < 0.5:
            print("⚠️  Low confidence - results may be unreliable!")
        print()

def predict_folder(folder_path):
    """Batch predict all WAV files in a folder."""
    device = config.get_device()
    
    # Load global normalization statistics
    load_global_statistics(config.GLOBAL_STATS_PATH)

    # Load model and classes
    if not os.path.exists(config.CLASSES_PATH):
        print("Error: classes.json not found. Run train.py first.")
        return
    with open(config.CLASSES_PATH, 'r') as f:
        labels = json.load(f)

    model = EnhancedBiLSTM(num_classes=len(labels)).to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found. Run train.py first.")
        return
    model.eval()

    print(f"\n--- Batch Prediction Mode ---")
    print(f"Processing files in: {folder_path}\n")
    
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    if not wav_files:
        print("No .wav files found!")
        return
    
    results = []
    for filename in wav_files:
        filepath = os.path.join(folder_path, filename)
        
        try:
            audio, sr = librosa.load(filepath, sr=config.SAMPLE_RATE)
        except Exception as e:
            print(f"❌ {filename}: Error loading ({e})")
            continue
        
        mel_db = process_audio_to_spectrogram(audio, sr)
        features = torch.tensor(mel_db.T).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
        
        result_label = labels[idx.item()]
        confidence = conf.item()
        results.append((filename, result_label, confidence))
        
        status = "✓" if confidence > 0.7 else "⚠"
        print(f"{status} {filename}: {result_label} ({confidence:.1%})")
    
    print(f"\n--- Summary ---")
    print(f"Processed: {len(results)} files")
    high_conf = sum(1 for _, _, c in results if c > 0.7)
    print(f"High confidence (>70%): {high_conf}/{len(results)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        if len(sys.argv) > 2:
            predict_folder(sys.argv[2])
        else:
            print("Usage: python predict.py batch <folder_path>")
    else:
        predict_single_file()
