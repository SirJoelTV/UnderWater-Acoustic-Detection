import json
import numpy as np
import torch

import config
from model import SimpleCNN
from preprocessing import load_audio, split_into_chunks, audio_to_melspectrogram


def get_time_steps():
    return (config.SAMPLE_RATE * config.CHUNK_DURATION) // config.HOP_LENGTH + 1


def predict(audio_path):
    """
    Predict the class of an audio file.

    For long files (e.g. 3-minute ship recording):
        - Split into 3s chunks
        - Predict each chunk
        - Take the majority vote across all chunks
        → Much more reliable than predicting from one 3s snippet

    For short files (e.g. 1s dolphin click):
        - Pad to 3s
        - Single prediction
    """
    # Load model and classes
    with open(config.CLASSES_PATH) as f:
        classes = json.load(f)

    time_steps = get_time_steps()
    model      = SimpleCNN(num_classes=len(classes), n_mels=config.N_MELS, time_steps=time_steps)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    # Load and chunk audio
    audio, sr = load_audio(audio_path)
    chunk_len  = config.SAMPLE_RATE * config.CHUNK_DURATION
    chunks     = split_into_chunks(audio, chunk_len)
    print(f"Audio split into {len(chunks)} chunk(s) of {config.CHUNK_DURATION}s each")

    # Predict each chunk
    all_probs = []
    with torch.no_grad():
        for chunk in chunks:
            mel  = audio_to_melspectrogram(chunk, config.SAMPLE_RATE)
            X    = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
            out  = model(X)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    # Average probabilities across all chunks (soft voting)
    avg_probs     = np.mean(all_probs, axis=0)
    predicted_idx = np.argmax(avg_probs)

    predicted_class = classes[predicted_idx]
    confidence      = avg_probs[predicted_idx] * 100

    print(f"\nPrediction : {predicted_class}")
    print(f"Confidence : {confidence:.1f}%")
    print(f"\nAll class probabilities:")
    for cls, prob in sorted(zip(classes, avg_probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {cls:<45}: {prob*100:>5.1f}%  {bar}")


# Ask user for audio path
path = input("Enter path to audio file: ").strip().strip('"')
predict(path)
