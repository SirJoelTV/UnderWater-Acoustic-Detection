import os
import torch
import torch.nn as nn
import librosa
import numpy as np

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Config ===
SAMPLE_RATE = 32000
DURATION = 15
MODEL_PATH = "best_gpu_lite_rnn_model.pth"

# === All labels from your folder structure ===
LABELS = [
    'ships_Cargo',
    'ships_Passengership',
    'ships_Tanker',
    'ships_Tug',
    'marine_life_Beluga, White Whale',
    'marine_life_Bottlenose Dolphin',
    'marine_life_Common Dolphin',
    'marine_life_dolphin',
    'marine_life_Humpback Whale',
    'marine_life_Killer whale',
    'marine_life_spermwhale',
    'marine_life_spinner dolphin',
    'marine_life_striped dolphin',
    'marine_life_white sided dolphin'
]
label_to_index = {label: idx for idx, label in enumerate(LABELS)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

# === Model definition must match training ===
class EnhancedBiLSTM(nn.Module):
    def __init__(self, num_classes, input_size=64, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def attention(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * w, dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

def predict_audio(model, audio_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    if not os.path.isfile(audio_path):
        print(f"File '{audio_path}' does not exist.")
        return None, None

    audio, sr = librosa.load(audio_path, sr=sample_rate)
    desired_len = int(sample_rate * duration)
    if len(audio) < desired_len:
        audio = np.pad(audio, (0, desired_len - len(audio)), mode='constant')
    else:
        audio = audio[:desired_len]

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    mel_db = torch.tensor(mel_db.T).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(mel_db)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        predicted_label = index_to_label[idx.item()]
        confidence = conf.item()
    return predicted_label, confidence

def main():
    num_classes = len(LABELS)
    model = EnhancedBiLSTM(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded. Enter file paths to classify (type 'exit' to quit):")
    while True:
        filepath = input("Enter path to audio file: ").strip()
        if filepath.lower() == 'exit':
            print("Exiting.")
            break
        label, conf = predict_audio(model, filepath)
        if label is not None:
            print(f"Prediction: {label} (Confidence: {conf:.3f})")

if __name__ == "__main__":
    main()
