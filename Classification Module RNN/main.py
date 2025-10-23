import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split

# === GPU detection and device assignment: ONLY ONCE per Python session ===
if not hasattr(sys, "_gpu_printed"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected. Using CPU.")
    sys._gpu_printed = True
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 32000
DURATION = 15
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
MODEL_PATH = "best_gpu_lite_rnn_model.pth"
DATA_DIR = r'D:\Main Project\UnderWater-Acoustic-Detection\data'


class AudioAugmentation:
    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    @staticmethod
    def time_shift(audio, shift_max=0.2):
        shift = np.random.randint(int(len(audio) * shift_max))
        direction = np.random.randint(0, 2)
        return np.roll(audio, shift if direction == 1 else -shift)
    @staticmethod
    def pitch_shift(audio, sr, n_steps=2):
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    @staticmethod
    def speed_change(audio, speed_factor=1.0):
        return librosa.effects.time_stretch(audio, rate=speed_factor)


class UnderwaterAcousticDataset(Dataset):
    def __init__(self, data_dir, categories=['ships', 'marine_life'], sample_rate=SAMPLE_RATE,
                 duration=DURATION, training=True):
        self.data_dir = data_dir
        self.categories = categories
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_length = int(sample_rate * duration)
        self.training = training
        self.samples = self._load_dataset()
        self.labels = [sample['label'] for sample in self.samples]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
    def _load_dataset(self):
        samples = []
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            if not os.path.exists(category_path):
                continue
            for subcategory in os.listdir(category_path):
                subcat_dir = os.path.join(category_path, subcategory)
                if not os.path.isdir(subcat_dir):
                    continue
                for filename in os.listdir(subcat_dir):
                    if filename.lower().endswith('.wav'):
                        samples.append({
                            'file': os.path.join(subcat_dir, filename),
                            'label': f"{category}_{subcategory}"
                        })
        return samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = librosa.load(sample['file'], sr=self.sample_rate)
        if self.training and np.random.rand() < 0.5:
            aug = np.random.choice(['noise', 'shift', 'pitch', 'speed'])
            if aug == 'noise':
                audio = AudioAugmentation.add_noise(audio)
            elif aug == 'shift':
                audio = AudioAugmentation.time_shift(audio)
            elif aug == 'pitch':
                audio = AudioAugmentation.pitch_shift(audio, sr, np.random.randint(-2, 3))
            elif aug == 'speed':
                audio = AudioAugmentation.speed_change(audio, np.random.uniform(0.9, 1.1))
        if len(audio) > self.audio_length:
            audio = audio[:self.audio_length]
        else:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=512)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = torch.tensor(mel_db.T).float()
        label = torch.tensor(self.label_encoder.transform([sample['label']])[0]).long()
        return mel_db, label


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


def train_model(model, train_dl, val_dl, criterion, optimizer, scheduler, epochs=EPOCHS):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_dl, criterion)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Acc={train_acc:.2f}% Val Acc={val_acc:.2f}% "
              f"| Val Loss={val_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved (val acc: {best_acc:.2f}%)")

def evaluate(model, dataloader, criterion):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += criterion(out, y).item() * X.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss / total, 100 * correct / total

def main():
    print("\n🚀 Underwater Acoustic Detection (Memory-optimized for 4GB GPU)")
    print("="*60)
    dataset = UnderwaterAcousticDataset(DATA_DIR, training=True)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)
    num_classes = len(dataset.label_encoder.classes_)
    model = EnhancedBiLSTM(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print(f"\n[Training on device: {device}] Training on {len(train_loader.dataset)} samples...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS)

if __name__ == "__main__":
    main()
