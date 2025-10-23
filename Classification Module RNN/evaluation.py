import torch
import torch.nn as nn
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 32000
DURATION = 15
BATCH_SIZE = 8
MODEL_PATH = "best_gpu_lite_rnn_model.pth"
DATA_DIR = r'D:\Main Project\UnderWater-Acoustic-Detection\data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnderwaterAcousticDataset(Dataset):
    def __init__(self, data_dir, categories=['ships', 'marine_life'],
                 sample_rate=SAMPLE_RATE, duration=DURATION, training=False):
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
            if not os.path.exists(category_path): continue
            for subcategory in os.listdir(category_path):
                subcat_dir = os.path.join(category_path, subcategory)
                if not os.path.isdir(subcat_dir): continue
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

def evaluate_and_confusion(model, dataloader, criterion, label_encoder):
    model.eval()
    loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss += criterion(out, y).item() * X.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = loss / total
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Classification report (precision, recall, f1)
    report = classification_report(
        all_labels, all_preds, target_names=label_encoder.classes_, digits=3
    )
    print("\nClassification Report (precision, recall, F1-score):")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    print(f"Using device: {DEVICE}")
    dataset = UnderwaterAcousticDataset(DATA_DIR, training=False)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    num_classes = len(dataset.label_encoder.classes_)
    model = EnhancedBiLSTM(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()
    evaluate_and_confusion(model, val_loader, criterion, dataset.label_encoder)

if __name__ == "__main__":
    main()
