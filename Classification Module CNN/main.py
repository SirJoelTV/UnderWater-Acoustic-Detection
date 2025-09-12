import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split

# Config defaults
SAMPLE_RATE = 32000
DURATION = 15
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PATH = "best_model.pth"
DATA_DIR = r'D:\Main Project\UnderWater-Acoustic-Detection\data'


class UnderwaterAcousticDataset(Dataset):
    def __init__(self, data_dir, categories=['ships', 'marine_life'], sample_rate=SAMPLE_RATE, duration=DURATION):
        self.data_dir = data_dir
        self.categories = categories
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_length = int(sample_rate * duration)

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
                subcat_path = os.path.join(category_path, subcategory)
                if not os.path.isdir(subcat_path):
                    continue
                for filename in os.listdir(subcat_path):
                    if filename.lower().endswith('.wav'):
                        samples.append({
                            'file_path': os.path.join(subcat_path, filename),
                            'category': category,
                            'subcategory': subcategory,
                            'label': f"{category}_{subcategory}"
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = librosa.load(sample['file_path'], sr=self.sample_rate)
        if len(audio) > self.audio_length:
            audio = audio[:self.audio_length]
        else:
            audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)

        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()

        label_encoded = self.label_encoder.transform([sample['label']])[0]
        label_tensor = torch.tensor(label_encoded).long()

        return mel_spec_db, label_tensor


def compute_flattened_size(sample_rate, duration):
    audio = np.zeros(int(sample_rate * duration))
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
    mel_tensor = torch.tensor(librosa.power_to_db(mel, ref=np.max)).unsqueeze(0).unsqueeze(0).float()

    class CNNBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            return x

    model = CNNBase()
    with torch.no_grad():
        out = model(mel_tensor)
    flattened_size = out.view(out.size(0), -1).size(1)
    return flattened_size


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, flattened_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, 100 * correct / total


def predict_audio(model, label_encoder, audio_path, sample_rate=32000, duration=15.0, device='cpu'):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    desired_len = int(sample_rate * duration)
    if len(audio) < desired_len:
        audio = np.pad(audio, (0, desired_len - len(audio)), mode='constant')
    else:
        audio = audio[:desired_len]
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)
    mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(mel_spec_db)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        label = label_encoder.inverse_transform([idx.item()])[0]
    return label, conf.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        print("\nChoose mode: \n1 - Train model\n2 - Predict audio file\n3 - Exit")
        choice = input("Enter 1, 2 or 3: ").strip()

        if choice == '1':
            data_dir = input(f"Enter dataset root folder (default: {DATA_DIR}): ").strip() or DATA_DIR
            if not os.path.exists(data_dir):
                print(f"Error: '{data_dir}' does not exist.")
                continue
            dataset = UnderwaterAcousticDataset(data_dir)
            print(f"Loaded {len(dataset)} samples with classes: {list(dataset.label_encoder.classes_)}")
            num_classes = len(dataset.label_encoder.classes_)

            flattened_size = compute_flattened_size(SAMPLE_RATE, DURATION)

            train_len = int(0.8 * len(dataset))
            val_len = len(dataset) - train_len
            train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            model = SimpleCNN(num_classes, flattened_size).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())

            best_val_acc = 0
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"Model saved with val accuracy: {best_val_acc:.2f}%")
            print("Training complete!")

        elif choice == '2':
            audio_file = input("Enter path to audio file for prediction: ").strip()
            if not os.path.isfile(audio_file):
                print(f"Error: '{audio_file}' not found.")
                continue
            if not os.path.isfile(MODEL_PATH):
                print(f"Model '{MODEL_PATH}' not found. Train the model first.")
                continue
            dataset = UnderwaterAcousticDataset(DATA_DIR)
            flattened_size = compute_flattened_size(SAMPLE_RATE, DURATION)
            model = SimpleCNN(len(dataset.label_encoder.classes_), flattened_size).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

            label_encoder = dataset.label_encoder
            pred_label, confidence = predict_audio(model, label_encoder, audio_file, device=device)
            print(f"Prediction: {pred_label} with confidence {confidence:.3f}")

        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")


if __name__ == '__main__':
    main()
