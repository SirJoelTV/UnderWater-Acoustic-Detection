import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config

# Global statistics tracker (computed from training data)
GLOBAL_MEAN = None
GLOBAL_STD = None


class AudioAugmentation:
    """Realistic audio augmentation for underwater acoustics."""

    @staticmethod
    def apply_random(audio, sr):
        """Apply random augmentation matching real acoustic variations."""
        choice = np.random.choice(
            ["noise", "shift", "pitch", "speed", "none"], p=[0.25, 0.25, 0.2, 0.2, 0.1]
        )

        if choice == "noise":
            noise_level = np.random.uniform(0.003, 0.008)
            return np.clip(audio + noise_level * np.random.randn(len(audio)), -1.0, 1.0)
        elif choice == "shift":
            shift = np.random.randint(int(len(audio) * 0.1))
            return np.roll(audio, shift)
        elif choice == "pitch":
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
        elif choice == "speed":
            return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.95, 1.05))
        return audio


def process_audio_to_spectrogram(audio, sr, target_sr=config.SAMPLE_RATE, duration=config.DURATION, normalize=True):
    """Converts audio to Mel spectrogram (dB). Uses global normalization if available."""
    target_len = int(target_sr * duration)
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP_LENGTH)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if normalize:
        global GLOBAL_MEAN, GLOBAL_STD
        if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
            mel_db = (mel_db - GLOBAL_MEAN) / (GLOBAL_STD + 1e-9)
        else:
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    return mel_db


def compute_global_statistics(data_dir, categories=["ships", "marine_life"], sample_size=500):
    """Compute mean/std from a sample of training spectrograms and store in globals."""
    global GLOBAL_MEAN, GLOBAL_STD
    all_specs = []
    file_count = 0

    print(f"Computing global statistics from training data (up to {sample_size} files)...")

    for cat in categories:
        path = os.path.join(data_dir, cat)
        if not os.path.exists(path):
            continue
        for subcat in os.listdir(path):
            sub_path = os.path.join(path, subcat)
            if not os.path.isdir(sub_path):
                continue
            for f in os.listdir(sub_path):
                if not f.endswith(".wav"):
                    continue
                if file_count >= sample_size:
                    break
                try:
                    audio, sr = librosa.load(os.path.join(sub_path, f), sr=config.SAMPLE_RATE)
                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP_LENGTH)
                    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
                    all_specs.append(mel_db.flatten())
                    file_count += 1
                    if file_count % 50 == 0:
                        print(f"  Processed {file_count} files...")
                except Exception as e:
                    print(f"  Error processing {f}: {e}")

    if all_specs:
        all_specs = np.concatenate(all_specs)
        GLOBAL_MEAN = np.mean(all_specs)
        GLOBAL_STD = np.std(all_specs)
        print(f"Global statistics computed: mean={GLOBAL_MEAN:.4f}, std={GLOBAL_STD:.4f}")
    else:
        print("Warning: Could not compute global statistics. Falling back to per-sample normalization.")
        GLOBAL_MEAN = None
        GLOBAL_STD = None


def save_global_statistics(filepath="global_stats.npz"):
    global GLOBAL_MEAN, GLOBAL_STD
    if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
        np.savez(filepath, mean=GLOBAL_MEAN, std=GLOBAL_STD)
        print(f"Saved global statistics to {filepath}")


def load_global_statistics(filepath="global_stats.npz"):
    global GLOBAL_MEAN, GLOBAL_STD
    try:
        stats = np.load(filepath)
        GLOBAL_MEAN = float(stats["mean"])
        GLOBAL_STD = float(stats["std"])
        print(f"Loaded global statistics: mean={GLOBAL_MEAN:.4f}, std={GLOBAL_STD:.4f}")
        return True
    except Exception:
        print(f"Warning: Could not load global statistics from {filepath}. Using per-sample normalization.")
        GLOBAL_MEAN = None
        GLOBAL_STD = None
        return False


class UnderwaterDataset(Dataset):
    def __init__(self, data_dir, categories=["ships", "marine_life"], training=True, encoder=None, sample_indices=None):
        self.samples = []
        self.training = training
        self.encoder = encoder

        print("Scanning dataset files...")
        all_samples = []
        for cat in categories:
            path = os.path.join(data_dir, cat)
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist")
                continue
            for subcat in os.listdir(path):
                sub_path = os.path.join(path, subcat)
                if not os.path.isdir(sub_path):
                    continue
                for f in os.listdir(sub_path):
                    if f.endswith('.wav'):
                        all_samples.append({
                            'path': os.path.join(sub_path, f),
                            'label': f"{cat}_{subcat}"
                        })

        if sample_indices is not None:
            self.samples = [all_samples[i] for i in sample_indices]
        else:
            self.samples = all_samples

        self.labels = [s['label'] for s in self.samples]
        if encoder is None:
            self.encoder = LabelEncoder()
            self.encoded_labels = self.encoder.fit_transform(self.labels)
        else:
            self.encoder = encoder
            self.encoded_labels = self.encoder.transform(self.labels)

        print(f"Loaded {len(self.samples)} samples (training={training})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]['path']
        label = self.encoded_labels[idx]

        try:
            audio, sr = librosa.load(path, sr=config.SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio = np.zeros(int(config.SAMPLE_RATE * config.DURATION))
            sr = config.SAMPLE_RATE

        if self.training and np.random.rand() < 0.9:
            audio = AudioAugmentation.apply_random(audio, sr)

        mel_db = process_audio_to_spectrogram(audio, sr)
        return torch.tensor(mel_db.T).float(), torch.tensor(label).long()


def create_stratified_split(data_dir, test_size=0.2, val_size=0.2, categories=["ships", "marine_life"]):
    """Create stratified train/val split and compute global stats from training set."""
    full_dataset = UnderwaterDataset(data_dir, categories=categories, training=False)
    labels = [s['label'] for s in full_dataset.samples]
    label_indices = np.arange(len(full_dataset))

    train_indices, val_indices = train_test_split(label_indices, test_size=val_size, stratify=labels, random_state=42)

    print(f"Stratified split: {len(train_indices)} train, {len(val_indices)} val")

    train_dataset = UnderwaterDataset(data_dir, categories=categories, training=True, encoder=full_dataset.encoder, sample_indices=train_indices)

    # Compute global statistics from the training partition
    compute_global_statistics(data_dir, categories, sample_size=len(train_indices))

    val_dataset = UnderwaterDataset(data_dir, categories=categories, training=False, encoder=full_dataset.encoder, sample_indices=val_indices)

    return train_dataset, val_dataset, full_dataset.encoder.classes_