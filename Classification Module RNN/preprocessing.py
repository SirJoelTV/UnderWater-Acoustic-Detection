import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import config

class AudioAugmentation:
    """Simple static methods for audio augmentation."""
    @staticmethod
    def apply_random(audio, sr):
        choice = np.random.choice(['noise', 'shift', 'pitch', 'speed', 'none'])
        if choice == 'noise':
            return audio + 0.005 * np.random.randn(len(audio))
        elif choice == 'shift':
            shift = np.random.randint(int(len(audio) * 0.2))
            return np.roll(audio, shift)
        elif choice == 'pitch':
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
        elif choice == 'speed':
            return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
        return audio

def process_audio(file_path, target_sr=config.SAMPLE_RATE, duration=config.DURATION):
    """Loads and converts audio to Mel Spectrogram DB."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Pad or truncate
    target_len = int(target_sr * duration)
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Convert to Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return torch.tensor(mel_db.T).float() # [Time, Freq]

class UnderwaterDataset(Dataset):
    def __init__(self, data_dir, categories=['ships', 'marine_life'], training=True):
        self.samples = []
        self.training = training
        
        # Load file list
        print("Scanning dataset files...")
        for cat in categories:
            path = os.path.join(data_dir, cat)
            if not os.path.exists(path): continue
            for subcat in os.listdir(path):
                sub_path = os.path.join(path, subcat)
                if not os.path.isdir(sub_path): continue
                for f in os.listdir(sub_path):
                    if f.endswith('.wav'):
                        self.samples.append({
                            'path': os.path.join(sub_path, f),
                            'label': f"{cat}_{subcat}"
                        })

        # Encode labels
        self.labels = [s['label'] for s in self.samples]
        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]['path']
        label = self.encoded_labels[idx]
        
        # Load raw audio to apply augmentation before processing
        audio, sr = librosa.load(path, sr=config.SAMPLE_RATE)
        target_len = int(config.SAMPLE_RATE * config.DURATION)
        
        # Augmentation
        if self.training and np.random.rand() < 0.5:
            audio = AudioAugmentation.apply_random(audio, sr)

        # Pad/Truncate logic repeated here to handle augmented audio
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        # Convert
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=config.N_MELS, hop_length=config.HOP_LENGTH)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        
        return torch.tensor(mel_db.T).float(), torch.tensor(label).long()