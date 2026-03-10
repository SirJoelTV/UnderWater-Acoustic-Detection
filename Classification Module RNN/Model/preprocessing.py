import os
import random
import numpy as np
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

import config


def load_audio(path):
    audio, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), sr


def split_into_chunks(audio, chunk_len):
    chunks = []
    start  = 0
    while start < len(audio):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)
        start += chunk_len
    return chunks


def augment_audio(audio, sr):
    choice = np.random.choice(["noise", "shift", "pitch", "speed", "none"],
                               p=[0.25, 0.25, 0.2, 0.2, 0.1])
    if choice == "noise":
        audio = np.clip(audio + np.random.randn(len(audio)) * 0.005, -1.0, 1.0)
    elif choice == "shift":
        audio = np.roll(audio, np.random.randint(0, int(len(audio) * 0.1)))
    elif choice == "pitch":
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
    elif choice == "speed":
        audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
    return audio


def audio_to_melspectrogram(audio, sr):
    mel    = librosa.feature.melspectrogram(
                y=audio, sr=sr,
                n_mels=config.N_MELS,
                hop_length=config.HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return mel_db[np.newaxis, :, :]  # (1, N_MELS, time_steps)


def load_valid_files(data_dir):
    metadata_path = os.path.join(data_dir, config.METADATA_FILE)
    df    = pd.read_csv(metadata_path)
    valid = df[df['duration_second'] >= 1.0]['file_name'].tolist()
    print(f"Metadata: {len(df)} total, {len(valid)} valid (>=1s), "
          f"{len(df)-len(valid)} skipped (too short)")
    return set(valid)


def collect_files(data_dir):
    valid_files = load_valid_files(data_dir)
    file_paths, labels = [], []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder in config.SKIP_FOLDERS:
            print(f"  Skipping folder: {folder}")
            continue

        for subclass in os.listdir(folder_path):
            subclass_path = os.path.join(folder_path, subclass)
            if not os.path.isdir(subclass_path):
                continue
            if subclass in config.SKIP_CLASSES:
                print(f"  Skipping class: {subclass}")
                continue

            label = f"{folder}_{subclass}"

            for fname in os.listdir(subclass_path):
                if not fname.endswith(".wav"):
                    continue
                if fname not in valid_files:
                    continue
                file_paths.append(os.path.join(subclass_path, fname))
                labels.append(label)

    print(f"\nFound {len(file_paths)} files across {len(set(labels))} classes.")
    return file_paths, labels


class UnderwaterDataset(Dataset):
    def __init__(self, file_paths, labels, encoder, is_training=False):
        self.encoder     = encoder
        self.is_training = is_training
        self.chunk_len   = config.SAMPLE_RATE * config.CHUNK_DURATION

        split_name = 'train' if is_training else 'val/test'
        print(f"  Chunking {split_name} files...")

        self.chunks         = []
        self.encoded_labels = []

        for path, label in zip(file_paths, labels):
            try:
                audio, sr = load_audio(path)
                chunks    = split_into_chunks(audio, self.chunk_len)

                # Cap chunks per file so one long recording doesn't dominate
                if len(chunks) > config.MAX_CHUNKS_PER_FILE:
                    indices = np.linspace(0, len(chunks)-1,
                                          config.MAX_CHUNKS_PER_FILE, dtype=int)
                    chunks  = [chunks[i] for i in indices]

                for chunk in chunks:
                    self.chunks.append(chunk)
                    self.encoded_labels.append(encoder.transform([label])[0])

            except Exception as e:
                print(f"  Error loading {path}: {e}")

        self.encoded_labels = np.array(self.encoded_labels)

        # Balance classes so ships don't get overwhelmed by 28 marine life classes
        if is_training:
            self._balance_classes()

        print(f"  → {len(self.chunks)} chunks from {len(file_paths)} files")

    def _balance_classes(self):
        """Keep at most MAX_CHUNKS_PER_CLASS chunks per class."""
        class_indices = defaultdict(list)
        for i, lbl in enumerate(self.encoded_labels):
            class_indices[lbl].append(i)

        kept = []
        for lbl, indices in class_indices.items():
            if len(indices) > config.MAX_CHUNKS_PER_CLASS:
                indices = random.sample(indices, config.MAX_CHUNKS_PER_CLASS)
            kept.extend(indices)

        self.chunks         = [self.chunks[i] for i in kept]
        self.encoded_labels = np.array([self.encoded_labels[i] for i in kept])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio = self.chunks[idx].copy()
        label = self.encoded_labels[idx]

        if self.is_training:
            audio = augment_audio(audio, config.SAMPLE_RATE)
            if len(audio) > self.chunk_len:
                audio = audio[:self.chunk_len]
            elif len(audio) < self.chunk_len:
                audio = np.pad(audio, (0, self.chunk_len - len(audio)))

        mel = audio_to_melspectrogram(audio, config.SAMPLE_RATE)
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_train_val_test_datasets(data_dir, val_size=0.15, test_size=0.15):
    file_paths, labels = collect_files(data_dir)

    print("\nFile distribution:")
    for cls, count in sorted(Counter(labels).items()):
        print(f"  {cls:<55}: {count} files")

    encoder = LabelEncoder()
    encoder.fit(labels)

    # Step 1: split off test set first and lock it away
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    # Step 2: split remaining into train and val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=42
    )

    print(f"\nFile-level split: {len(train_paths)} train | "
          f"{len(val_paths)} val | {len(test_paths)} test")
    print("(Test set locked — never used during training)\n")

    train_ds = UnderwaterDataset(train_paths, train_labels, encoder, is_training=True)
    val_ds   = UnderwaterDataset(val_paths,   val_labels,   encoder, is_training=False)
    test_ds  = UnderwaterDataset(test_paths,  test_labels,  encoder, is_training=False)

    print("\nFinal chunk counts after balancing:")
    train_counts = Counter(train_ds.encoded_labels)
    val_counts   = Counter(val_ds.encoded_labels)
    for i, cls in enumerate(encoder.classes_):
        print(f"  {cls:<55}: {train_counts.get(i,0):>5} train | "
              f"{val_counts.get(i,0):>4} val")

    return train_ds, val_ds, test_ds, encoder