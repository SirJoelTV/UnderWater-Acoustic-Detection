import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from utils.audio_utils import AudioProcessor
import yaml

class UnderwaterAcousticDataset(Dataset):
    def __init__(self, data_dir, config, mode='train'):
        self.data_dir = data_dir
        self.config = config
        self.mode = mode
        self.audio_processor = AudioProcessor(config)
        
        # Define categories
        self.categories = {
            'ships': ['cargo', 'tanker', 'passenger', 'tug', 'fishing', 'military'],
            'marine_life': ['whale', 'dolphin', 'seal', 'fish', 'shrimp'],
            'ambient': ['wind', 'rain', 'waves', 'thermal'],
            'anomaly': ['unknown']
        }
        
        self.label_encoder = LabelEncoder()
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset from directory structure"""
        samples = []
        
        # Expected structure: data_dir/category/subcategory/files.wav
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            if not os.path.exists(category_path):
                continue
                
            for subcategory in self.categories[category]:
                subcat_path = os.path.join(category_path, subcategory)
                if not os.path.exists(subcat_path):
                    continue
                    
                for file in os.listdir(subcat_path):
                    if file.endswith(('.wav', '.flac', '.mp3')):
                        samples.append({
                            'file_path': os.path.join(subcat_path, file),
                            'category': category,
                            'subcategory': subcategory,
                            'label': f"{category}_{subcategory}"
                        })
        
        # Encode labels
        labels = [sample['label'] for sample in samples]
        self.label_encoder.fit(labels)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess audio
        audio, _ = self.audio_processor.load_audio(sample['file_path'])
        if audio is None:
            # Return zero tensor if loading fails
            audio = np.zeros(int(self.config['data']['sample_rate'] * 
                                self.config['data']['duration']))
        
        audio = self.audio_processor.preprocess_audio(audio)
        features = self.audio_processor.extract_features(audio)
        
        # Convert to tensors
        mel_spec = torch.FloatTensor(features['mel_spec'])
        mfcc = torch.FloatTensor(features['mfcc'])
        lofar = torch.FloatTensor(features['lofar'])
        
        # Get label
        label = self.label_encoder.transform([sample['label']])[0]
        
        return {
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'lofar': lofar,
            'label': torch.LongTensor([label]),
            'category': sample['category'],
            'file_path': sample['file_path']
        }

def create_dataloaders(data_dir, config, split_ratio=(0.7, 0.2, 0.1)):
    """Create train, validation, and test dataloaders"""
    
    # Load full dataset
    full_dataset = UnderwaterAcousticDataset(data_dir, config)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, full_dataset.label_encoder