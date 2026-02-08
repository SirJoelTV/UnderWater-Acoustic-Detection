import torch
import os

# Paths
DATA_DIR = r'D:\Main Project\UnderWater-Acoustic-Detection\data_balanced'  # Updated to balanced dataset
MODEL_PATH = "best_model.pth"
CLASSES_PATH = "classes.json"
GLOBAL_STATS_PATH = "global_stats.npz"  # For normalization statistics

# Audio Settings
SAMPLE_RATE = 32000
DURATION = 30
N_MELS = 64
HOP_LENGTH = 512

# Training Hyperparameters
BATCH_SIZE = 32  # Increased from 8 to reduce gradient noise
EPOCHS = 50  # Increased to allow more learning
LEARNING_RATE = 0.001  # Increased slightly for better learning
WEIGHT_DECAY = 1e-4  # Reduced to allow more learning
PATIENCE = 10  # Increased patience for early stopping
WARMUP_EPOCHS = 3  # Add learning rate warmup

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    print(f"Using device: {DEVICE}")
    print(f"Normalization: Global statistics from training set")
    print(f"Augmentation: Realistic (frequency shift, time shift, subtle noise)")
    return DEVICE