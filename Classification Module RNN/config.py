import torch
import os

# Paths
DATA_DIR = r'D:\Main Project\UnderWater-Acoustic-Detection\data'
MODEL_PATH = "best_gpu_lite_rnn_model.pth"
CLASSES_PATH = "classes.json"

# Audio Settings
SAMPLE_RATE = 32000
DURATION = 15
N_MELS = 64
HOP_LENGTH = 512

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    print(f"Using device: {DEVICE}")
    return DEVICE