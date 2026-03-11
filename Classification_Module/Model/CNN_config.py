import torch

# --- Paths ---
DATA_DIR     = r'D:\MAIN PROJECT\UnderWater-Acoustic-Detection\Dataset\DDDDD'
METADATA_FILE = r'D:\MAIN PROJECT\UnderWater-Acoustic-Detection\Dataset\DDDDD\audio_metadata1.csv'
CNN_MODEL_PATH   = r'D:\MAIN PROJECT\UnderWater-Acoustic-Detection\Classification_Module\Model\CNN_best_model.pth'
CLASSES_PATH = r'D:\MAIN PROJECT\UnderWater-Acoustic-Detection\Classification_Module\Model\classes.json'

# --- Audio Settings ---
SAMPLE_RATE    = 16000
CHUNK_DURATION = 3
N_MELS         = 64
HOP_LENGTH     = 512

# --- Training ---
BATCH_SIZE    = 32
EPOCHS        = 50
PATIENCE      = 10
LEARNING_RATE = 0.001
MAX_CHUNKS_PER_FILE = 10
MAX_CHUNKS_PER_CLASS = 100  # every class gets at most 100 chunks
CONFIDENCE_THRESHOLD = 70

# --- Skip these — too few files to learn from ---
SKIP_CLASSES = [
    "Blue Whale",             # 10 files
    "Fin Whale",              # 14 files
    "Leopard Seal",           # 10 files
    "Short-Finned (Pacific) Pilot Whale",  # 16 files
    "Weddell Seal",           # 2 files
    "Hurricane",              # 2 files
    "Waves",                  # 5 files
    "Wind",                   # 5 files
    "Rainfall",      
    "Soundscape"
]

# --- Skip entire top-level folders ---
SKIP_FOLDERS = ["Other anthropogenic", "Natural Sounds"]

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")