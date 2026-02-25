from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ===========================
# ASVspoof2019 Configuration
# ===========================
ASVSPOOF2019_BASE = PROJECT_ROOT / "data" / "raw" / "asvspoof2019" / "LA"

ASVSPOOF2019_LA_TRAIN_FLAC = ASVSPOOF2019_BASE / "ASVspoof2019_LA_train" / "flac"
ASVSPOOF2019_LA_DEV_FLAC = ASVSPOOF2019_BASE / "ASVspoof2019_LA_dev" / "flac"
ASVSPOOF2019_LA_EVAL_FLAC = ASVSPOOF2019_BASE / "ASVspoof2019_LA_eval" / "flac"

ASVSPOOF2019_PROTOCOLS = ASVSPOOF2019_BASE / "ASVspoof2019_LA_cm_protocols"

# ===========================
# ASVspoof2021 Configuration
# ===========================
ASVSPOOF2021_BASE = PROJECT_ROOT / "data" / "raw" / "asvspoof2021"

# Supported tracks: LA (Logical Access), PA (Physical Access), DF (DeepFake)
ASVSPOOF2021_TRACK = "LA"  # Change this to "PA" or "DF" as needed

# ===========================
# Training Configuration
# ===========================
TRAINING_CONFIG = {
    "device": "cuda",  # or "cpu"
    "sample_rate": 16000,
    "n_mels": 128,
    "hop_length": 160,
    "n_fft": 512,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,  # Set to 0 to avoid Windows issues
    "pin_memory": False,
}

# ===========================
# Model Configuration
# ===========================
MODEL_CONFIG = {
    "num_classes": 2,
    "model_type": "custom_cnn",  # "custom_cnn" or "resnet18"
}

# ===========================
# Paths
# ===========================
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp"
LOG_DIR = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
