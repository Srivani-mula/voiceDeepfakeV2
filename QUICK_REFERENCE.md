# Voice Deepfake Detection - Quick Reference

## 🚀 Launch Commands

### Start Streamlit App
```bash
streamlit run app.py
```

### Windows Automatic
```batch
run_app.bat
```

### Linux/Mac Automatic
```bash
./run_app.sh
```

## 📊 Train Models

### Quick Train (5 min)
```bash
python src/train/train_asvspoof2021.py --track LA --epochs 5
```

### Full Training
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 32 \
    --epochs 50
```

### Multi-Track
```bash
# LA
python src/train/train_asvspoof2021.py --track LA --epochs 50

# PA
python src/train/train_asvspoof2021.py --track PA --epochs 50

# DF
python src/train/train_asvspoof2021.py --track DF --epochs 50
```

## 🔍 Verify Setup
```bash
python verify_setup.py
```

## 📦 Docker

### Build
```bash
docker build -t deepfake-detection .
```

### Run
```bash
docker run -p 8501:8501 deepfake-detection
```

### Compose
```bash
docker-compose up
```

## 🎯 Dataset Setup

1. Download: https://www.asvspoof.org/index2021.html
2. Extract to: `data/raw/asvspoof2021/`
3. Verify: `python verify_setup.py`

## 📁 Directory Structure

```
├── app.py                      # Streamlit app
├── config.py                   # Configuration
├── verify_setup.py             # Verification
│
├── src/
│   ├── data/                   # Dataset loaders
│   ├── models/                 # Model definitions
│   ├── features/               # Feature extraction
│   ├── train/                  # Training scripts
│   └── eval/                   # Evaluation
│
├── data/raw/asvspoof2021/      # ASVspoof2021 dataset
├── models/                     # Pre-trained models
└── temp/                       # Temporary files
```

## 🎮 Web Interface

- **🏠 Home**: Overview
- **🔍 Inference**: Upload audio, get prediction
- **📊 Train**: Train models
- **📈 Evaluate**: View metrics
- **ℹ️ About**: Documentation

## 💻 Python API

### Load Model
```python
import torch
from src.models.custom_cnn import CustomCNN

model = CustomCNN()
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
```

### Load Dataset
```python
from src.data.asvspoof2021_dataset import ASVspoof2021Dataset

dataset = ASVspoof2021Dataset(track="LA", split="eval")
sample, label = dataset[0]
```

### Get Predictions
```python
from src.features.log_mel import LogMelExtractor
import torchaudio

waveform, sr = torchaudio.load('audio.wav')
extractor = LogMelExtractor()
x = extractor(waveform)
output = model(x)
prob = torch.softmax(output, dim=1)
```

## ⚙️ Configuration (config.py)

```python
# Change dataset track
ASVSPOOF2021_TRACK = "LA"  # or PA, DF

# Adjust training
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
}
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | `--batch-size 8` |
| Module not found | `pip install -r requirements.txt` |
| Dataset missing | Download from asvspoof.org |
| Slow training | Use GPU, increase batch size |
| Dataset error | Run `python verify_setup.py` |

## 📊 Training Arguments

```bash
--track {LA,PA,DF}         Dataset track
--split {train,dev,eval}   Data split
--batch-size INT           Batch size
--epochs INT               Training epochs
--lr FLOAT                 Learning rate
--weight-decay FLOAT       L2 regularization
--device {cuda,cpu}        Device
--save-path STR            Model save path
--no-save                  Don't save model
```

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| EER | ~2-3% |
| Min-tDCF | ~0.1 |

## 🌐 Deployment

### Cloud (Streamlit Cloud)
1. Push to GitHub
2. Go to streamlit.io/cloud
3. Select repo → deploy

### Docker
```bash
docker-compose up
```

### AWS/Azure/GCP
See DEPLOYMENT_GUIDE.md

## 📚 Documentation

- **README.md** - Overview
- **DEPLOYMENT_GUIDE.md** - Deployment
- **TRAINING_GUIDE.md** - Training
- **PROJECT_SUMMARY.md** - Changes
- **config.py** - Configuration

## 🔗 Links

- ASVspoof: https://www.asvspoof.org/
- Streamlit Docs: https://docs.streamlit.io/
- PyTorch: https://pytorch.org/
- LibROSA: https://librosa.org/

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted
- [ ] Setup verified (`python verify_setup.py`)
- [ ] Model trained or pre-trained weights available
- [ ] Streamlit app running (`streamlit run app.py`)

## 🎨 Model Architecture

```
Input (1×128×128)
    ↓
Conv2d(1→16) + ReLU + MaxPool2d
    ↓
Conv2d(16→32) + ReLU + MaxPool2d
    ↓
Conv2d(32→64) + ReLU + MaxPool2d
    ↓
Flatten
    ↓
Linear(4096→128) + ReLU
    ↓
Linear(128→2) + Softmax
    ↓
Output: [Bonafide, Spoof]
```

## 🎵 Supported Formats

- WAV
- FLAC
- MP3

All converted to 16kHz mono

---

**Version**: 2.0 | **Status**: ✅ Production Ready | **Updated**: 2026
