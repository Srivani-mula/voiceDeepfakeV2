# Voice Deepfake Detection System - Streamlit Deployment Guide

## Overview
This is a complete deep learning system for detecting deepfaked or spoofed audio using the ASVspoof datasets. It's now fully configured for Streamlit deployment with ASVspoof2021 support.

## Features
- ✅ Real-time audio inference with Streamlit web interface
- ✅ ASVspoof2019 & ASVspoof2021 dataset support
- ✅ Model training and retraining capabilities
- ✅ Comprehensive evaluation metrics
- ✅ Audio visualization (spectrograms)
- ✅ Multi-page Streamlit application

## Project Structure
```
voice-deepfake-detection-v2/
├── app.py                          # Main Streamlit application
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── src/
│   ├── data/
│   │   ├── asvspoof2019_dataset.py       # ASVspoof2019 loader
│   │   ├── asvspoof2021_dataset.py       # ASVspoof2021 loader (NEW)
│   │   └── asvspoof2019_feature_dataset.py
│   ├── models/
│   │   ├── custom_cnn.py           # Custom CNN model
│   │   ├── pretrained_resnet.py
│   │   └── cnn_model.py
│   ├── features/
│   │   ├── log_mel.py              # Log-Mel spectrogram extraction
│   │   └── augment.py
│   ├── train/
│   │   ├── train.py
│   │   ├── train_asvspoof2021.py   # ASVspoof2021 training (NEW)
│   │   └── train_resnet.py
│   └── eval/
│       └── eval_resnet_dev.py
├── models/                          # Pre-trained models
│   ├── best_model.pth
│   └── *.pth
├── data/
│   └── raw/
│       ├── asvspoof2019/           # ASVspoof2019 dataset
│       └── asvspoof2021/           # ASVspoof2021 dataset
└── temp/                           # Temporary files
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Step 1: Clone/Setup the Project
```bash
cd voice-deepfake-detection-v2
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n deepfake-detection python=3.10
conda activate deepfake-detection
pip install -r requirements.txt
```

### Step 3: Configure Dataset Paths
Edit `config.py` and ensure the dataset paths point to your ASVspoof data:

```python
ASVSPOOF2021_BASE = Path("/path/to/data/raw/asvspoof2021")
ASVSPOOF2021_TRACK = "LA"  # or "PA", "DF"
```

## Running the Application

### Local Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using Python Scripts Directly

#### Train a new model (ASVspoof2021)
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda
```

#### Evaluate model performance
```bash
python src/eval/eval_resnet_dev.py
```

## ASVspoof2021 Dataset

### Downloading the Dataset
Download from: https://www.asvspoof.org/index2021.html

### Supported Tracks
1. **LA (Logical Access)**: Voice conversion and speech synthesis attacks
2. **PA (Physical Access)**: Replay and voice conversion attacks  
3. **DF (DeepFake)**: Deepfake audio attacks

### Dataset Structure
```
data/raw/asvspoof2021/
├── ASVspoof2021_LA_train/
│   ├── flac/                        # Training audio files
│   └── *.txt                        # Protocol files
├── ASVspoof2021_LA_dev/
│   ├── flac/
│   └── *.txt
├── ASVspoof2021_LA_eval/
│   ├── flac/
│   └── *.txt
├── ASVspoof2021_PA_eval_part00/
│   └── ASVspoof2021_PA_eval/
└── ASVspoof2021_DF_eval_part00/
    └── ASVspoof2021_DF_eval/
```

## Training a New Model

### Command Line
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --device cuda
```

### Arguments
- `--track`: Dataset track (LA, PA, DF)
- `--split`: Training split (train, dev, eval)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: L2 regularization (default: 0)
- `--device`: Device (cuda or cpu)
- `--no-save`: Don't save the best model
- `--save-path`: Custom save path

### Model Architecture
```
CustomCNN(
    Conv2d(1, 16, 3) → ReLU → MaxPool2d
    Conv2d(16, 32, 3) → ReLU → MaxPool2d
    Conv2d(32, 64, 3) → ReLU → MaxPool2d
    Flatten
    Linear(64*16*16, 128) → ReLU
    Linear(128, 2)  # 2 classes: Bonafide/Spoof
)
```

## Streamlit App Features

### Pages

#### 1. 🏠 Home
- Project overview
- Quick start guide
- Model status

#### 2. 🔍 Inference
- Upload audio files (WAV, FLAC, MP3)
- Real-time prediction
- Confidence scores
- Spectrogram visualization
- Probability charts

#### 3. 📊 Train Model
- Select dataset track (LA/PA/DF)
- Configure training parameters
- Start training (requires command line for full training)

#### 4. 📈 Evaluate
- Model performance metrics
- Architecture details
- Training history
- Accuracy, EER, Min-tDCF scores

#### 5. ℹ️ About
- Project information
- Dataset details
- Configuration guide
- Usage documentation

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create new app and select repo/branch
4. Configure environment variables (if needed)

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t deepfake-detection .
docker run -p 8501:8501 deepfake-detection
```

### AWS/Azure/GCP Deployment
1. Install cloud CLI tools
2. Upload code to cloud storage
3. Create compute instance
4. Install dependencies
5. Run: `streamlit run app.py --server.port 8501`
6. Configure load balancer/firewall

## Troubleshooting

### Issue: Model not found
**Solution**: Place model weights in `models/` directory or train a new model

### Issue: Dataset not found
**Solution**: Check paths in `config.py` and ensure datasets are in correct location

### Issue: CUDA out of memory
**Solution**: Reduce batch size:
```bash
python src/train/train_asvspoof2021.py --batch-size 16 --epochs 50
```

### Issue: Slow inference
**Solution**: Use GPU, ensure CUDA is available and installed

### Issue: Streamlit not loading
**Solution**: 
```bash
pip install --upgrade streamlit
streamlit run app.py --logger.level=debug
```

## Configuration Options

Edit `config.py` to customize:

```python
# Training
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
}

# Audio Processing
"n_mels": 128,
"hop_length": 160,
"n_fft": 512,

# Model
"num_classes": 2,
"model_type": "custom_cnn",
```

## Performance Metrics

The system evaluates using standard ASVspoof metrics:
- **Accuracy**: Percentage of correct predictions
- **EER** (Equal Error Rate): False positive rate = false negative rate
- **Min-tDCF** (minimum tandem detection cost function): Detection cost metric used in ASVspoof

## Data Labels

- **0 = Bonafide (Genuine)**: Real human voice
- **1 = Spoof/Fake**: Artificially generated or manipulated voice

## Citation

If using this system, cite the relevant papers:

```
@inproceedings{wang2020asvspoof,
  title={The ASVspoof 2021 accelerating progress on spoofed and deepfake speech detection},
  author={Wang, Xin and others},
  booktitle={Interspeech 2021},
  year={2021}
}
```

## Further Resources

- ASVspoof Website: https://www.asvspoof.org/
- Librosa Documentation: https://librosa.org/
- PyTorch Documentation: https://pytorch.org/
- Streamlit Documentation: https://docs.streamlit.io/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review dataset loading in `src/data/asvspoof2021_dataset.py`
3. Check model architecture in `src/models/custom_cnn.py`
4. Review training progress in `models/training_history_*.json`

---

**Version**: 2.0  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready
