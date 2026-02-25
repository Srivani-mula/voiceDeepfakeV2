# 🎙️ Voice Deepfake Detection System

A production-ready deep learning system for detecting deepfaked and spoofed audio using Streamlit and ASVspoof datasets.

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🚀 Quick Start

### Windows Users
Double-click `run_app.bat` and the app will launch automatically.

### Linux/Mac Users
```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual Launch
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser to http://localhost:8501
```

## 🎯 Features

✅ **Real-time Audio Inference**
- Upload audio files (WAV, FLAC, MP3)
- Instant deepfake detection
- Confidence scores and probabilities
- Spectrogram visualization

✅ **Dataset Support**
- ASVspoof2019 (Logical Access)
- ASVspoof2021 (LA, PA, DF tracks)
- Flexible dataset configuration

✅ **Model Training**
- Train custom models on ASVspoof2021
- Configurable hyperparameters
- Best model saving
- Training history tracking

✅ **Comprehensive Evaluation**
- Accuracy metrics
- EER (Equal Error Rate)
- Min-tDCF scores
- Performance visualization

✅ **Production Ready**
- Streamlit cloud deployment
- Docker containerization
- AWS/Azure/GCP compatible

## 📋 Requirements

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)
- 4GB RAM minimum
- 10GB storage (for datasets)

## 📦 Installation

### Option 1: Automated Setup (Recommended)

**Windows:**
```batch
run_app.bat
```

**Linux/Mac:**
```bash
./run_app.sh
```

### Option 2: Manual Setup

```bash
# Clone/navigate to project
cd voice-deepfake-detection-v2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 🎮 Using the Application

### Home Page
Overview of the system and quick start guide.

### 🔍 Inference
1. Upload an audio file (WAV, FLAC, or MP3)
2. Get instant prediction
3. View confidence score
4. Analyze spectrogram

### 📊 Train Model
Configure and train models on ASVspoof2021 dataset.

### 📈 Evaluate
View model metrics and performance statistics.

### ℹ️ About
Project information and documentation.

## 🔧 Training a New Model

### Command Line Training
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda
```

### Training Options
```
--track {LA,PA,DF}          Dataset track (default: LA)
--split {train,dev,eval}    Data split (default: train)
--batch-size              Batch size (default: 32)
--epochs                  Number of epochs (default: 50)
--lr                      Learning rate (default: 0.001)
--weight-decay            L2 regularization (default: 0)
--device {cuda,cpu}       Device (default: cuda)
--save-path               Custom model save path
```

## 📊 Datasets

### ASVspoof2021
Download from: https://www.asvspoof.org/index2021.html

**Tracks:**
- **LA (Logical Access)**: Voice conversion and synthesis attacks
- **PA (Physical Access)**: Replay and voice conversion
- **DF (DeepFake)**: AI-generated deepfake audio

**Setup:**
1. Download datasets from ASVspoof website
2. Extract to `data/raw/asvspoof2021/`
3. Update paths in `config.py` if needed

### Dataset Structure
```
data/raw/asvspoof2021/
├── ASVspoof2021_LA_train/
│   ├── flac/          # Audio files
│   └── *.txt          # Protocol files
├── ASVspoof2021_LA_dev/
├── ASVspoof2021_LA_eval/
├── ASVspoof2021_PA_eval_part00/
└── ASVspoof2021_DF_eval_part00/
```

## 🤖 Model Architecture

**CustomCNN:**
- Input: Log-Mel Spectrogram (1×128×128)
- Conv2d(1→16) → ReLU → MaxPool2d
- Conv2d(16→32) → ReLU → MaxPool2d
- Conv2d(32→64) → ReLU → MaxPool2d
- Flatten → Linear(4096→128) → ReLU
- Linear(128→2) → Softmax
- Output: 2 classes (Bonafide/Spoof)

## 🚢 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create new app and select your repo
4. App deploys automatically on each push

### Docker Deployment
```bash
docker build -t deepfake-detection .
docker run -p 8501:8501 deepfake-detection
```

### AWS EC2
```bash
ssh -i key.pem ec2-user@instance-ip
sudo apt update && sudo apt install python3-pip
git clone your-repo
cd voice-deepfake-detection-v2
pip install -r requirements.txt
streamlit run app.py
```

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 📈 Performance Metrics

The system uses standard ASVspoof evaluation metrics:

- **Accuracy**: Percentage of correct classifications
- **EER (Equal Error Rate)**: False accept rate = false reject rate
- **Min-tDCF**: Minimum tandem detection cost function

## 🔍 Labels

- **0 = Bonafide (Genuine)**: Real human voice
- **1 = Spoof/Fake**: Synthesized or manipulated audio

## 📁 Project Structure

```
voice-deepfake-detection-v2/
├── app.py                          # Main Streamlit app
├── config.py                        # Core configuration
├── requirements.txt                 # Dependencies
├── run_app.py/bat/sh               # Quick start scripts
├── DEPLOYMENT_GUIDE.md             # Detailed deployment guide
├── README.md                        # This file
│
├── .streamlit/
│   └── config.toml                 # Streamlit settings
│
├── src/
│   ├── data/
│   │   ├── asvspoof2019_dataset.py       # ASVspoof2019 loader
│   │   ├── asvspoof2021_dataset.py       # ASVspoof2021 loader
│   │   └── asvspoof2019_feature_dataset.py
│   ├── models/
│   │   ├── custom_cnn.py           # Main model
│   │   ├── pretrained_resnet.py
│   │   └── cnn_model.py
│   ├── features/
│   │   ├── log_mel.py              # Spectrogram extraction
│   │   └── augment.py              # Data augmentation
│   ├── train/
│   │   ├── train_asvspoof2021.py   # ASVspoof2021 training
│   │   ├── train.py
│   │   └── train_resnet.py
│   ├── eval/
│   │   ├── eval_resnet_dev.py
│   │   └── plot_det_curve.py
│   ├── utils/
│   │   ├── loss.py
│   │   └── grad_cam.py
│   └── metrics/
│       └── eer.py
│
├── models/                          # Pre-trained models
│   └── best_model.pth
│
├── data/
│   ├── raw/
│   │   ├── asvspoof2019/           # ASVspoof2019 dataset
│   │   └── asvspoof2021/           # ASVspoof2021 dataset
│   └── processed/
│
└── temp/                           # Temporary files
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Training parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "num_workers": 0,  # Set to 0 on Windows
}

# Audio processing
"n_mels": 128,
"hop_length": 160,
"n_fft": 512,
"sample_rate": 16000,

# Dataset paths
ASVSPOOF2021_TRACK = "LA"  # Options: LA, PA, DF
```

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Ensure you're in the correct directory and have installed requirements
pip install -r requirements.txt
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
python src/train/train_asvspoof2021.py --batch-size 16
```

### Issue: Streamlit crashes
```bash
# Update Streamlit
pip install --upgrade streamlit
```

### Issue: Dataset not found
1. Check paths in `config.py`
2. Verify dataset location: `data/raw/asvspoof2021/`
3. See DEPLOYMENT_GUIDE.md for setup

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Full deployment and configuration guide
- **[config.py](config.py)** - All configuration options
- **[src/train/train_asvspoof2021.py](src/train/train_asvspoof2021.py)** - Training pipeline
- **[src/data/asvspoof2021_dataset.py](src/data/asvspoof2021_dataset.py)** - Dataset loader

## 📖 References

- [ASVspoof Challenge](https://www.asvspoof.org/)
- [ASVspoof 2021 Paper](https://arxiv.org/abs/2107.05477)
- [Librosa Documentation](https://librosa.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 Citation

If you use this system, please cite:

```bibtex
@inproceedings{wang2021asvspoof,
  title={The ASVspoof 2021 accelerating progress on spoofed 
         and deepfake speech detection},
  author={Wang, Xin and Yamagishi, Junichi and others},
  booktitle={Interspeech 2021},
  year={2021}
}
```

## 💡 Tips

1. **First Run**: Use the inference page with sample audio to test the model
2. **Training**: Start with smaller batch sizes if you have limited GPU memory
3. **Performance**: Use CUDA for faster training and inference
4. **Monitoring**: Check `models/training_history_*.json` for training progress
5. **Evaluation**: Run evaluation after training to measure model performance

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## ⚖️ License

MIT License - See LICENSE file for details

## 📧 Support

For issues or questions:
1. Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) troubleshooting section
2. Review dataset loading errors in detail
3. Check system logs for runtime errors

---

**Version**: 2.0  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready

**Happy deepfake detecting! 🎵🔍**
