# Voice Deepfake Detection - Complete Training Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Quick Training](#quick-training)
4. [Advanced Training](#advanced-training)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8+
- 8GB RAM minimum
- 20GB storage (for datasets)
- GPU recommended (CUDA 11.0+)

### Software Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

### Check GPU Availability
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # GPU name
```

---

## Dataset Setup

### Step 1: Download ASVspoof2021

Visit: https://www.asvspoof.org/index2021.html

Download the dataset(s) you want:
- **LA (Logical Access)** - Recommended for beginners
- **PA (Physical Access)** - For advanced training
- **DF (DeepFake)** - Latest deepfake detection

### Step 2: Extract Dataset

Extract to the following structure:
```
data/raw/asvspoof2021/
├── ASVspoof2021_LA_train/
│   ├── ASVspoof2021_LA_train/
│   │   └── flac/              (audio files)
│   └── ASVspoof2021.LA.cm.train.trn.txt  (protocol)
├── ASVspoof2021_LA_dev/
│   ├── ASVspoof2021_LA_dev/
│   │   └── flac/
│   └── ASVspoof2021.LA.cm.dev.trl.txt
└── ASVspoof2021_LA_eval/
    ├── ASVspoof2021_LA_eval/
    │   └── flac/
    └── ASVspoof2021.LA.cm.eval.trl.txt
```

### Step 3: Verify Dataset

```bash
python verify_setup.py
```

Should show:
```
✅ ASVspoof2021: data/raw/asvspoof2021 (XXXX files)
```

---

## Quick Training

### Minimal Setup (5 minutes)

Train a model with default settings on LA eval set:

```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split eval \
    --epochs 5
```

Expected output:
```
Loading ASVspoof2021 LA eval split...
Dataset loaded. Total: 71237 samples
Bonafide: 63373 (89.0%)
Spoof: 7864 (11.0%)

Starting training for 5 epochs...
Epoch 1/5 | Batch 100/500 | Loss: 0.1234
...
✓ Best model saved: models/best_model_asvspoof2021_LA.pth
```

### Train with Development Set (Recommended)

```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 32 \
    --epochs 20 \
    --lr 0.001
```

Expected time: 2-4 hours on GPU

---

## Advanced Training

### High-Performance Configuration

For best results on full training set:

```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --device cuda
```

### Multi-Track Training

Train separate models for each track:

**LA (Logical Access):**
```bash
python src/train/train_asvspoof2021.py \
    --track LA --split train --epochs 50
```

**PA (Physical Access):**
```bash
python src/train/train_asvspoof2021.py \
    --track PA --split train --epochs 50
```

**DF (DeepFake):**
```bash
python src/train/train_asvspoof2021.py \
    --track DF --split eval --epochs 50
```

### Custom Hyperparameters

Optimize for your hardware:

**Low Memory (4GB GPU):**
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 8 \
    --epochs 50 \
    --lr 0.0005
```

**High Memory (16GB+ GPU):**
```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.002
```

### Save Custom Location

```bash
python src/train/train_asvspoof2021.py \
    --track LA \
    --split train \
    --save-path custom_models/my_model_v1.pth \
    --epochs 50
```

---

## Monitoring Training

### Real-time Monitoring

Check loss in console output:
```
Epoch 1/50 | Avg Loss: 0.4532 | LR: 0.001000
Epoch 2/50 | Avg Loss: 0.3421 | LR: 0.001000
Epoch 3/50 | Avg Loss: 0.2987 | LR: 0.000998
```

### Training History

After training, view saved history:

```bash
python -c "
import json
with open('models/training_history_LA.json') as f:
    history = json.load(f)
    for i, loss in enumerate(history['losses'][-10:]):
        print(f'Epoch {len(history[\"losses\"])-10+i+1}: {loss:.4f}')
"
```

### Visualize Training

```python
import json
import matplotlib.pyplot as plt

with open('models/training_history_LA.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['epochs'], history['losses'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history['epochs'], history['learning_rates'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
```

---

## Using Trained Models

### Load Trained Model

```python
import torch
from src.models.custom_cnn import CustomCNN

# Load model
model = CustomCNN(num_classes=2)
state = torch.load('models/best_model_asvspoof2021_LA.pth', 
                   map_location='cpu')
model.load_state_dict(state)
model.eval()

# Use for inference
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    prediction = torch.argmax(probs, dim=1)
```

### Test on New Audio

```python
import torchaudio
from src.features.log_mel import LogMelExtractor

# Load audio
waveform, sr = torchaudio.load('test_audio.wav')
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Extract features
extractor = LogMelExtractor()
x = extractor(waveform)

# Predict
with torch.no_grad():
    output = model(x)
    probs = torch.softmax(output, dim=1)
    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0, label].item()

print(f"Prediction: {'Bonafide' if label == 0 else 'Spoof'}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## Troubleshooting

### Out of Memory (OOM)

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python src/train/train_asvspoof2021.py \
    --batch-size 8 --track LA --split train

# Use CPU
python src/train/train_asvspoof2021.py \
    --device cpu --batch-size 32
```

### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Use GPU instead of CPU
2. Increase batch size (if memory allows)
3. Use smaller dataset:
   ```bash
   python src/train/train_asvspoof2021.py \
       --track LA --split eval  # Smaller eval set
   ```

### Audio File Not Found

**Problem:**
```
FileNotFoundError: Audio file not found
```

**Solutions:**
1. Verify dataset extraction structure
2. Check file extensions (.flac or .wav)
3. Run verification script:
   ```bash
   python verify_setup.py
   ```

### Protocol File Not Found

**Problem:**
```
FileNotFoundError: Could not find protocol file
```

**Solutions:**
1. Ensure protocol files are in the extracted dataset
2. Check file naming matches ASVspoof format
3. Re-download dataset if corrupted

### Library Conflicts

**Problem:** Import errors or version conflicts

**Solutions:**
```bash
# Fresh install in new environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Performance Tips

### Training Optimization

1. **Use GPU**: 5-10x faster than CPU
2. **Increase batch size**: Faster iterations (if memory allows)
3. **Reduce logging**: Less I/O overhead
4. **Pin memory**: Better GPU utilization
   ```python
   DataLoader(..., pin_memory=True, num_workers=4)
   ```

### Model Optimization

1. **Use smaller input**: Instead of 128×128, try 64×64
2. **Reduce model size**: Fewer filters in Conv layers
3. **Quantization**: Convert to int8 or half precision
4. **Pruning**: Remove unnecessary weights

### Data Optimization

1. **Preprocessing cache**: Pre-extract log-mel features
2. **Data augmentation**: Avoid computational overhead
3. **Sampling**: Train on representative subset first

---

## Expected Results

### ASVspoof2021 LA Baseline

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| EER | ~2-3% |
| Min-tDCF | ~0.1 |

Your results may vary based on:
- Training data size
- Model architecture
- Hyperparameters
- Data augmentation

---

## Model Checkpoints

Models are saved to `models/`:
- `best_model_asvspoof2021_LA.pth` - Best model during training
- `asvspoof2021_LA_final.pth` - Final model after all epochs
- `training_history_LA.json` - Training metrics

Use the best model for deployment:
```bash
cp models/best_model_asvspoof2021_LA.pth models/best_model.pth
```

---

## Automation Script

Create `auto_train.sh` for automated training:

```bash
#!/bin/bash
for track in LA PA
do
    echo "Training $track track..."
    python src/train/train_asvspoof2021.py \
        --track $track \
        --split train \
        --epochs 50 \
        --batch-size 32 \
        --lr 0.001
done
echo "All training complete!"
```

Run with:
```bash
chmod +x auto_train.sh
./auto_train.sh
```

---

For more information, see:
- [README.md](README.md) - Project overview
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment instructions
- [src/train/train_asvspoof2021.py](src/train/train_asvspoof2021.py) - Training implementation
