import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR = "data/raw/asvspoof2019/LA"

AUDIO_DIR = (
    f"{BASE_DIR}/ASVspoof2019_LA_train/flac"
)

PROTOCOL_FILE = (
    f"{BASE_DIR}/ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)

# ================= DATASET =================
class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, protocol_file):
        self.samples = []

        # FIXED mel settings
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80
        )

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                label = parts[-1]

                y = 0 if label == "bonafide" else 1
                audio_path = os.path.join(audio_dir, file_id + ".flac")

                if os.path.exists(audio_path):
                    self.samples.append((audio_path, y))

        print(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ✅ SAFE AUDIO LOADING (NO torchcodec)
        waveform, sr = sf.read(path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample manually if needed
        if sr != 16000:
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                scale_factor=16000 / sr,
                mode="linear",
                align_corners=False
            ).squeeze(0)

        # Mel Spectrogram
        mel = self.melspec(waveform)      # [1, 80, T]
        mel = mel[:, :, :128]

        if mel.shape[2] < 128:
            mel = F.pad(mel, (0, 128 - mel.shape[2]))

        return mel, torch.tensor(label, dtype=torch.long)

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 20 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ================= TRAIN =================
def train():
    dataset = ASVspoofDataset(AUDIO_DIR, PROTOCOL_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "models/cnn_asvspoof.pth")
    print("✅ Model saved to models/cnn_asvspoof.pth")

# ================= RUN =================
if __name__ == "__main__":
    train()
