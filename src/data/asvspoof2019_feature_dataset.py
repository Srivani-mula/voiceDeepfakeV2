import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

from src.data.asvspoof2019_dataset import ASVspoof2019Dataset
from src.features.log_mel import LogMelExtractor


# =========================
# Waveform Augmentations
# =========================

def add_noise(waveform, snr_db_range=(5, 20)):
    """
    Add Gaussian noise with random SNR
    """
    noise = torch.randn_like(waveform)

    signal_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr_db = random.uniform(*snr_db_range)
    snr = 10 ** (snr_db / 10)

    scale = torch.sqrt(signal_power / (snr * noise_power))
    return waveform + scale * noise


def random_gain(waveform, gain_range=(0.7, 1.3)):
    """
    Random volume scaling
    """
    gain = random.uniform(*gain_range)
    return waveform * gain


# =========================
# SpecAugment
# =========================

class SpecAugment:
    """
    CPU-friendly SpecAugment (Time + Frequency masking)
    """

    def __init__(self, time_mask=30, freq_mask=8):
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def __call__(self, spec):
        """
        spec: [1, n_mels, T]
        """
        _, n_mels, time_steps = spec.shape

        # ---- Time Mask ----
        t = torch.randint(0, self.time_mask + 1, (1,)).item()
        if t > 0:
            t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
            spec[:, :, t0:t0 + t] = 0

        # ---- Frequency Mask ----
        f = torch.randint(0, self.freq_mask + 1, (1,)).item()
        if f > 0:
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            spec[:, f0:f0 + f, :] = 0

        return spec


# =========================
# Final Feature Dataset
# =========================

class ASVspoof2019FeatureDataset(Dataset):
    """
    FINAL dataset for CNN training

    ✔ Fixed-length Log-Mel Spectrograms
    ✔ Optional waveform augmentation
    ✔ SpecAugment (train only)
    ✔ CPU-safe & stable
    """

    def __init__(
        self,
        split="train",
        sample_rate=16000,
        n_mels=80,
        max_frames=300,
        augment=True,
    ):
        self.split = split
        self.max_frames = max_frames
        self.augment = augment and split == "train"

        self.base_dataset = ASVspoof2019Dataset(split=split)

        self.feature_extractor = LogMelExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        self.spec_augment = SpecAugment()

    def _pad_or_truncate(self, x):
        """
        x: [1, n_mels, T]
        """
        T = x.shape[-1]

        if T > self.max_frames:
            return x[:, :, :self.max_frames]
        else:
            pad = self.max_frames - T
            return F.pad(x, (0, pad))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        waveform, label = self.base_dataset[idx]

        # ---- Waveform Augmentation (train only) ----
        if self.augment:
            if random.random() < 0.5:
                waveform = add_noise(waveform)
            if random.random() < 0.5:
                waveform = random_gain(waveform)

        # ---- Log-Mel Extraction ----
        logmel = self.feature_extractor(waveform)  # [1, 80, T]

        # ---- Fix length ----
        logmel = self._pad_or_truncate(logmel)

        # ---- SpecAugment ----
        if self.augment:
            logmel = self.spec_augment(logmel)

        return logmel, torch.tensor(label, dtype=torch.long)
