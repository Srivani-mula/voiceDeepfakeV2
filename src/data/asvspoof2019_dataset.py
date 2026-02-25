import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


# =========================================
# PAD OR TRUNCATE
# =========================================
def pad_or_truncate(spec, target_length=200):
    if spec.shape[1] > target_length:
        spec = spec[:, :target_length]
    elif spec.shape[1] < target_length:
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")
    return spec


# =========================================
# SPEC AUGMENT
# =========================================
def spec_augment(spec, freq_mask=10, time_mask=20):

    spec = spec.copy()

    # Frequency masking
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, spec.shape[0] - f)
    spec[f0:f0 + f, :] = 0

    # Time masking
    t = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, spec.shape[1] - t)
    spec[:, t0:t0 + t] = 0

    return spec


# =========================================
# DATASET
# =========================================
class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_path, split="train", target_length=200):

        self.root_path = root_path
        self.split = split
        self.target_length = target_length

        protocol_dir = os.path.join(
            root_path,
            "ASVspoof2019_LA_cm_protocols"
        )

        if split == "train":
            protocol_file = os.path.join(
                protocol_dir,
                "ASVspoof2019.LA.cm.train.trn.txt"
            )
            audio_dir = os.path.join(
                root_path,
                "ASVspoof2019_LA_train",
                "flac"
            )

        elif split == "dev":
            protocol_file = os.path.join(
                protocol_dir,
                "ASVspoof2019.LA.cm.dev.trl.txt"
            )
            audio_dir = os.path.join(
                root_path,
                "ASVspoof2019_LA_dev",
                "flac"
            )
        else:
            raise ValueError("Split must be train or dev")

        self.audio_dir = audio_dir
        self.file_list = []

        with open(protocol_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            file_name = parts[1]
            label = 0 if parts[-1] == "bonafide" else 1
            file_path = os.path.join(audio_dir, file_name + ".flac")
            self.file_list.append((file_path, label))

        print(f"{split.upper()} samples loaded: {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        file_path, label = self.file_list[idx]

        waveform, sr = librosa.load(file_path, sr=16000)

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel = pad_or_truncate(log_mel, self.target_length)

        # 🔥 Apply augmentation ONLY during training
        if self.split == "train":
            log_mel = spec_augment(log_mel)

        # Normalize
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

        # Convert to 3-channel
        log_mel = np.stack([log_mel, log_mel, log_mel], axis=0)

        log_mel = torch.tensor(log_mel, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return log_mel, label