import os
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ASVspoof2021Dataset(Dataset):
    """
    Dataset loader for ASVspoof2021
    Supports LA (Logical Access), PA (Physical Access), and DF (DeepFake) tracks
    """

    def __init__(
        self,
        track="LA",  # 'LA', 'PA', or 'DF'
        split="eval",  # 'train', 'dev', 'eval'
        base_dir="data/raw/asvspoof2021",
        sample_rate=16000,
        verbose=True
    ):
        self.track = track.upper()
        self.split = split.lower()
        self.sample_rate = sample_rate
        self.verbose = verbose

        # Validate track and split
        valid_tracks = ["LA", "PA", "DF"]
        valid_splits = ["train", "dev", "eval"]

        if self.track not in valid_tracks:
            raise ValueError(f"track must be one of {valid_tracks}, got {self.track}")
        if self.split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {self.split}")

        self.base_dir = base_dir

        # Find the correct directory structure
        self.data_dir = self._find_data_dir()
        self.protocol_file = self._find_protocol_file()

        if self.verbose:
            print(f"Data directory: {self.data_dir}")
            print(f"Protocol file: {self.protocol_file}")

        self.data = self._load_protocol()

        if self.verbose:
            print(f"Loaded {len(self.data)} samples for {self.track} {self.split}")

    def _find_data_dir(self):
        """Find the directory containing audio files"""
        base_path = Path(self.base_dir)
        
        # For LA, DF, PA without parts
        dir_name = f"ASVspoof2021_{self.track}_{self.split}"
        test_paths = [
            base_path / dir_name / dir_name / "flac",  # Nested structure
            base_path / dir_name / "flac",              # Flat structure
        ]
        
        for path in test_paths:
            if path.exists():
                if self.verbose:
                    print(f"Found FLAC directory: {path}")
                return str(path)
        
        # For multi-part (PA, DF) - try part00, part01, etc.
        for part_idx in range(10):
            part_name = f"ASVspoof2021_{self.track}_{self.split}_part{part_idx:02d}"
            test_paths = [
                base_path / part_name / dir_name / "flac",  # Nested
                base_path / part_name / "flac",             # Flat
            ]
            
            for path in test_paths:
                if path.exists():
                    if self.verbose:
                        print(f"Found FLAC directory (part {part_idx:02d}): {path}")
                    return str(path)
        
        # If nothing found, list what we have
        if self.verbose:
            print(f"\nAvailable directories in {self.base_dir}:")
            if base_path.exists():
                for item in sorted(base_path.iterdir()):
                    if item.is_dir():
                        print(f"  {item.name}/")
        
        raise FileNotFoundError(
            f"Could not find audio directory for track={self.track}, split={self.split} "
            f"in {self.base_dir}\n"
            f"Expected: ASVspoof2021_{self.track}_{self.split}/ with flac/ subdirectory"
        )

    def _find_protocol_file(self):
        """Find the protocol file for the dataset split"""
        base_path = Path(self.base_dir)

        # Protocol files are usually in nested directories
        # Try to find ASVspoof2021.TRACK.cm.SPLIT.trl.txt

        # Check direct structure first
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if (f"ASVspoof2021.{self.track}.cm" in file and
                    self.split in file and
                    file.endswith(".txt")):
                    return os.path.join(root, file)

        raise FileNotFoundError(
            f"Could not find protocol file for track={self.track}, split={self.split}"
        )

    def _load_protocol(self):
        """Load protocol file and return list of (audio_id, label) tuples
        
        Supports two formats:
        1. Train/Dev: <speaker> <audio_id> <env> <label>
        2. Eval: <audio_id> (no label - will assign dummy label 0)
        """
        data = []
        with open(self.protocol_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue

                parts = line.split()
                if not parts:
                    continue
                
                # Determine format based on number of columns
                if len(parts) >= 4:
                    # Train/Dev format: speaker audio_id env label
                    audio_id = parts[1]
                    label_str = parts[-1].lower()
                    
                    # Label mapping
                    if label_str in ["spoof", "fake"]:
                        label = 1  # Spoof/Fake
                    elif label_str in ["bonafide", "genuine"]:
                        label = 0  # Bonafide/Genuine
                    else:
                        continue
                        
                elif len(parts) == 1:
                    # Eval format: just audio_id (no label)
                    # For eval data, we assign a dummy label of 0
                    audio_id = parts[0]
                    label = 0  # Dummy label for eval
                    
                    if self.verbose and len(data) == 0:
                        print(f"  Note: Loading eval data without ground truth labels")
                else:
                    continue

                data.append((audio_id, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id, label = self.data[idx]
        
        # Try to find the audio file
        audio_path = os.path.join(self.data_dir, f"{audio_id}.flac")
        
        if not os.path.exists(audio_path):
            # Try .wav extension
            audio_path = os.path.join(self.data_dir, f"{audio_id}.wav")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            waveform = torch.tensor(waveform, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            raise

        return waveform, label

    def get_statistics(self):
        """Return dataset statistics"""
        labels = [label for _, label in self.data]
        bonafide_count = sum(1 for label in labels if label == 0)
        spoof_count = sum(1 for label in labels if label == 1)
        total = len(self.data)

        return {
            "total": total,
            "bonafide": bonafide_count,
            "spoof": spoof_count,
            "bonafide_ratio": bonafide_count / total if total > 0 else 0,
            "spoof_ratio": spoof_count / total if total > 0 else 0
        }


if __name__ == "__main__":
    # Test the dataset loader
    try:
        dataset = ASVspoof2021Dataset(
            track="LA",
            split="eval",
            base_dir="data/raw/asvspoof2021"
        )
        print(f"Dataset size: {len(dataset)}")
        print(f"Statistics: {dataset.get_statistics()}")

        # Test loading a sample
        waveform, label = dataset[0]
        print(f"Sample shape: {waveform.shape}, Label: {label}")
    except Exception as e:
        print(f"Error: {e}")
