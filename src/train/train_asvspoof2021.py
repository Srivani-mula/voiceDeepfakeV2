"""
Training script for ASVspoof2021 dataset
Supports LA, PA, and DF tracks
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import json
from datetime import datetime

from src.data.asvspoof2021_dataset import ASVspoof2021Dataset
from src.data.asvspoof2019_feature_dataset import ASVspoof2019FeatureDataset
from src.models.custom_cnn import CustomCNN
from src.features.log_mel import LogMelExtractor
from config import MODELS_DIR, TRAINING_CONFIG


class FeatureDatasetWrapper:
    """Wrap raw waveforms to extract features on-the-fly"""
    
    def __init__(self, dataset, extractor):
        self.dataset = dataset
        self.extractor = extractor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        
        # Ensure waveform has shape [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract log-mel spectrogram
        x = self.extractor(waveform)  # [1, 1, 128, 128]
        
        # Remove the batch dimension that LogMelExtractor adds
        # We want [1, 128, 128] so DataLoader can batch it
        x = x.squeeze(0)  # [1, 128, 128]
        
        return x, label


def train_asvspoof2021(
    track="LA",
    split="train",
    batch_size=32,
    epochs=50,
    lr=0.001,
    weight_decay=0,
    device="cuda",
    save_best=True,
    model_save_path=None
):
    """
    Train a CustomCNN model on ASVspoof2021 dataset
    
    Args:
        track: Dataset track ('LA', 'PA', or 'DF')
        split: Training split ('train', 'dev', or 'eval')
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        device: Device to use ('cuda' or 'cpu')
        save_best: Whether to save the best model
        model_save_path: Path to save the model
    """
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print(f"Loading ASVspoof2021 {track} {split} split...")
    try:
        dataset = ASVspoof2021Dataset(
            track=track,
            split=split,
            base_dir="data/raw/asvspoof2021",
            sample_rate=16000,
            verbose=True
        )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nMake sure the ASVspoof2021 dataset is properly placed:")
        print("  Expected structure:")
        print("    data/raw/asvspoof2021/ASVspoof2021_LA_{train,dev,eval}/")
        print("      └── ASVspoof2021_LA_{train,dev,eval}/")
        print("          ├── flac/  (audio files)")
        print("          └── *.txt  (protocol files)")
        print("\nDownload from: https://www.asvspoof.org/index2021.html")
        return None, None
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Bonafide: {stats['bonafide']} ({stats['bonafide_ratio']*100:.1f}%)")
    print(f"  Spoof: {stats['spoof']} ({stats['spoof_ratio']*100:.1f}%)")
    
    # Create feature extractor
    extractor = LogMelExtractor(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=128
    )
    
    # Wrap dataset to extract features
    feature_dataset = FeatureDatasetWrapper(dataset, extractor)
    
    # Create data loader
    train_loader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    print(f"Data loader created with {len(train_loader)} batches")
    
    # Initialize model
    model = CustomCNN(num_classes=2).to(device)
    print(f"Model initialized: CustomCNN")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_loss = float('inf')
    history = {
        'epochs': [],
        'losses': [],
        'learning_rates': []
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Learning rate: {lr}, Batch size: {batch_size}")
    print("-" * 60)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)  # [B, 1, 128, 128]
            y = y.to(device)  # [B]
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Progress
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['epochs'].append(epoch + 1)
        history['losses'].append(avg_loss)
        history['learning_rates'].append(current_lr)
        
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = model_save_path or MODELS_DIR / f"best_model_asvspoof2021_{track}.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print(f"✓ Best model saved: {best_model_path} (loss: {avg_loss:.4f})")
    
    print("-" * 60)
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    
    # Save training history
    history_path = MODELS_DIR / f"training_history_{track}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    # Save final model
    final_model_path = MODELS_DIR / f"asvspoof2021_{track}_final.pth"
    torch.save(model.state_dict(), str(final_model_path))
    print(f"Final model saved: {final_model_path}")
    
    return model, history


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Train CustomCNN on ASVspoof2021 dataset"
    )
    
    parser.add_argument(
        "--track",
        type=str,
        choices=["LA", "PA", "DF"],
        default="LA",
        help="ASVspoof2021 track (default: LA)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="eval",
        help="Dataset split (default: eval)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="Weight decay (default: 0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the best model"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the model"
    )
    
    args = parser.parse_args()
    
    # Train
    result = train_asvspoof2021(
        track=args.track,
        split=args.split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_best=not args.no_save,
        model_save_path=args.save_path
    )
    
    if result[0] is None:
        print("\n❌ Training failed. Check the errors above.")
        sys.exit(1)
    
    model, history = result


if __name__ == "__main__":
    main()
