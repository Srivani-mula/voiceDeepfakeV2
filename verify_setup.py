#!/usr/bin/env python
"""
Verification script for ASVspoof2021 dataset setup
Tests dataset loading and configuration
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    modules = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit',
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    return all_ok


def test_project_structure():
    """Test if project directories exist"""
    print("\n" + "=" * 60)
    print("Checking project structure...")
    print("=" * 60)
    
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/features",
        "src/train",
        "models",
    ]
    
    all_ok = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (MISSING)")
            all_ok = False
    
    return all_ok


def test_dataset_paths():
    """Test if dataset paths are accessible"""
    print("\n" + "=" * 60)
    print("Checking dataset paths...")
    print("=" * 60)
    
    paths_to_check = {
        "ASVspoof2019": "data/raw/asvspoof2019",
        "ASVspoof2021": "data/raw/asvspoof2021",
    }
    
    for name, path in paths_to_check.items():
        if Path(path).exists():
            num_files = len(list(Path(path).rglob("*")))
            print(f"✅ {name}: {path} ({num_files} files)")
        else:
            print(f"⚠️  {name}: {path} (not found)")
    
    return True


def test_asvspoof2021_loader():
    """Test ASVspoof2021 dataset loader"""
    print("\n" + "=" * 60)
    print("Testing ASVspoof2021 Dataset Loader...")
    print("=" * 60)
    
    try:
        from src.data.asvspoof2021_dataset import ASVspoof2021Dataset
        print("✅ ASVspoof2021Dataset imported successfully")
        
        # Try to load LA eval split (often smaller)
        try:
            print("\nAttempting to load ASVspoof2021 LA eval split...")
            dataset = ASVspoof2021Dataset(
                track="LA",
                split="eval",
                base_dir="data/raw/asvspoof2021",
                verbose=True
            )
            print(f"✅ Dataset loaded successfully")
            print(f"  Total samples: {len(dataset)}")
            
            stats = dataset.get_statistics()
            print(f"  Bonafide: {stats['bonafide']} ({stats['bonafide_ratio']*100:.1f}%)")
            print(f"  Spoof: {stats['spoof']} ({stats['spoof_ratio']*100:.1f}%)")
            
            # Try loading a sample
            print("\nTesting sample loading...")
            try:
                waveform, label = dataset[0]
                print(f"✅ Sample loaded")
                print(f"  Waveform shape: {waveform.shape}")
                print(f"  Label: {label} ({'Bonafide' if label == 0 else 'Spoof'})")
            except Exception as e:
                print(f"⚠️  Could not load sample: {e}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"⚠️  Could not load dataset: {e}")
            print("\nMake sure ASVspoof2021 is in: data/raw/asvspoof2021/")
            print("Download from: https://www.asvspoof.org/")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import ASVspoof2021Dataset: {e}")
        return False


def test_log_mel_extractor():
    """Test LogMelExtractor"""
    print("\n" + "=" * 60)
    print("Testing LogMelExtractor...")
    print("=" * 60)
    
    try:
        import torch
        from src.features.log_mel import LogMelExtractor
        
        print("✅ LogMelExtractor imported successfully")
        
        # Create extractor
        extractor = LogMelExtractor()
        print("✅ LogMelExtractor initialized")
        
        # Test with dummy waveform
        dummy_waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        output = extractor(dummy_waveform)
        
        print(f"✅ Feature extraction works")
        print(f"  Input shape: {dummy_waveform.shape}")
        print(f"  Output shape: {output.shape}")
        
        if output.shape == torch.Size([1, 1, 128, 128]):
            print("✅ Output shape is correct [1, 1, 128, 128]")
            return True
        else:
            print(f"⚠️  Output shape is {output.shape}, expected [1, 1, 128, 128]")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "=" * 60)
    print("Testing Model Loading...")
    print("=" * 60)
    
    try:
        import torch
        from src.models.custom_cnn import CustomCNN
        
        print("✅ CustomCNN imported successfully")
        
        # Create model
        model = CustomCNN(num_classes=2)
        print("✅ Model created successfully")
        
        # Check number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {num_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check if model weights available
        best_model_path = Path("models/best_model.pth")
        if best_model_path.exists():
            state = torch.load(str(best_model_path), map_location="cpu")
            model.load_state_dict(state)
            print(f"✅ Pre-trained model loaded: {best_model_path}")
            return True
        else:
            print(f"⚠️  Pre-trained model not found: {best_model_path}")
            print("   You can train a new model with:")
            print("   python src/train/train_asvspoof2021.py")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("Testing Configuration...")
    print("=" * 60)
    
    try:
        from config import (
            PROJECT_ROOT,
            ASVSPOOF2019_BASE,
            ASVSPOOF2021_BASE,
            TRAINING_CONFIG,
            MODEL_CONFIG,
            MODELS_DIR
        )
        
        print("✅ Config imported successfully")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  ASVspoof2019: {ASVSPOOF2019_BASE}")
        print(f"  ASVspoof2021: {ASVSPOOF2021_BASE}")
        print(f"  Models dir: {MODELS_DIR}")
        print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("🔍 Voice Deepfake Detection - System Verification")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Project Structure": test_project_structure(),
        "Dataset Paths": test_dataset_paths(),
        "Configuration": test_config(),
        "LogMelExtractor": test_log_mel_extractor(),
        "Model": test_model_loading(),
        "ASVspoof2021 Dataset": test_asvspoof2021_loader(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "⚠️  WARN"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! System is ready.")
        print("\nYou can now run:")
        print("  streamlit run app.py")
    else:
        print("⚠️  Some tests had warnings. Review above for details.")
        print("\nTo get started:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download dataset: https://www.asvspoof.org/")
        print("3. Extract to: data/raw/asvspoof2021/")
        print("4. Run: streamlit run app.py")
    
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
