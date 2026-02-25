#!/usr/bin/env python
"""
Quick Start Script for Voice Deepfake Detection System
Verifies setup and runs the Streamlit app
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if all dependencies are installed"""
    required = [
        'torch',
        'torchaudio',
        'librosa',
        'numpy',
        'matplotlib',
        'streamlit',
    ]
    
    print("\nChecking dependencies...")
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")
    directories = [
        "data",
        "models",
        "src",
        ".streamlit",
        "temp"
    ]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ⚠️  {directory}/ (will be created if needed)")
    
    # Create temp directory if needed
    Path("temp").mkdir(exist_ok=True)
    
    return True


def check_dataset():
    """Check if dataset is present"""
    print("\nChecking datasets...")
    
    asvspoof2019 = Path("data/raw/asvspoof2019")
    asvspoof2021 = Path("data/raw/asvspoof2021")
    
    if asvspoof2019.exists():
        print(f"  ✅ ASVspoof2019 found")
    else:
        print(f"  ⚠️  ASVspoof2019 not found")
    
    if asvspoof2021.exists():
        print(f"  ✅ ASVspoof2021 found")
    else:
        print(f"  ⚠️  ASVspoof2021 not found (required for training)")
    
    return True


def check_models():
    """Check if pre-trained models exist"""
    print("\nChecking pre-trained models...")
    
    model_paths = [
        "models/best_model.pth",
        "models/cnn_asvspoof.pth",
    ]
    
    found = False
    for model in model_paths:
        if Path(model).exists():
            print(f"  ✅ {model}")
            found = True
        else:
            print(f"  ⚠️  {model}")
    
    if not found:
        print("\n⚠️  No pre-trained models found!")
        print("   You can train a new model with:")
        print("   python src/train/train_asvspoof2021.py --track LA --epochs 50")
    
    return True


def run_streamlit():
    """Launch Streamlit app"""
    print("\n" + "="*60)
    print("🚀 Launching Streamlit app...")
    print("="*60)
    print("\nThe app will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ])
    except KeyboardInterrupt:
        print("\n\nStreamlit stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    
    return True


def main():
    """Run all checks and launch app"""
    print("\n" + "="*60)
    print("Voice Deepfake Detection System - Quick Start")
    print("="*60 + "\n")
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Dataset", check_dataset),
        ("Models", check_models),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ All checks passed!")
        run_streamlit()
    else:
        print("⚠️  Some checks failed. Please review above.")
        print("\nQuick fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download datasets: https://www.asvspoof.org/")
        print("3. Train model: python src/train/train_asvspoof2021.py")
        print("\nFor more help, see DEPLOYMENT_GUIDE.md")


if __name__ == "__main__":
    main()
