#!/bin/bash

# Voice Deepfake Detection - Linux/Mac Quick Start

echo ""
echo "============================================================"
echo "Voice Deepfake Detection System - Quick Start"
echo "============================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Setup complete! Launching Streamlit app..."
echo "============================================================"
echo ""
echo "The app will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run app.py
