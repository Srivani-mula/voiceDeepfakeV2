@echo off
REM Voice Deepfake Detection - Windows Quick Start

echo.
echo ============================================================
echo Voice Deepfake Detection System - Windows Quick Start
echo ============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Python is installed
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo Installing/upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo Setup complete! Launching Streamlit app...
echo ============================================================
echo.
echo The app will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Launch Streamlit
streamlit run app.py

pause
