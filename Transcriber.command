#!/bin/bash
cd "$(dirname "$0")"

# 🎛️ Transcriber - Smart Launcher

# 1. Instant Launch (If installed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    python3 app.py gui
    exit 0
fi

# 2. Setup Flow (Only runs first time)
clear
echo "============================================================"
echo "  Transcriber - One-Time Setup"
echo "============================================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "[1/3] Creating environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/3] Installing AI components..."
python -m pip install --upgrade pip --quiet

echo ""
read -p "Do you have an NVIDIA GPU with CUDA? (y/N): " gpuchoice
if [[ "$gpuchoice" =~ ^[Yy]$ ]]; then
    python -m pip install --quiet faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn torch torchaudio
else
    python -m pip install --quiet faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo "[3/3] Finalizing setup..."
chmod +x Transcriber.command
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp Transcriber.command ~/Desktop/Transcriber.command > /dev/null 2>&1
    echo "[OK] Created launcher on your Desktop"
fi

echo ""
echo "============================================================"
echo "  SUCCESS! Transcriber is ready."
echo "  Launching now..."
echo "============================================================"
echo ""
python3 app.py gui
read -p "Press Enter to close..."
