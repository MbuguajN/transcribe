# 🎤 AI Transcriber (Portable Edition)

**Transcribe any audio or video file into text with high-accuracy AI—instantly and locally.**

This application is designed to be **Zero-Setup**. It is a portable "Smart App" that handles its own dependencies, so you can move the folder anywhere and it just works.

---

## 🚀 Quick Start

To start transcribing, just run the launcher for your computer:

- **🪟 Windows**: Double-click `Transcriber.bat`
- **🍎 Mac / Linux**: Double-click `Transcriber.command`

### 💡 How it works:
1.  **First Run**: The app will intelligently set up a private AI environment and download the necessary models (~300MB). This only happens once.
2.  **Everyday Use**: After the first run, the app opens instantly.
3.  **Desktop Shortcut**: The Windows installer automatically puts a "Transcriber" shortcut on your Desktop for easy access.

---

## ✨ Features

- 🎯 **High Accuracy** - Powered by OpenAI's Whisper (Faster-Whisper), one of the world's most accurate AI speech models.
- 👥 **Speaker Identification** - Automatically detects who said what (Speaker 1, Speaker 2, etc.).
- 🌐 **Privacy First** - Everything stays on your computer. Nothing is ever uploaded to the cloud.
- 🎬 **Video & Audio** - Supports MP4, MP3, WAV, AVI, MOV, MKV, and more.
- 💬 **99+ Languages** - Auto-detects the language or allows you to specify one.

---

## ⚙️ Pro Tips for Better Results

### Better Speaker Identification
By default, the app uses a basic local AI to guess speakers. For professional-grade accuracy:
1.  Visit [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the terms.
2.  Get a free token from [Hugging Face](https://hf.co/settings/tokens).
3.  Set your token as an environment variable (`HUGGINGFACE_TOKEN`) and relaunch. The app will automatically upgrade itself.

### Speed vs. Accuracy
Choose **GPU (Fast)** if you have an NVIDIA card. If the app closes unexpectedly, switch back to **CPU (Safe)**—it's slower but works on every computer.

---

## 🛠️ System Requirements

- **Python:** 3.10 or higher must be installed on your computer.
- **Storage:** ~2-4GB of space for the AI models and library files.
- **Internet:** Only needed the very first time you run a specific model.

---

## 🔧 Troubleshooting

- **"Could not load libtorchcodec" Tracebacks**: You might see long technical errors in the console window. **Ignore them.** They are standard warnings from the AI engine that do NOT affect your transcription.
- **App window is cut off**: The app is designed for modern screens. If the "Start" button is hidden, check your computer's display scaling or try resizing the window manually.
- **Stuck on "Initializing"**: The first run takes a few minutes to download the AI models. Be patient—it's worth it!

---

**License:** MIT - Free to use, share, and modify.
