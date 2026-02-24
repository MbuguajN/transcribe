@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: 🎛️ Transcriber - Smart Launcher
color 0A
title Transcriber

:: 1. Instant Launch (If installed)
if exist "venv\Scripts\activate.bat" (
    echo Starting Transcriber...
    call venv\Scripts\activate.bat
    start /b pythonw app.py gui
    exit /b 0
)

:: 2. Setup Flow (Only runs first time)
cls
echo ============================================================
echo   Transcriber - One-Time Setup
echo ============================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/3] Creating environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing AI components...
python -m pip install --upgrade pip --quiet
echo.
echo Do you have an NVIDIA GPU? (y/N)
set /p gpuchoice=: 
if /I "%gpuchoice%"=="y" (
    python -m pip install --quiet faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn torch torchaudio
) else (
    python -m pip install --quiet faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)

echo [3/3] Finalizing setup...
:: Create simple desktop shortcut
set "SCRIPT_PATH=%~dp0Transcriber.bat"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Transcriber.lnk"
set "WORKING_DIR=%~dp0"
powershell -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%SCRIPT_PATH%';$s.WorkingDirectory='%WORKING_DIR%';$s.Save()" >nul 2>&1

echo.
echo ============================================================
echo   SUCCESS! Transcriber is ready.
echo   Launching now...
echo ============================================================
echo.
start /b pythonw app.py gui
pause
