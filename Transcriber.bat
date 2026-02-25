@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: 🎛️ Transcriber - Smart Launcher
color 0A
title Transcriber

:: 1. Instant Launch (Check if fully installed)
if exist "venv\Scripts\activate.bat" (
    echo Verifying dependencies...
    venv\Scripts\python.exe -c "import faster_whisper, ttkbootstrap, torch" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Starting Transcriber...
        call venv\Scripts\activate.bat
        start /b pythonw app.py gui
        exit /b 0
    )
    echo [NOTICE] Dependencies missing or corrupted. Running setup...
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

echo [2/3] Installing AI components (this may take a few mins)...
python -m pip install --upgrade pip
echo.
echo Do you have an NVIDIA GPU? (y/N)
set /p gpuchoice=: 
if /I "%gpuchoice%"=="y" (
    python -m pip install faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
) else (
    python -m pip install faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
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
