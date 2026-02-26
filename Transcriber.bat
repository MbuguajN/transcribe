@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: 🎛️ Transcriber - Universal Portable Launcher
color 0A
title Transcriber

:: 1. SMART CHECK: Is the app ready to run?
if exist "venv\Scripts\python.exe" (
    :: Quick dependency check (headless)
    "venv\Scripts\python.exe" -c "import faster_whisper, ttkbootstrap, torch" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Starting Transcriber...
        
        :: 🚀 Launch with pythonw (no console) or python (fallback)
        if exist "venv\Scripts\pythonw.exe" (
            start "" "venv\Scripts\pythonw.exe" app.py gui
        ) else (
            start "" "venv\Scripts\python.exe" app.py gui
        )
        
        :: Short delay to ensure process starts
        timeout /t 2 /nobreak >nul
        exit /b 0
    )
    echo [NOTICE] Environment found but dependencies are missing. Repairing...
)

:: 2. AUTO-SETUP: Runs if venv is missing or broken
cls
echo ============================================================
echo   Transcriber - Universal Setup (Any Machine)
echo ============================================================
echo.
echo  This tool will now set up a private AI environment on this
echo  machine. No system-wide changes will be made.
echo.

:: Check for system Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on this system.
    echo.
    echo Please install Python 3.10 or newer to continue.
    echo Download: https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box 
    echo            "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: Ensure we are in the right directory
if not exist "app.py" (
    echo [ERROR] app.py not found in %CD%
    echo Please make sure you are running this from the app folder.
    pause
    exit /b 1
)

echo [1/3] Creating private environment (venv)...
if exist "venv" rd /s /q "venv"
python -m venv venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Failed to create environment.
    echo Try running as Administrator if this persists.
    pause
    exit /b 1
)

echo [2/3] Installing AI components (this may take 2-5 mins)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo.
echo Checking hardware...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [DETECTED] NVIDIA GPU found. Installing High-Speed mode...
    python -m pip install faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn --quiet
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 --quiet
) else (
    echo [DETECTED] No NVIDIA GPU. Installing Compatibility mode...
    python -m pip install faster-whisper pydub moviepy ttkbootstrap librosa scikit-learn --quiet
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)

echo [3/3] Finalizing portability...
:: Create/Refresh desktop shortcut
set "SCRIPT_PATH=%~dp0Transcriber.bat"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Transcriber.lnk"
set "WORKING_DIR=%~dp0"
powershell -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%SCRIPT_PATH%';$s.WorkingDirectory='%WORKING_DIR%';$s.IconLocation='%SCRIPT_PATH%';$s.Save()" >nul 2>&1

echo.
echo ============================================================
echo   SUCCESS! Transcriber is now optimized for this machine.
echo ============================================================
echo.
echo  - A shortcut has been added to your Desktop.
echo  - The app will now launch.
echo.

if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" app.py gui
) else (
    start "" "venv\Scripts\python.exe" app.py gui
)

timeout /t 3 /nobreak >nul
exit /b 0
