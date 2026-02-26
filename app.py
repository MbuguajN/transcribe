#!/usr/bin/env python3
"""
Faster-Whisper Transcriber - Complete All-in-One Application
Everything you need for audio/video transcription with AI

Usage:
  python app.py setup              # Install & configure
  python app.py gui                # Launch GUI
  python app.py verify             # Check system
  python app.py transcribe audio.mp3
  python app.py transcribe folder/ -m large
  python app.py batch              # Run examples

Repository: https://github.com/example/faster-whisper-transcriber
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple, Union

# Try optional imports
try:
    import ttkbootstrap as ttk_bootstrap
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

try:
    from faster_whisper import WhisperModel
    import torch
    from pyannote.audio import Pipeline
    HAS_TRANSCRIBER_DEPS = True
except ImportError:
    HAS_TRANSCRIBER_DEPS = False

# Silence noisy torchcodec/library loading messages
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Could not load libtorchcodec")
warnings.filterwarnings("ignore", message="Could not load this library")

# ============================================================================
# TRANSCRIPTION ENGINE
# ============================================================================

class TranscriptionApp:
    """
    Core transcription engine using Faster-Whisper and Pyannote for diarization.
    """

    def __init__(
        self, 
        model_size: str = "base", 
        device: str = "cpu", 
        compute_type: str = "auto", 
        use_speaker_detection: bool = True,
        info_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize the transcription app.
        """
        if not HAS_TRANSCRIBER_DEPS:
            print("[ERROR] Required packages not installed.")
            print("Please run the 'Transcriber' launcher (.bat or .command) to complete setup.")
            sys.exit(1)

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_speaker_detection = use_speaker_detection
        self.info_callback = info_callback
        self.progress_callback = progress_callback
        self.diarization_pipeline = None

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {self.device}")

        # Compute type safety: float16 is NOT supported on most CPUs
        if self.compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "float32"
        
        if self.device == "cpu" and self.compute_type == "float16":
            print("Warning: float16 not supported on CPU. Falling back to float32.")
            self.compute_type = "float32"

        if self.info_callback: self.info_callback(f"Loading Whisper Engine ({self.device})...")
        print(f"Loading Faster-Whisper model: {model_size} ({self.device}/{self.compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root="./models"
        )
        print("Model loaded successfully!")

        if self.use_speaker_detection:
            self._init_diarization()

    def _init_diarization(self):
        """Initialize speaker diarization pipeline with fallback logic."""
        if self.info_callback: self.info_callback("Loading Diarization Model...")
        print("Loading speaker diarization model...")
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        kwargs = {"token": token} if token else {}
        
        try:
            # Attempt latest gated model
            model_name = "pyannote/speaker-diarization-3.1"
            self.diarization_pipeline = Pipeline.from_pretrained(model_name, **kwargs)
            if self.diarization_pipeline:
                self.diarization_pipeline.to(torch.device(self.device))
                print(f"Speaker diarization model {model_name} loaded!")
                return
        except Exception as e:
            if "401" in str(e):
                print("Note: Speaker diarization requires a Hugging Face token for best accuracy.")
                print("Visit https://hf.co/pyannote/speaker-diarization-3.1 to accept terms.")
            else:
                print(f"Warning: Primary diarization failed: {e}")

        try:
            # Fallback to older/public model
            print("Attempting fallback to public model...")
            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", **kwargs)
            if self.diarization_pipeline:
                self.diarization_pipeline.to(torch.device(self.device))
                print("Fallback diarization model loaded!")
                return
        except Exception as e:
            if "401" in str(e):
                print("Note: Public diarization model also requires a token/access.")
            else:
                print(f"Fallback diarization failed: {e}")
            
        print("--- Using offline local AI for speaker detection ---")


    def extract_audio_from_mp4(self, mp4_path: str, output_wav: str) -> bool:
        """Extract audio from MP4 file"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", mp4_path, "-q:a", "9", "-n", output_wav],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✓ Extracted audio from {mp4_path}")
                return True
            else:
                print("ffmpeg not found, using moviepy...")
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(mp4_path)
                video.audio.write_audiofile(output_wav, verbose=False, logger=None)
                print(f"✓ Extracted audio from {mp4_path}")
                return True
        except Exception as e:
            print(f"✗ Error extracting audio: {e}")
            return False

    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp from seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def transcribe(self, audio_path: str, language: str = None):
        """Transcribe audio file with progress reporting"""
        print(f"\nTranscribing: {audio_path}")
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=False
        )
        
        full_segments = []
        duration = info.duration
        
        for segment in segments_gen:
            full_segments.append(segment)
            if self.progress_callback and duration > 0:
                percent = min(100, int((segment.end / duration) * 100))
                self.progress_callback(percent)
        
        if self.progress_callback:
            self.progress_callback(100)
            
        return full_segments, info

    def generate_sentence_timestamps(self, segments):
        """Generate sentence-level timestamps"""
        sentences_with_time = []
        current_sentence = ""
        sentence_start = None
        sentence_end = None

        for segment in segments:
            text = segment.text.strip()
            start_time = segment.start
            end_time = segment.end

            if sentence_start is None:
                sentence_start = start_time
            current_sentence += " " + text if current_sentence else text
            sentence_end = end_time

            if text and text[-1] in '.!?':
                sentences_with_time.append((
                    current_sentence.strip(),
                    self.format_timestamp(sentence_start),
                    self.format_timestamp(sentence_end)
                ))
                current_sentence = ""
                sentence_start = None
                sentence_end = None

        if current_sentence.strip():
            sentences_with_time.append((
                current_sentence.strip(),
                self.format_timestamp(sentence_start),
                self.format_timestamp(sentence_end)
            ))

        return sentences_with_time

    def get_speaker_diarization(self, audio_path: str) -> Dict[float, Dict]:
        """Get speaker diarization using pipeline or naive offline method."""
        # Try pipeline if available
        if self.diarization_pipeline is not None:
            try:
                diarization = self.diarization_pipeline(audio_path)
                speaker_segments = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments[turn.start] = {
                        'start': turn.start, 'end': turn.end, 'speaker': speaker
                    }
                return speaker_segments
            except Exception as e:
                print(f"Warning: pipeline diarization failed: {e}")

        # Fallback naive clustering
        print("Using offline naive speaker detection...")
        try:
            import librosa
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError:
            print("Naive diarization requires librosa and scikit-learn; please install them.")
            return {}

        try:
            y, sr = librosa.load(audio_path, sr=None)
            hop = int(sr * 0.5)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop).T
            
            if len(mfcc) == 0:
                return {}
                
            k = min(2, len(mfcc))
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(mfcc)
            
            speaker_segments = {}
            for i, label in enumerate(kmeans.labels_):
                start = i * 0.5
                end = (i + 1) * 0.5
                speaker_segments[start] = {
                    'start': start, 
                    'end': end, 
                    'speaker': f'SPEAKER_{label:02d}'
                }
            return speaker_segments
        except Exception as e:
            print(f"Naive clustering failed: {e}")
            return {}

    def get_speaker_for_timestamp(self, timestamp: float, speaker_segments: Optional[Dict[float, Dict]]) -> Optional[str]:
        """Get standardized speaker label for a specific timestamp."""
        if not speaker_segments:
            return None
            
        # Find the segment that contains this timestamp
        # Pyannote labels are usually SPEAKER_00, SPEAKER_01, etc.
        # Fallback labels are also SPEAKER_00, SPEAKER_01
        
        best_match = None
        for seg_start, segment in speaker_segments.items():
            if segment['start'] <= timestamp <= segment['end']:
                best_match = segment['speaker']
                break
        
        if not best_match:
            # Try to find the closest segment if no exact match
            distances = [(abs(s['start'] - timestamp), s['speaker']) for s in speaker_segments.values()]
            if distances:
                best_match = min(distances, key=lambda x: x[0])[1]

        if best_match:
            # Standardize label to "Speaker 1", "Speaker 2", etc.
            try:
                # Handle SPEAKER_XX format
                if '_' in best_match:
                    num = int(best_match.split('_')[-1]) + 1
                    return f"Speaker {num}"
                # Handle "Speaker X" format
                match = re.search(r'\d+', best_match)
                if match:
                    return f"Speaker {int(match.group())}"
            except:
                pass
            return best_match
            
        return None

    def save_transcript(self, output_path: str, sentences_with_time: List[Tuple], speaker_segments: Optional[Dict[float, Dict]] = None) -> bool:
        """Save transcript with timestamps."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in sentences_with_time:
                    if len(item) == 3:
                        sentence, start_time, end_time = item
                        start_seconds = self._time_to_seconds(start_time)
                        speaker = self.get_speaker_for_timestamp(start_seconds, speaker_segments) if speaker_segments else None
                        f.write(f"[{start_time} --> {end_time}]\n")
                        if speaker:
                            f.write(f"{speaker}: {sentence}\n\n")
                        else:
                            f.write(f"{sentence}\n\n")
            print(f"✓ Transcript saved to: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving transcript: {e}")
            return False

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert timestamp to seconds"""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0

    def process_file(self, file_path: str, output_dir: str = None, language: str = None) -> bool:
        """Process a single file"""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"✗ File not found: {file_path}")
            return False

        if output_dir is None:
            output_dir = file_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = None
        temp_file = None

        if file_path.suffix.lower() == '.mp4':
            temp_file = output_dir / f"{file_path.stem}_temp_audio.wav"
            if self.extract_audio_from_mp4(str(file_path), str(temp_file)):
                audio_path = temp_file
            else:
                return False
        elif file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']:
            audio_path = file_path
        else:
            print(f"✗ Unsupported file format: {file_path.suffix}")
            return False

        try:
            segments, info = self.transcribe(str(audio_path), language)
            sentences_with_time = self.generate_sentence_timestamps(segments)
            speaker_segments = None
            if self.use_speaker_detection:
                print("Detecting speakers...")
                speaker_segments = self.get_speaker_diarization(str(audio_path))
                if speaker_segments:
                    print(f"  Found speakers in audio")

            output_path = output_dir / f"{file_path.stem}_transcript.txt"
            success = self.save_transcript(str(output_path), sentences_with_time, speaker_segments)

            if success:
                print(f"✓ Processed: {file_path.name}")
                print(f"  Language: {info.language}")
                print(f"  Sentences: {len(sentences_with_time)}")
                if speaker_segments:
                    print(f"  Speakers: {len(set(seg['speaker'] for seg in speaker_segments.values()))}")
            return success
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()

    def process_directory(self, directory: str, output_dir: str = None, language: str = None):
        """Process all files in directory"""
        directory = Path(directory)
        if not directory.is_dir():
            print(f"✗ Directory not found: {directory}")
            return

        supported_formats = {'.mp3', '.mp4', '.wav', '.m4a', '.flac'}
        files = [f for f in directory.iterdir() if f.suffix.lower() in supported_formats]

        if not files:
            print(f"✗ No supported files found in: {directory}")
            return

        print(f"\nFound {len(files)} file(s) to process\n")
        successful = 0
        failed = 0

        for file_path in sorted(files):
            if self.process_file(str(file_path), output_dir, language):
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*50}")
        print(f"Complete! Successful: {successful}, Failed: {failed}")
        print(f"{'='*50}")


# ============================================================================
# GUI
# ============================================================================

class TranscriberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcriber")
        self.root.geometry("450x500")
        self.root.resizable(False, False)

        # Build UI
        self.bg_color = "#FFFFFF"
        self.root.configure(bg=self.bg_color)
        
        self.file_path = tk.StringVar(value="None")
        self.device_var = tk.StringVar(value="cpu")
        self.progress_var = tk.StringVar(value="Ready")
        self.is_processing = False
        self.start_btn = None
        self.progress_bar = None

        self._setup_ui()

    def _setup_ui(self):
        main = tk.Frame(self.root, bg=self.bg_color, padx=25, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="AI Transcriber", font=('Segoe UI', 16, 'bold'), bg=self.bg_color).pack(pady=(0, 10))

        # File selection
        f_frame = tk.LabelFrame(main, text=" 📁 Source File ", bg=self.bg_color, font=('Segoe UI', 9))
        f_frame.pack(fill=tk.X, pady=5)
        tk.Label(f_frame, textvariable=self.file_path, font=('Segoe UI', 9), bg=self.bg_color, fg="#555", wraplength=350).pack(padx=10, pady=2)
        tk.Button(f_frame, text="Select File", command=self._select_file, bg="#eee", relief=tk.GROOVE).pack(pady=5)

        # Device selection
        d_frame = tk.LabelFrame(main, text=" ⚙️ Processing Device ", bg=self.bg_color, font=('Segoe UI', 9))
        d_frame.pack(fill=tk.X, pady=5)
        tk.Radiobutton(d_frame, text="CPU (Safe)", variable=self.device_var, value="cpu", bg=self.bg_color).pack(side=tk.LEFT, padx=20, pady=2)
        tk.Radiobutton(d_frame, text="GPU (Fast)", variable=self.device_var, value="cuda", bg=self.bg_color).pack(side=tk.LEFT, padx=20, pady=2)

        # Progress
        tk.Label(main, textvariable=self.progress_var, font=('Segoe UI', 9, 'italic'), bg=self.bg_color, fg="#888").pack(pady=5)
        self.progress_bar = ttk.Progressbar(main, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=2)

        # Start button
        self.start_btn = tk.Button(main, text="Start Transcription", command=self._start, font=('Segoe UI', 10, 'bold'), bg="#3B82F6", fg="white", relief=tk.FLAT, pady=12)
        self.start_btn.pack(fill=tk.X, pady=15)

    def _select_file(self):
        f = filedialog.askopenfilename(filetypes=[("Media Files", "*.mp3 *.mp4 *.wav *.m4a *.flac"), ("All", "*.*")])
        if f: self.file_path.set(f)

    def _start(self):
        if self.file_path.get() == "None":
            messagebox.showerror("Error", "Please select a file")
            return
        if self.is_processing: return

        self.is_processing = True
        self.start_btn.config(state="disabled", text="Processing...")
        self.progress_bar.start()
        
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.progress_var.set("Initializing AI...")
            
            # Use info_callback to update GUI status during long init steps
            def status_update(msg):
                self.root.after(0, lambda: self.progress_var.set(msg))

            # Use progress_callback to update progress bar and percentage
            def progress_update(percent):
                def _update():
                    self.progress_bar['value'] = percent
                    self.progress_var.set(f"Transcribing... ({percent}%)")
                self.root.after(0, _update)

            # GPU fallback: check CUDA availability before attempting GPU mode
            selected_device = self.device_var.get()
            if selected_device == "cuda":
                try:
                    import torch as _torch
                    if not _torch.cuda.is_available():
                        self.root.after(0, lambda: messagebox.showwarning(
                            "GPU Unavailable",
                            "CUDA is not available on this system.\n\n"
                            "Possible reasons:\n"
                            "• No NVIDIA GPU installed\n"
                            "• NVIDIA drivers not installed\n"
                            "• PyTorch CPU-only version installed\n\n"
                            "Falling back to CPU mode."
                        ))
                        selected_device = "cpu"
                        self.root.after(0, lambda: self.device_var.set("cpu"))
                except ImportError:
                    selected_device = "cpu"

            app = TranscriptionApp(
                model_size="base", 
                device=selected_device, 
                use_speaker_detection=True,
                info_callback=status_update,
                progress_callback=progress_update
            )
            
            p = Path(self.file_path.get())
            app.process_file(str(p))
            
            self.progress_var.set("Done!")
            messagebox.showinfo("Success", "Transcription complete!")
        except Exception as e:
            self.progress_var.set("Error occurred")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_processing = False
            self.start_btn.config(state="normal", text="Start Transcription")
            self.progress_bar.config(mode='indeterminate', value=0)
            self.root.update()



# ============================================================================
# SETUP & UTILITIES
# ============================================================================

def create_virtualenv():
    """Create virtual environment"""
    if os.path.isdir("venv"):
        print("✓ Virtual environment exists")
        return True

    print("Creating virtual environment...")
    import venv
    try:
        venv.create("venv", with_pip=True)
        print("✓ Done")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def install_packages():
    """Install dependencies, asking GPU/CPU preference"""
    pip_exe = os.path.join("venv", "Scripts" if os.name == "nt" else "bin", "pip")
    if not os.path.isfile(pip_exe):
        print("✗ pip not found")
        return False

    print("Installing packages...")
    subprocess.run([pip_exe, "install", "--upgrade", "pip"])

    choice = input("Do you have a CUDA-capable NVIDIA GPU? (y/N): ").strip().lower()
    if choice == 'y':
        print("[SETUP] Installing dependencies with GPU support...")
        subprocess.run([pip_exe, "install", "--quiet", "faster-whisper", "pydub", "moviepy", "ttkbootstrap", "librosa", "scikit-learn"])
        subprocess.run([pip_exe, "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126", "--quiet"])
    else:
        print("[SETUP] Installing CPU-only requirements...")
        subprocess.run([pip_exe, "install", "--quiet", "faster-whisper", "pydub", "moviepy", "ttkbootstrap", "librosa", "scikit-learn"])
        subprocess.run([pip_exe, "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])

    print("✓ Done")
    return True


def setup_environment():
    """Full setup"""
    print("\n" + "="*60)
    print("  Faster-Whisper Transcriber Setup")
    print("="*60 + "\n")

    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"✗ Python 3.10+ required (you have {major}.{minor})")
        return

    if not create_virtualenv():
        return

    if not install_packages():
        return

    print("\n" + "="*60)
    print("  ✓ Setup Complete!")
    print("="*60)
    print("\nRun:")
    print("  python app.py gui              # Launch GUI")
    print("  python app.py transcribe file.mp3   # CLI")


def verify_setup():
    """Verify installation"""
    print("\n" + "="*60)
    print("  System Verification")
    print("="*60)

    # Python
    version = sys.version.split()[0]
    print(f"\nPython: {version} [OK]")

    # Packages
    print("\nPackages:")
    for module, name in [("faster_whisper", "Faster-Whisper"), ("torch", "PyTorch"), ("moviepy", "MoviePy")]:
        try:
            __import__(module)
            print(f"  [OK]   {name}")
        except:
            print(f"  [FAIL] {name}")

    # CUDA
    print("\nHardware:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [OK]   CUDA available: {gpu_name} ({vram:.1f} GB VRAM)")
            print(f"  [OK]   GPU mode: READY")
        else:
            cuda_tag = "+cu" in torch.__version__
            print(f"  [WARN] CUDA not available (GPU mode will fall back to CPU)")
            if cuda_tag:
                print(f"         PyTorch has CUDA support built-in, but no GPU was detected.")
                print(f"         Check: NVIDIA drivers installed? GPU present?")
            else:
                print(f"         PyTorch is CPU-only. To enable GPU, reinstall with CUDA support.")
            print(f"  [OK]   CPU mode: READY")
    except Exception as e:
        print(f"  [WARN] Could not check CUDA: {e}")


def launch_gui():
    """Launch GUI"""
    try:
        if HAS_BOOTSTRAP:
            try:
                root = ttk_bootstrap.Window(themename="darkly")
            except Exception as e:
                print(f"Warning: ttkbootstrap failed, using standard Tk: {e}")
                root = tk.Tk()
        else:
            root = tk.Tk()

        gui = TranscriberGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Faster-Whisper: Transcribe audio/video with AI",
        epilog="Examples:\n  python app.py setup\n  python app.py gui\n  python app.py transcribe audio.mp3\n  python app.py transcribe folder/ -m large",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup", help="Install & configure environment")
    subparsers.add_parser("verify", help="Check system & dependencies")
    subparsers.add_parser("gui", help="Launch graphical interface")

    tx = subparsers.add_parser("transcribe", help="Transcribe audio/video")
    tx.add_argument("path", help="File or directory")
    tx.add_argument("-o", "--output", help="Output directory", default=None)
    tx.add_argument("-m", "--model", choices=["tiny", "base", "small", "medium", "large"], default="base")
    tx.add_argument("-l", "--language", default=None)
    tx.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cpu")
    tx.add_argument("--compute-type", choices=["int8", "float16", "float32", "auto"], default="auto")
    tx.set_defaults(use_speaker_detection=True)

    args = parser.parse_args()

    if args.command == "setup":
        setup_environment()
    elif args.command == "verify":
        verify_setup()
    elif args.command == "gui":
        launch_gui()
    elif args.command == "transcribe":
        if not HAS_TRANSCRIBER_DEPS:
            print("✗ Dependencies not installed. Run: python app.py setup")
            sys.exit(1)

        try:
            transcriber = TranscriptionApp(
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
                use_speaker_detection=args.use_speaker_detection
            )
        except Exception as e:
            print(f"✗ Failed: {e}")
            sys.exit(1)

        path = Path(args.path)
        if path.is_dir():
            transcriber.process_directory(str(path), args.output, args.language)
        elif path.is_file():
            transcriber.process_file(str(path), args.output, args.language)
        else:
            print(f"✗ Not found: {args.path}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
