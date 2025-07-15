#!/usr/bin/env python3
"""
Improved Linux Voice Assistant with Intent Classification
Fixes and improvements for better Linux compatibility and functionality.
"""

from __future__ import annotations
import re
import sys
import subprocess
import webbrowser
import shutil
import logging
from pathlib import Path
from typing import Literal, Final, Optional
import argparse
import signal
import os

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch/transformers not available. Using heuristic-only mode.")

# ---------------------------------------------------------------------------
# Configuration and Setup
# ---------------------------------------------------------------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_NAME: Final = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
Intent = Literal["open_browser", "play_music", "stop", "unknown"]

# Global variables for model (lazy loading)
tokenizer = None
model = None
device = None

# ---------------------------------------------------------------------------
# Model loading (lazy initialization)
# ---------------------------------------------------------------------------

def load_model():
    """Load the model and tokenizer if not already loaded."""
    global tokenizer, model, device
    
    if not HAS_TORCH:
        logger.warning("PyTorch not available, using heuristic-only classification")
        return False
    
    if model is not None:
        return True
    
    try:
        logger.info("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

DEFAULT_FEWSHOT = (
    "Command: open google chrome → open_browser\n"
    "Command: open the browser → open_browser\n"
    "Command: open firefox → open_browser\n"
    "Command: browse the web → open_browser\n"
    "Command: play Bohemian Rhapsody → play_music\n"
    "Command: play music Beatles Let It Be → play_music\n"
    "Command: play song Imagine → play_music\n"
    "Command: listen to music → play_music\n"
    "Command: stop the assistant → stop\n"
    "Command: quit → stop\n"
    "Command: exit → stop"
)

SYSTEM_MESSAGE = (
    "You are an intent classifier for a Linux voice assistant. "
    "Respond with *only* one label from the set: "
    "open_browser, play_music, stop, unknown."
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}\n<|im_end|>\n"
    "<|im_start|>user\n{fewshot}\nCommand: {cmd}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)

LABELS = {"open_browser", "play_music", "stop"}

def classify_intent(command: str) -> Intent:
    """Classify user intent using ML model or heuristics."""
    if not command.strip():
        return "unknown"
    
    # Try ML classification first
    if HAS_TORCH and load_model():
        try:
            return _ml_classify_intent(command)
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to heuristics")
    
    # Fallback to heuristic classification
    return heuristic_intent(command)

def _ml_classify_intent(command: str) -> Intent:
    """Use ML model for intent classification."""
    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_MESSAGE, 
        fewshot=DEFAULT_FEWSHOT, 
        cmd=command
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,  # Increased for better results
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    label = tokenizer.decode(
        output_ids[0, inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    ).strip().lower()
    
    # Clean up the label
    label = re.sub(r'[^a-z_]', '', label.replace(' ', '_'))
    
    if label in LABELS:
        return label  # type: ignore[return-value]
    
    # Fallback to heuristics if ML fails
    return heuristic_intent(command)

# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

KEYWORD_PATTERNS = [
    (re.compile(r'\b(play|music|song|listen|audio)\b', re.I), "play_music"),
    (re.compile(r'\b(open|launch|start).*\b(browser|chrome|firefox|web)\b', re.I), "open_browser"),
    (re.compile(r'\b(browse|web|internet|google)\b', re.I), "open_browser"),
    (re.compile(r'\b(stop|quit|exit|end|terminate)\b', re.I), "stop"),
]

def heuristic_intent(command: str) -> Intent:
    """Classify intent using keyword patterns."""
    command = command.strip()
    
    for pattern, intent in KEYWORD_PATTERNS:
        if pattern.search(command):
            return intent  # type: ignore[return-value]
    
    return "unknown"

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def check_dependency(command: str) -> bool:
    """Check if a system command is available."""
    return shutil.which(command) is not None

def open_browser() -> None:
    """Open a web browser."""
    logger.info("Opening browser...")
    
    # Try different browsers in order of preference
    browsers = ['firefox', 'google-chrome', 'chromium-browser', 'chromium']
    
    for browser in browsers:
        if check_dependency(browser):
            try:
                subprocess.Popen([browser], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"Opened {browser}")
                return
            except Exception as e:
                logger.warning(f"Failed to open {browser}: {e}")
    
    # Fallback to Python's webbrowser module
    try:
        webbrowser.open("about:blank")
        logger.info("Opened browser using webbrowser module")
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

def play_with_mpv(resource: str | Path) -> bool:
    """Play audio/video with mpv."""
    if not check_dependency("mpv"):
        logger.error("mpv not found. Please install mpv: sudo apt install mpv")
        return False
    
    try:
        subprocess.Popen([
            "mpv", 
            "--no-video", 
            "--really-quiet",
            str(resource)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Playing with mpv: {resource}")
        return True
    except Exception as e:
        logger.error(f"Failed to play with mpv: {e}")
        return False

def youtube_to_audio(query: str) -> Optional[Path]:
    """Download audio from YouTube using yt-dlp."""
    if not check_dependency("yt-dlp"):
        logger.error("yt-dlp not found. Please install: pip install yt-dlp")
        return None
    
    cache_dir = Path.home() / ".cache" / "intent_app" / "yt_audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean filename for better compatibility
    safe_query = re.sub(r'[^\w\s-]', '', query).strip()
    out_tpl = str(cache_dir / "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp", 
        "--no-playlist", 
        "--extract-audio", 
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "-o", out_tpl,
        f"ytsearch1:{query}",
    ]
    
    logger.info(f"Downloading audio for: '{query}'")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return None
        
        # Find the most recent mp3 file
        mp3_files = list(cache_dir.glob("*.mp3"))
        if mp3_files:
            latest = max(mp3_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Downloaded: {latest.name}")
            return latest
        else:
            logger.warning("No audio files found after download")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def handle_play(command: str) -> None:
    """Handle music/audio playback commands."""
    # Extract the track/query from the command
    remainder = re.sub(r'^\s*(?:play|music|song|listen(?:\s+to)?)\s*', '', command, flags=re.I).strip()
    
    if not remainder:
        logger.warning("No track specified")
        return
    
    # Check if it's a local file path
    path = Path(remainder).expanduser()
    if path.exists() and path.is_file():
        if play_with_mpv(path):
            return
    
    # Try to download from YouTube
    audio_file = youtube_to_audio(remainder)
    if audio_file:
        play_with_mpv(audio_file)
    else:
        logger.error("Could not find or play the requested audio")

def handle_stop() -> None:
    """Handle stop command."""
    logger.info("Stopping assistant...")
    
    # Try to stop any running mpv processes
    try:
        subprocess.run(["pkill", "-f", "mpv"], check=False)
        logger.info("Stopped media playback")
    except Exception as e:
        logger.debug(f"Could not stop media playback: {e}")
    
    sys.exit(0)

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def dispatch(intent: Intent, command: str) -> None:
    """Dispatch commands based on detected intent."""
    logger.info(f"Executing intent: {intent}")
    
    if intent == "open_browser":
        open_browser()
    elif intent == "play_music":
        handle_play(command)
    elif intent == "stop":
        handle_stop()
    else:
        logger.info(f"Unknown command: {command}")

# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info("Received interrupt signal, stopping...")
    handle_stop()

def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Linux Voice Assistant with Intent Classification"
    )
    parser.add_argument(
        "command", 
        nargs="+", 
        help="Voice command to execute"
    )
    parser.add_argument(
        "--no-ml", 
        action="store_true", 
        help="Use only heuristic classification (faster)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Join command arguments
    command = " ".join(args.command)
    
    # Force heuristic mode if requested
    if args.no_ml:
        global HAS_TORCH
        HAS_TORCH = False
    
    # Classify and execute
    intent = classify_intent(command)
    logger.info(f"Detected intent: {intent}")
    dispatch(intent, command)

if __name__ == "__main__":
    main()
