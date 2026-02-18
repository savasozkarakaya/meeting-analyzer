import subprocess
import tempfile
import os
import logging
import soundfile as sf
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts input audio to 16kHz mono PCM WAV using ffmpeg.
    Returns the path to the converted file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        # Create a temp file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # ffmpeg command: -y (overwrite), -i input, -ar 16000 (rate), -ac 1 (mono), -c:a pcm_s16le (codec)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    logger.info(f"Running ffmpeg: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        # Best-effort cleanup of temp output file if conversion failed.
        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        raise RuntimeError(f"FFmpeg failed to convert {input_path}") from e
    except FileNotFoundError:
        # Best-effort cleanup of temp output file if conversion failed.
        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        raise RuntimeError("FFmpeg not found. Please install ffmpeg and add it to PATH.")

    return output_path

def load_audio(path: str) -> np.ndarray:
    """
    Loads audio from a file.
    Returns numpy array of float32.
    """
    # We assume the file is already converted to wav 16k mono by convert_to_wav if needed.
    # But soundfile can read many formats. 
    # However, for consistency with the pipeline, we usually work with the converted wav.
    
    data, samplerate = sf.read(path)
    
    # Ensure float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    return data

@contextmanager
def ensure_wav_16k_mono(input_path: str):
    """
    Context manager that yields a 16kHz mono WAV path.
    If a temporary converted file is created, it is cleaned up automatically.
    """
    wav_path = None
    try:
        wav_path = convert_to_wav(input_path)
        yield wav_path
    finally:
        # If convert_to_wav created a temp file, it should not be the same as input_path.
        # Best-effort cleanup.
        try:
            if wav_path and wav_path != input_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp wav '{wav_path}': {e}")
