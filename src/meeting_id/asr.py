import whisperx
import logging
import torch

logger = logging.getLogger(__name__)

def load_model(device: str = "auto", compute_type: str = "float16", lang: str = "tr"):
    """
    Loads the WhisperX model.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fallback to int8 if cpu
    if device == "cpu":
        compute_type = "int8"

    logger.info(f"Loading WhisperX model on {device} with {compute_type}...")
    # Using large-v2 or large-v3 is common, but let's stick to a reasonable default or let whisperx decide.
    # WhisperX load_model defaults to large-v2 usually.
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=lang)
    return model, device

def transcribe(model, audio_np, batch_size=16):
    """
    Transcribes audio using the loaded model.
    """
    logger.info("Transcribing audio...")
    result = model.transcribe(audio_np, batch_size=batch_size)
    return result

def align(result, audio_np, device):
    """
    Aligns the transcription result.
    """
    logger.info("Aligning transcription...")
    # We need to load the alignment model. 
    # Note: This might re-download if not cached.
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_np, device, return_char_alignments=False)
    
    # Cleanup alignment model to save memory? 
    # In a script it's fine, but maybe we want to keep it if processing multiple files.
    # For now, we just return the result.
    return result_aligned
