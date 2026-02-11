import whisperx
import logging
import os

logger = logging.getLogger(__name__)

def diarize(audio_np, device: str, hf_token: str = None, min_speakers=None, max_speakers=None):
    """
    Performs speaker diarization.
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
    if not hf_token:
        logger.warning("HF_TOKEN not found. Diarization might fail if models are not cached.")
        # We proceed, maybe it's cached.

    logger.info(f"Loading Diarization pipeline on {device}...")
    # Fix for AttributeError: module 'whisperx' has no attribute 'DiarizationPipeline'
    # It seems in some versions it's under whisperx.diarize
    try:
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    except ImportError:
        # Fallback if it is top level (just in case)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    logger.info("Diarizing...")
    diarize_segments = diarize_model(audio_np, min_speakers=min_speakers, max_speakers=max_speakers)
    
    return diarize_segments

def assign_speakers(diarize_segments, transcript_result):
    """
    Assigns speakers to the transcript.
    """
    logger.info("Assigning speakers to words...")
    result = whisperx.assign_word_speakers(diarize_segments, transcript_result)
    return result
