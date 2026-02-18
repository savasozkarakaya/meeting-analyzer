import whisperx
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def _detect_vad_regions(
    audio_np,
    sample_rate: int = 16000,
    frame_sec: float = 0.03,
    min_speech_sec: float = 0.4,
    max_silence_sec: float = 0.35,
    padding_sec: float = 0.15,
):
    """
    Lightweight energy-based VAD region detector.
    Returns list of (start_sec, end_sec) speech regions.
    """
    if audio_np is None or len(audio_np) == 0:
        return []

    frame_size = max(1, int(sample_rate * frame_sec))
    hop = frame_size
    n_frames = int(np.ceil(len(audio_np) / hop))

    energies = []
    for i in range(n_frames):
        start = i * hop
        end = min(len(audio_np), start + frame_size)
        frame = audio_np[start:end]
        if len(frame) == 0:
            energies.append(0.0)
            continue
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        energies.append(rms)

    energies = np.asarray(energies, dtype=np.float64)
    nonzero = energies[energies > 0]
    if nonzero.size == 0:
        return []

    # Adaptive threshold with lower-bound for quiet recordings.
    adaptive_thr = max(float(np.percentile(nonzero, 25)) * 0.6, 1e-4)
    speech_mask = energies >= adaptive_thr

    min_speech_frames = max(1, int(min_speech_sec / frame_sec))
    max_silence_frames = max(1, int(max_silence_sec / frame_sec))
    padding_frames = max(0, int(padding_sec / frame_sec))

    regions = []
    i = 0
    while i < n_frames:
        if not speech_mask[i]:
            i += 1
            continue

        start = i
        silence_run = 0
        i += 1
        while i < n_frames:
            if speech_mask[i]:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run > max_silence_frames:
                    break
            i += 1

        end = i - silence_run
        if end - start >= min_speech_frames:
            start = max(0, start - padding_frames)
            end = min(n_frames, end + padding_frames)
            regions.append((start * frame_sec, end * frame_sec))

    if not regions:
        return []

    # Merge overlapping or near-overlapping regions after padding.
    merged = [regions[0]]
    for cur_start, cur_end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged

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

def transcribe(
    model,
    audio_np,
    batch_size: int = 16,
    vad_presegment: bool = False,
    vad_min_speech_sec: float = 0.4,
    vad_max_silence_sec: float = 0.35,
    vad_padding_sec: float = 0.15,
):
    """
    Transcribes audio using the loaded model.
    """
    logger.info("Transcribing audio...")
    if not vad_presegment:
        return model.transcribe(audio_np, batch_size=batch_size)

    regions = _detect_vad_regions(
        audio_np,
        sample_rate=16000,
        min_speech_sec=vad_min_speech_sec,
        max_silence_sec=vad_max_silence_sec,
        padding_sec=vad_padding_sec,
    )
    if not regions:
        logger.info("VAD pre-segmentation found no speech chunks; falling back to full-audio transcription.")
        return model.transcribe(audio_np, batch_size=batch_size)

    logger.info(f"VAD pre-segmentation enabled: {len(regions)} chunks will be transcribed.")
    merged_segments = []
    language = None
    seg_id = 0

    for idx, (start_sec, end_sec) in enumerate(regions):
        start_sample = max(0, int(start_sec * 16000))
        end_sample = min(len(audio_np), int(end_sec * 16000))
        if end_sample <= start_sample:
            continue

        chunk = audio_np[start_sample:end_sample]
        if len(chunk) < 1600:
            continue

        logger.debug(
            "Transcribing VAD chunk %s/%s (%.2fs-%.2fs)",
            idx + 1,
            len(regions),
            start_sec,
            end_sec,
        )
        chunk_result = model.transcribe(chunk, batch_size=batch_size)
        if language is None:
            language = chunk_result.get("language")

        for seg in chunk_result.get("segments", []):
            seg_copy = dict(seg)
            seg_copy["id"] = seg_id
            seg_id += 1
            seg_copy["start"] = float(seg.get("start", 0.0) + start_sec)
            seg_copy["end"] = float(seg.get("end", 0.0) + start_sec)

            words = seg.get("words")
            if isinstance(words, list):
                adjusted_words = []
                for word in words:
                    word_copy = dict(word)
                    if "start" in word_copy and word_copy["start"] is not None:
                        word_copy["start"] = float(word_copy["start"] + start_sec)
                    if "end" in word_copy and word_copy["end"] is not None:
                        word_copy["end"] = float(word_copy["end"] + start_sec)
                    adjusted_words.append(word_copy)
                seg_copy["words"] = adjusted_words

            merged_segments.append(seg_copy)

    if not merged_segments:
        logger.info("VAD chunk transcription produced no segments; falling back to full-audio transcription.")
        return model.transcribe(audio_np, batch_size=batch_size)

    return {
        "segments": merged_segments,
        "language": language or "tr",
    }

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
