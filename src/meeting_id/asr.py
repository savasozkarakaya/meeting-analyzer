import whisperx
import logging
import time
import torch
import numpy as np

logger = logging.getLogger(__name__)
WHISPER_MODEL_VERSION = "large-v2"


def _safe_cuda_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Skipping torch.cuda.empty_cache() due to runtime limitation.", exc_info=True)


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

def load_model(
    device: str = "auto",
    compute_type: str = "float16",
    lang: str = "tr",
    model_size: str = None,
):
    """
    Loads the WhisperX model.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fallback to int8 if cpu
    if device == "cpu":
        compute_type = "int8"

    selected_model = model_size or WHISPER_MODEL_VERSION
    logger.info(
        "ASR model load started model=%s device=%s compute_type=%s lang=%s",
        selected_model,
        device,
        compute_type,
        lang,
    )
    # Using large-v2 or large-v3 is common, but let's stick to a reasonable default or let whisperx decide.
    # WhisperX load_model defaults to large-v2 usually.
    start = time.perf_counter()
    model = whisperx.load_model(selected_model, device, compute_type=compute_type, language=lang)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    logger.info(
        "ASR model load completed model=%s device=%s duration_ms=%s",
        selected_model,
        device,
        duration_ms,
    )
    return model, device


def release_model(model):
    """
    Best-effort release for ASR model resources.
    """
    if model is None:
        return
    try:
        del model
    finally:
        _safe_cuda_empty_cache()

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
    logger.info("ASR transcription started vad_presegment=%s batch_size=%s", vad_presegment, batch_size)
    start = time.perf_counter()
    if not vad_presegment:
        result = model.transcribe(audio_np, batch_size=batch_size)
        logger.info(
            "ASR transcription completed duration_ms=%s segments=%s",
            round((time.perf_counter() - start) * 1000.0, 2),
            len(result.get("segments", [])),
        )
        return result

    regions = _detect_vad_regions(
        audio_np,
        sample_rate=16000,
        min_speech_sec=vad_min_speech_sec,
        max_silence_sec=vad_max_silence_sec,
        padding_sec=vad_padding_sec,
    )
    if not regions:
        logger.info("VAD pre-segmentation found no speech chunks; falling back to full-audio transcription.")
        result = model.transcribe(audio_np, batch_size=batch_size)
        logger.info(
            "ASR transcription completed duration_ms=%s segments=%s",
            round((time.perf_counter() - start) * 1000.0, 2),
            len(result.get("segments", [])),
        )
        return result

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
        result = model.transcribe(audio_np, batch_size=batch_size)
        logger.info(
            "ASR transcription completed duration_ms=%s segments=%s",
            round((time.perf_counter() - start) * 1000.0, 2),
            len(result.get("segments", [])),
        )
        return result

    result = {
        "segments": merged_segments,
        "language": language or "tr",
    }
    logger.info(
        "ASR transcription completed duration_ms=%s segments=%s",
        round((time.perf_counter() - start) * 1000.0, 2),
        len(result.get("segments", [])),
    )
    return result

def align(result, audio_np, device):
    """
    Aligns the transcription result.
    """
    logger.info("ASR alignment started language=%s device=%s", result.get("language"), device)
    start = time.perf_counter()
    # We need to load the alignment model. 
    # Note: This might re-download if not cached.
    model_a = None
    metadata = None
    try:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, audio_np, device, return_char_alignments=False
        )
        logger.info(
            "ASR alignment completed duration_ms=%s aligned_segments=%s",
            round((time.perf_counter() - start) * 1000.0, 2),
            len(result_aligned.get("segments", [])),
        )
        return result_aligned
    finally:
        try:
            del model_a
            del metadata
        except Exception:
            pass
        _safe_cuda_empty_cache()
