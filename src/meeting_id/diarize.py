import whisperx
import logging
import os
from bisect import bisect_right

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

def _normalize_diarization_segments(diarize_segments):
    """
    Converts diarization output into sorted list of (start, end, speaker).
    Supports pandas DataFrame-like objects and basic list/dict structures.
    """
    normalized = []

    if diarize_segments is None:
        return normalized

    # whisperx diarization output is commonly a pandas DataFrame.
    if hasattr(diarize_segments, "iterrows"):
        for _, row in diarize_segments.iterrows():
            start = float(row["start"])
            end = float(row["end"])
            speaker = str(row["speaker"])
            if end > start:
                normalized.append((start, end, speaker))
    elif isinstance(diarize_segments, list):
        for row in diarize_segments:
            if not isinstance(row, dict):
                continue
            start = float(row.get("start", 0.0))
            end = float(row.get("end", 0.0))
            speaker = str(row.get("speaker", "UNKNOWN"))
            if end > start:
                normalized.append((start, end, speaker))
    elif isinstance(diarize_segments, dict):
        rows = diarize_segments.get("segments", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                start = float(row.get("start", 0.0))
                end = float(row.get("end", 0.0))
                speaker = str(row.get("speaker", "UNKNOWN"))
                if end > start:
                    normalized.append((start, end, speaker))

    normalized.sort(key=lambda x: x[0])
    return normalized


def _best_speaker_for_span(start, end, diar_intervals, starts, cursor_idx):
    """
    Chooses speaker with maximum overlap for [start, end).
    Uses cursor + sorted starts to avoid full scans on long recordings.
    """
    if not diar_intervals:
        return "UNKNOWN", cursor_idx

    # Keep cursor close to current word/segment start.
    if cursor_idx >= len(diar_intervals):
        cursor_idx = len(diar_intervals) - 1
    while cursor_idx < len(diar_intervals) and diar_intervals[cursor_idx][1] <= start:
        cursor_idx += 1

    if cursor_idx >= len(diar_intervals):
        cursor_idx = len(diar_intervals) - 1

    # Jump backwards if needed (rare with monotonic traversal).
    if cursor_idx > 0 and diar_intervals[cursor_idx][0] > start:
        cursor_idx = max(0, bisect_right(starts, start) - 1)

    overlap_by_speaker = {}
    scan_idx = cursor_idx
    while scan_idx < len(diar_intervals) and diar_intervals[scan_idx][0] < end:
        d_start, d_end, d_speaker = diar_intervals[scan_idx]
        overlap = max(0.0, min(end, d_end) - max(start, d_start))
        if overlap > 0:
            overlap_by_speaker[d_speaker] = overlap_by_speaker.get(d_speaker, 0.0) + overlap
        scan_idx += 1

    if overlap_by_speaker:
        speaker = max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]
        return speaker, cursor_idx

    # If no overlap, choose nearest diarization interval boundary.
    left_idx = max(0, min(cursor_idx, len(diar_intervals) - 1))
    candidates = [left_idx]
    if left_idx + 1 < len(diar_intervals):
        candidates.append(left_idx + 1)
    if left_idx - 1 >= 0:
        candidates.append(left_idx - 1)

    best_idx = left_idx
    best_dist = float("inf")
    for idx in candidates:
        d_start, d_end, _ = diar_intervals[idx]
        if end < d_start:
            dist = d_start - end
        elif start > d_end:
            dist = start - d_end
        else:
            dist = 0.0
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return diar_intervals[best_idx][2], best_idx


def assign_speakers(diarize_segments, transcript_result):
    """
    Assigns speakers to the transcript.
    """
    logger.info("Assigning speakers to words...")
    diar_intervals = _normalize_diarization_segments(diarize_segments)
    if not diar_intervals:
        logger.warning("No diarization intervals available; returning transcript without speaker attribution.")
        return transcript_result

    starts = [s for s, _, _ in diar_intervals]
    cursor_idx = 0

    result = dict(transcript_result or {})
    out_segments = []

    for seg in result.get("segments", []):
        seg_copy = dict(seg)
        seg_start = float(seg_copy.get("start", 0.0))
        seg_end = float(seg_copy.get("end", seg_start))

        words = seg_copy.get("words")
        if isinstance(words, list) and words:
            out_words = []
            word_speaker_durations = {}
            for word in words:
                word_copy = dict(word)
                w_start = word_copy.get("start")
                w_end = word_copy.get("end")
                if w_start is None or w_end is None:
                    # If word timestamps are missing, keep unknown and continue.
                    word_copy["speaker"] = word_copy.get("speaker", "UNKNOWN")
                    out_words.append(word_copy)
                    continue

                w_start = float(w_start)
                w_end = float(w_end)
                if w_end <= w_start:
                    word_copy["speaker"] = "UNKNOWN"
                    out_words.append(word_copy)
                    continue

                speaker, cursor_idx = _best_speaker_for_span(
                    w_start, w_end, diar_intervals, starts, cursor_idx
                )
                word_copy["speaker"] = speaker
                out_words.append(word_copy)
                word_speaker_durations[speaker] = word_speaker_durations.get(speaker, 0.0) + (w_end - w_start)

            seg_copy["words"] = out_words
            if word_speaker_durations:
                seg_copy["speaker"] = max(word_speaker_durations.items(), key=lambda kv: kv[1])[0]
            else:
                seg_copy["speaker"], cursor_idx = _best_speaker_for_span(
                    seg_start, seg_end, diar_intervals, starts, cursor_idx
                )
        else:
            seg_copy["speaker"], cursor_idx = _best_speaker_for_span(
                seg_start, seg_end, diar_intervals, starts, cursor_idx
            )

        out_segments.append(seg_copy)

    result["segments"] = out_segments
    return result
