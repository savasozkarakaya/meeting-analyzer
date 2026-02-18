import json
import os
import logging

logger = logging.getLogger(__name__)

def write_segments(segments, path):
    """
    Writes segments to a JSON file.
    """
    logger.info(f"Writing segments to {path}...")
    
    # Convert to list if needed, handle numpy types
    def default(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False, default=default)

def write_transcript(segments, path):
    """
    Writes speaker-attributed transcript to a text file.
    Format: [Start - End] Speaker: Text
    """
    logger.info(f"Writing transcript to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            decision = seg.get("decision", "N/A")
            
            # Only include accepted or uncertain? Or all?
            # Prompt says: "speaker_attributed_transcript.txt"
            # Usually we want the final result. 
            # If rejected, maybe we shouldn't include it or mark it.
            # Let's include everything but mark if rejected?
            # Or maybe just the accepted ones for the "clean" transcript.
            # But for debugging, maybe all.
            # Let's stick to a standard format.
            
            line = f"[{start:.2f} - {end:.2f}] {speaker} ({decision}): {text}\n"
            f.write(line)


def _flatten_word_attribution(segments):
    rows = []
    for seg_idx, seg in enumerate(segments):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        seg_speaker = seg.get("speaker", "UNKNOWN")
        words = seg.get("words", [])
        if not isinstance(words, list):
            continue

        for word_idx, word in enumerate(words):
            if not isinstance(word, dict):
                continue
            rows.append(
                {
                    "segment_index": seg_idx,
                    "word_index": word_idx,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "segment_speaker": seg_speaker,
                    "word": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "speaker": word.get("speaker", seg_speaker),
                    "score": word.get("score"),
                }
            )
    return rows


def write_word_speaker_attribution_json(segments, path):
    """
    Persists word-level speaker attribution as JSON rows.
    """
    logger.info(f"Writing word-level speaker attribution to {path}...")

    rows = _flatten_word_attribution(segments)

    def default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=default)


def write_word_speaker_attribution_txt(segments, path):
    """
    Persists word-level speaker attribution as a readable text file.
    """
    logger.info(f"Writing word-level speaker attribution text to {path}...")
    rows = _flatten_word_attribution(segments)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            start = row.get("start")
            end = row.get("end")
            word = row.get("word", "")
            speaker = row.get("speaker", "UNKNOWN")
            if start is None or end is None:
                f.write(f"[N/A] {speaker}: {word}\n")
            else:
                f.write(f"[{float(start):.2f} - {float(end):.2f}] {speaker}: {word}\n")
