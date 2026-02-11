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
