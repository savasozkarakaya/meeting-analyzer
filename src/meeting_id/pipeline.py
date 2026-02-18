import os
import logging
import torch
import numpy as np
from . import audio, asr, diarize, embed, io

logger = logging.getLogger(__name__)

def run_pipeline(
    audio_path: str,
    references: list, # List of {'name': str, 'path': str}
    out_dir: str,
    device: str = "auto",
    lang: str = "tr",
    accept_threshold: float = 0.65,
    reject_threshold: float = 0.45,
    min_segment_sec: float = 2.0,
    hf_token: str = None
):
    """
    Runs the full pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Convert/Load Audio
    logger.info("Step 1: Audio Processing")
    with audio.ensure_wav_16k_mono(audio_path) as wav_path:
        audio_np = audio.load_audio(wav_path)
    
    # 2. Load Models
    logger.info("Step 2: Loading Models")
    asr_model, device = asr.load_model(device, lang=lang)
    embedder = embed.Embedder(device=device)
    
    # 3. Transcribe
    logger.info("Step 3: Transcription")
    transcript_result = asr.transcribe(asr_model, audio_np)
    
    # 4. Align
    logger.info("Step 4: Alignment")
    aligned_result = asr.align(transcript_result, audio_np, device)
    
    # 5. Diarize
    logger.info("Step 5: Diarization")
    diarize_segments = diarize.diarize(audio_np, device, hf_token=hf_token)
    
    # 6. Assign Speakers
    logger.info("Step 6: Assigning Speakers")
    final_result = diarize.assign_speakers(diarize_segments, aligned_result)
    
    # 7. Reference Embeddings
    logger.info("Step 7: Reference Embeddings")
    ref_embeddings = {}
    for ref in references:
        name = ref.get("name", "Unknown")
        path = ref.get("path")
        if path and os.path.exists(path):
            logger.info(f"Extracting embedding for {name} from {path}")
            emb = embed.extract_reference_embedding(embedder, path)
            ref_embeddings[name] = emb
        else:
            logger.warning(f"Reference path not found for {name}: {path}")
            
    if not ref_embeddings:
        logger.warning("No valid reference embeddings found. All speakers will be Unknown.")
    
    # 8. Scoring and Decision
    logger.info("Step 8: Scoring")
    
    segments = final_result["segments"]
    processed_segments = []
    
    full_audio_tensor = torch.from_numpy(audio_np)
    
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        
        seg_info = {
            "start": start,
            "end": end,
            "original_speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg.get("text", ""),
            "flags": []
        }
        
        # Check duration
        if duration < min_segment_sec:
            seg_info["decision"] = "reject"
            seg_info["flags"].append("too_short")
            seg_info["score"] = 0.0
            seg_info["speaker"] = "UNKNOWN"
        else:
            # Extract segment audio
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_np), end_sample)
            
            if end_sample - start_sample < 160: 
                seg_info["decision"] = "reject"
                seg_info["flags"].append("empty")
                seg_info["score"] = 0.0
                seg_info["speaker"] = "UNKNOWN"
            else:
                seg_tensor = full_audio_tensor[start_sample:end_sample]
                seg_emb = embedder.get_embedding(seg_tensor)
                
                # Compare against all references
                best_score = -1.0
                best_speaker = "UNKNOWN"
                
                for name, ref_emb in ref_embeddings.items():
                    score = embedder.compute_similarity(seg_emb, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_speaker = name
                
                seg_info["score"] = best_score
                
                # Decision
                if best_score >= accept_threshold:
                    seg_info["decision"] = "accept"
                    seg_info["speaker"] = best_speaker
                elif best_score <= reject_threshold:
                    seg_info["decision"] = "reject"
                    seg_info["speaker"] = "UNKNOWN" # Or keep best_speaker but mark reject?
                else:
                    seg_info["decision"] = "uncertain"
                    seg_info["speaker"] = best_speaker # Tentative
        
        processed_segments.append(seg_info)
        
    # 9. Write Outputs
    logger.info("Step 9: Writing Outputs")
    io.write_segments(processed_segments, os.path.join(out_dir, "segments.json"))
    io.write_transcript(processed_segments, os.path.join(out_dir, "speaker_attributed_transcript.txt"))
    
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Summary generation is future work.\n")
        
    logger.info("Pipeline completed successfully.")
    return processed_segments
