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
    hf_token: str = None,
    top_k: int = 3,
    margin_threshold: float = 0.05,
    overlap_penalty: float = 0.5,
    vad_presegment: bool = False,
    vad_min_speech_sec: float = 0.4,
    vad_max_silence_sec: float = 0.35,
    vad_padding_sec: float = 0.15,
    min_speakers=None,
    max_speakers=None,
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
    transcript_result = asr.transcribe(
        asr_model,
        audio_np,
        vad_presegment=vad_presegment,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_max_silence_sec=vad_max_silence_sec,
        vad_padding_sec=vad_padding_sec,
    )
    
    # 4. Align
    logger.info("Step 4: Alignment")
    aligned_result = asr.align(transcript_result, audio_np, device)
    
    # 5. Diarize
    logger.info("Step 5: Diarization")
    diarize_segments = diarize.diarize(
        audio_np,
        device,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    
    # 6. Assign Speakers
    logger.info("Step 6: Assigning Speakers")
    final_result = diarize.assign_speakers(diarize_segments, aligned_result)
    
    # 7. Reference Embeddings
    logger.info("Step 7: Reference Embeddings")
    ref_embeddings_by_name = {}
    ref_infos_by_name = {}
    for ref in references:
        name = ref.get("name", "Unknown")
        path = ref.get("path")
        if path and os.path.exists(path):
            logger.info(f"Extracting embedding for {name} from {path}")
            emb_vec, info = embed.extract_reference_embedding_with_info(embedder, path)
            ref_embeddings_by_name.setdefault(name, []).append(emb_vec)
            ref_infos_by_name.setdefault(name, []).append(info)
        else:
            logger.warning(f"Reference path not found for {name}: {path}")
            
    # Combine multiple references per identity using quality-weighted aggregation.
    ref_embeddings = {}
    ref_combine_meta = {}
    for name, embs in ref_embeddings_by_name.items():
        try:
            infos = ref_infos_by_name.get(name, [])
            combined, combine_info = embed.combine_reference_embeddings(embs, infos)
            ref_embeddings[name] = combined
            ref_combine_meta[name] = combine_info
        except Exception as e:
            logger.warning(f"Failed to combine reference embeddings for {name}: {e}")

    if not ref_embeddings:
        logger.warning("No valid reference embeddings found. All speakers will be Unknown.")
    
    # 8. Scoring and Decision
    logger.info("Step 8: Scoring")
    
    segments = final_result["segments"]
    processed_segments = []
    
    full_audio_tensor = torch.from_numpy(audio_np)
    
    prev_end = None
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        
        seg_info = {
            "start": start,
            "end": end,
            "original_speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg.get("text", ""),
            "words": seg.get("words", []),
            "flags": [],
            "evidence_flags": [],
            "candidate_speakers": [],
            "confidence": 0.0,
            "embedding_model_version": embed.EMBEDDING_MODEL_VERSION,
            "decision_reason": "",
            "score_margin": None,
            "thresholds": {
                "accept_threshold": float(accept_threshold),
                "reject_threshold": float(reject_threshold),
                "margin_threshold": float(margin_threshold),
            },
        }

        # Overlap heuristic: segments overlapping in time (should be rare but can happen)
        if prev_end is not None and start < prev_end:
            seg_info["flags"].append("overlap")
            seg_info["evidence_flags"].append("overlap")
        
        # Check duration
        if duration < min_segment_sec:
            seg_info["decision"] = "reject"
            seg_info["flags"].append("too_short")
            seg_info["evidence_flags"].append("too_short")
            seg_info["score"] = 0.0
            seg_info["score_margin"] = 0.0
            seg_info["speaker"] = "UNKNOWN"
            seg_info["decision_reason"] = "segment_duration_below_minimum"
        else:
            # Extract segment audio
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_np), end_sample)
            
            if end_sample - start_sample < 160: 
                seg_info["decision"] = "reject"
                seg_info["flags"].append("empty")
                seg_info["evidence_flags"].append("empty")
                seg_info["score"] = 0.0
                seg_info["score_margin"] = 0.0
                seg_info["speaker"] = "UNKNOWN"
                seg_info["decision_reason"] = "segment_audio_too_short_or_empty"
            else:
                seg_tensor = full_audio_tensor[start_sample:end_sample]
                seg_emb = embedder.get_embedding(seg_tensor)
                
                # Compare against all references
                candidates = []
                
                for name, ref_emb in ref_embeddings.items():
                    score = embedder.compute_similarity(seg_emb, ref_emb)
                    candidates.append((name, float(score)))

                candidates.sort(key=lambda x: x[1], reverse=True)

                if candidates:
                    best_speaker, best_score = candidates[0]
                    second_best = candidates[1][1] if len(candidates) > 1 else None
                    margin = (best_score - second_best) if second_best is not None else None
                else:
                    best_speaker, best_score, margin = "UNKNOWN", -1.0, None

                seg_info["score"] = float(best_score)
                seg_info["score_margin"] = float(margin) if margin is not None else None

                top_k_n = max(1, int(top_k))
                seg_info["candidate_speakers"] = []
                for rank_idx, (n, s) in enumerate(candidates[:top_k_n], start=1):
                    candidate_payload = {
                        "name": n,
                        "score": float(s),
                        "rank": rank_idx,
                        "reference_count": int(ref_combine_meta.get(n, {}).get("num_references", 1)),
                        "reference_quality_mean": float(
                            np.mean(ref_combine_meta.get(n, {}).get("quality_scores", [1.0]))
                        ),
                    }
                    if rank_idx > 1 and candidates:
                        candidate_payload["delta_to_best"] = float(best_score - s)
                    else:
                        candidate_payload["delta_to_best"] = 0.0
                    seg_info["candidate_speakers"].append(candidate_payload)

                # Confidence (0..1) based on thresholds + margin
                if accept_threshold > reject_threshold:
                    score_conf = (best_score - reject_threshold) / (accept_threshold - reject_threshold)
                else:
                    score_conf = 0.0
                score_conf = float(max(0.0, min(1.0, score_conf)))

                margin_conf = 1.0
                if margin is not None and margin_threshold > 0:
                    margin_conf = float(max(0.0, min(1.0, margin / margin_threshold)))
                elif len(candidates) <= 1:
                    # Single-reference identity banks are inherently lower certainty.
                    margin_conf = 0.85

                duration_conf = float(max(0.6, min(1.0, duration / max(min_segment_sec * 2.0, 1e-6))))
                if "overlap" in seg_info["flags"]:
                    overlap_conf = float(max(0.0, min(1.0, overlap_penalty)))
                else:
                    overlap_conf = 1.0

                conf = float(score_conf * margin_conf * duration_conf * overlap_conf)
                seg_info["confidence"] = conf
                
                # Decision
                margin_ok = True
                if margin is not None and margin_threshold > 0:
                    margin_ok = margin >= margin_threshold

                if not candidates:
                    seg_info["decision"] = "reject"
                    seg_info["speaker"] = "UNKNOWN"
                    seg_info["evidence_flags"].append("no_reference_candidates")
                    seg_info["decision_reason"] = "reference_bank_empty_or_invalid"
                elif best_score >= accept_threshold and margin_ok and conf >= 0.5:
                    seg_info["decision"] = "accept"
                    seg_info["speaker"] = best_speaker
                    seg_info["decision_reason"] = "score_and_margin_above_acceptance"
                elif best_score <= reject_threshold:
                    seg_info["decision"] = "reject"
                    seg_info["speaker"] = "UNKNOWN"
                    seg_info["decision_reason"] = "score_below_reject_threshold"
                else:
                    seg_info["decision"] = "uncertain"
                    seg_info["speaker"] = best_speaker
                    seg_info["decision_reason"] = "score_in_gray_zone_or_low_margin"

                if margin is not None and margin_threshold > 0 and margin < margin_threshold:
                    seg_info["evidence_flags"].append("low_margin")
                if conf < 0.5:
                    seg_info["evidence_flags"].append("low_confidence")
                if len(candidates) <= 1:
                    seg_info["evidence_flags"].append("single_candidate_only")
                if "overlap" in seg_info["flags"] and "overlap" not in seg_info["evidence_flags"]:
                    seg_info["evidence_flags"].append("overlap")
        
        processed_segments.append(seg_info)
        prev_end = max(prev_end, end) if prev_end is not None else end
        
    # 9. Write Outputs
    logger.info("Step 9: Writing Outputs")
    io.write_segments(processed_segments, os.path.join(out_dir, "segments.json"))
    io.write_transcript(processed_segments, os.path.join(out_dir, "speaker_attributed_transcript.txt"))
    io.write_word_speaker_attribution_json(
        processed_segments, os.path.join(out_dir, "word_speaker_attribution.json")
    )
    io.write_word_speaker_attribution_txt(
        processed_segments, os.path.join(out_dir, "word_speaker_attribution.txt")
    )
    
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Summary generation is future work.\n")
        
    logger.info("Pipeline completed successfully.")
    return processed_segments
