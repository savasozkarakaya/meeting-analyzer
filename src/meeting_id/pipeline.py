import os
import json
import logging
import traceback
import uuid
import time
from datetime import datetime, timezone
import torch
import numpy as np
from . import audio, asr, diarize, embed, io

logger = logging.getLogger(__name__)

ERROR_CATEGORY_INPUT = "input"
ERROR_CATEGORY_MODEL = "model"
ERROR_CATEGORY_RUNTIME = "runtime"
ERROR_CATEGORY_DEPENDENCY = "dependency"


class PipelineStepError(RuntimeError):
    def __init__(self, step_name: str, category: str, original: Exception):
        self.step_name = step_name
        self.category = category
        self.original = original
        super().__init__(f"{step_name} failed ({category}): {original}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    name = exc.__class__.__name__.lower()
    if any(k in msg for k in ("file not found", "no such file", "invalid", "empty", "permission denied")):
        return ERROR_CATEGORY_INPUT
    if any(k in msg for k in ("cuda", "out of memory", "shape", "dimension")):
        return ERROR_CATEGORY_RUNTIME
    if any(k in msg for k in ("import", "module not found", "ffmpeg", "dependency", "token", "auth")):
        return ERROR_CATEGORY_DEPENDENCY
    if any(k in msg for k in ("model", "checkpoint", "whisper", "pyannote", "speechbrain")):
        return ERROR_CATEGORY_MODEL
    if "import" in name:
        return ERROR_CATEGORY_DEPENDENCY
    return ERROR_CATEGORY_RUNTIME


def _log_telemetry(run_id: str, event: str, **fields):
    payload = {"event": event, "run_id": run_id, **fields}
    logger.info("telemetry %s", json.dumps(payload, ensure_ascii=False, default=str))


def _execute_step(run_id: str, step_name: str, fn, step_durations_ms: dict):
    started = time.perf_counter()
    _log_telemetry(run_id, "step_start", step=step_name)
    try:
        result = fn()
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        step_durations_ms[step_name] = duration_ms
        category = _classify_error(exc)
        _log_telemetry(
            run_id,
            "step_error",
            step=step_name,
            duration_ms=duration_ms,
            error_category=category,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
        raise PipelineStepError(step_name=step_name, category=category, original=exc) from exc
    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    step_durations_ms[step_name] = duration_ms
    _log_telemetry(run_id, "step_end", step=step_name, duration_ms=duration_ms)
    return result


def _write_run_report(out_dir: str, report: dict):
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


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
    num_speakers=None,
    model_size: str = None,
    compute_type: str = None,
    batch_size: int = None,
    profile: str = None,
):
    """
    Runs the full pipeline.
    """
    run_id = uuid.uuid4().hex[:12]
    run_started_at = _utc_now_iso()
    run_started_perf = time.perf_counter()
    step_durations_ms = {}
    asr_model = None
    embedder = None
    report = {
        "run_id": run_id,
        "status": "running",
        "started_at": run_started_at,
        "finished_at": None,
        "duration_ms": None,
        "audio_path": audio_path,
        "out_dir": out_dir,
        "device_requested": device,
        "lang": lang,
        "thresholds": {
            "accept_threshold": float(accept_threshold),
            "reject_threshold": float(reject_threshold),
            "margin_threshold": float(margin_threshold),
        },
        "asr_config": {
            "profile": profile or "default",
            "model_size": model_size or asr.WHISPER_MODEL_VERSION,
            "compute_type": compute_type or "float16",
            "batch_size": int(batch_size) if batch_size is not None else 16,
        },
        "step_durations_ms": step_durations_ms,
        "models": {
            "asr": asr.WHISPER_MODEL_VERSION,
            "speaker_embedding": embed.EMBEDDING_MODEL_VERSION,
            "diarization": diarize.DIARIZATION_MODEL_VERSION,
        },
        "error": None,
    }

    _log_telemetry(
        run_id,
        "run_start",
        audio_path=audio_path,
        out_dir=out_dir,
        references_count=len(references or []),
        device_requested=device,
        lang=lang,
        asr_profile=profile or "default",
        asr_model_size=model_size or asr.WHISPER_MODEL_VERSION,
        asr_compute_type=compute_type or "float16",
        asr_batch_size=int(batch_size) if batch_size is not None else 16,
    )

    os.makedirs(out_dir, exist_ok=True)
    try:
        if min_speakers is not None and max_speakers is not None and int(min_speakers) > int(max_speakers):
            raise ValueError("min_speakers cannot be greater than max_speakers")
        if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
            raise ValueError("num_speakers cannot be used together with min_speakers/max_speakers")

        # 1. Convert/Load Audio
        logger.info("Step 1: Audio Processing")

        def _step_audio():
            with audio.ensure_wav_16k_mono(audio_path) as wav_path:
                return audio.load_audio(wav_path)

        audio_np = _execute_step(run_id, "audio_processing", _step_audio, step_durations_ms)

        # 2. Load Models
        logger.info("Step 2: Loading Models")

        def _step_models():
            asr_model_local, device_local = asr.load_model(
                device,
                compute_type=compute_type or "float16",
                lang=lang,
                model_size=model_size,
            )
            embedder_local = embed.Embedder(device=device_local)
            return asr_model_local, device_local, embedder_local

        asr_model, device, embedder = _execute_step(run_id, "load_models", _step_models, step_durations_ms)

        # 3. Transcribe
        logger.info("Step 3: Transcription")

        def _step_transcribe():
            return asr.transcribe(
                asr_model,
                audio_np,
                batch_size=int(batch_size) if batch_size is not None else 16,
                vad_presegment=vad_presegment,
                vad_min_speech_sec=vad_min_speech_sec,
                vad_max_silence_sec=vad_max_silence_sec,
                vad_padding_sec=vad_padding_sec,
            )

        transcript_result = _execute_step(run_id, "transcription", _step_transcribe, step_durations_ms)

        # 4. Align
        logger.info("Step 4: Alignment")
        aligned_result = _execute_step(
            run_id,
            "alignment",
            lambda: asr.align(transcript_result, audio_np, device),
            step_durations_ms,
        )

        # 5. Diarize
        logger.info("Step 5: Diarization")
        diarize_segments = _execute_step(
            run_id,
            "diarization",
            lambda: diarize.diarize(
                audio_np,
                device,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
            ),
            step_durations_ms,
        )

        # 6. Assign Speakers
        logger.info("Step 6: Assigning Speakers")
        final_result = _execute_step(
            run_id,
            "assign_speakers",
            lambda: diarize.assign_speakers(diarize_segments, aligned_result),
            step_durations_ms,
        )

        # 7. Reference Embeddings
        logger.info("Step 7: Reference Embeddings")

        ref_embeddings_by_name = {}
        ref_infos_by_name = {}

        def _step_reference_embeddings():
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

        _execute_step(run_id, "reference_embedding_extraction", _step_reference_embeddings, step_durations_ms)

        # Combine multiple references per identity using quality-weighted aggregation.
        ref_embeddings = {}
        ref_combine_meta = {}

        def _step_reference_combine():
            for name, embs in ref_embeddings_by_name.items():
                try:
                    infos = ref_infos_by_name.get(name, [])
                    combined, combine_info = embed.combine_reference_embeddings(embs, infos)
                    ref_embeddings[name] = combined
                    ref_combine_meta[name] = combine_info
                except Exception as e:
                    logger.warning(f"Failed to combine reference embeddings for {name}: {e}")

        _execute_step(run_id, "reference_embedding_combine", _step_reference_combine, step_durations_ms)

        if not ref_embeddings:
            logger.warning("No valid reference embeddings found. All speakers will be Unknown.")

        # 8. Scoring and Decision
        logger.info("Step 8: Scoring")

        segments = final_result["segments"]
        processed_segments = []

        def _step_scoring():
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

            return processed_segments

        processed_segments = _execute_step(run_id, "scoring", _step_scoring, step_durations_ms)

        # 9. Write Outputs
        logger.info("Step 9: Writing Outputs")

        def _step_outputs():
            io.write_segments(processed_segments, os.path.join(out_dir, "segments.json"))
            io.write_transcript(processed_segments, os.path.join(out_dir, "speaker_attributed_transcript.txt"))
            io.write_word_speaker_attribution_json(
                processed_segments, os.path.join(out_dir, "word_speaker_attribution.json")
            )
            io.write_word_speaker_attribution_txt(
                processed_segments, os.path.join(out_dir, "word_speaker_attribution.txt")
            )

            with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("Summary generation is future work.\n")

        _execute_step(run_id, "write_outputs", _step_outputs, step_durations_ms)

        report["status"] = "success"
        report["segment_count"] = len(processed_segments)
        report["uncertain_count"] = sum(1 for s in processed_segments if s.get("decision") == "uncertain")
        return processed_segments
    except PipelineStepError as step_exc:
        report["status"] = "failed"
        report["error"] = {
            "category": step_exc.category,
            "step": step_exc.step_name,
            "type": step_exc.original.__class__.__name__,
            "message": str(step_exc.original),
            "traceback": traceback.format_exc(),
        }
        raise RuntimeError(
            f"pipeline_failed category={step_exc.category} step={step_exc.step_name}: {step_exc.original}"
        ) from step_exc
    except Exception as exc:
        category = _classify_error(exc)
        report["status"] = "failed"
        report["error"] = {
            "category": category,
            "step": "unknown",
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise RuntimeError(f"pipeline_failed category={category} step=unknown: {exc}") from exc
    finally:
        # Explicit release to reduce memory pressure in long-running sessions.
        try:
            asr.release_model(asr_model)
        except Exception:
            logger.debug("ASR model release failed.", exc_info=True)
        try:
            if embedder is not None and hasattr(embedder, "verification"):
                del embedder.verification
            del embedder
        except Exception:
            logger.debug("Embedder release failed.", exc_info=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Skipping final torch.cuda.empty_cache().", exc_info=True)

        report["finished_at"] = _utc_now_iso()
        report["duration_ms"] = round((time.perf_counter() - run_started_perf) * 1000.0, 2)
        _write_run_report(out_dir, report)
        _log_telemetry(
            run_id,
            "run_end",
            status=report.get("status"),
            duration_ms=report.get("duration_ms"),
            step_durations_ms=step_durations_ms,
            error=report.get("error"),
        )
        logger.info("Pipeline completed with status=%s run_id=%s", report.get("status"), run_id)
