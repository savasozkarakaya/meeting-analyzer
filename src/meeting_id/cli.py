import argparse
import logging
import sys
import os


PROFILE_DEFAULTS = {
    "fast": {"model_size": "small", "compute_type": "int8", "batch_size": 24},
    "balanced": {"model_size": "large-v2", "compute_type": "float16", "batch_size": 16},
    "accurate": {"model_size": "large-v3", "compute_type": "float16", "batch_size": 8},
}

def _looks_like_windows_drive_path(s: str) -> bool:
    # e.g. "C:\\foo\\bar.wav" or "D:/foo/bar.wav"
    return (
        len(s) >= 3
        and s[1] == ":"
        and s[0].isalpha()
        and (s[2] == "\\" or s[2] == "/")
    )

def _parse_reference_arg(ref_str: str, default_name: str):
    """
    Parse --reference argument.
    Supported:
      - 'Name:Path'  (only when it does NOT look like a Windows drive path/UNC path)
      - 'Path'       (any path, including Windows paths with ':')
    """
    ref_str = (ref_str or "").strip()
    if not ref_str:
        return default_name, ref_str

    # If it looks like a Windows drive path or UNC path, never treat ':' as a Name:Path separator.
    if _looks_like_windows_drive_path(ref_str) or ref_str.startswith("\\\\"):
        return default_name, ref_str

    # Deterministic parsing:
    # - if it contains ':' (and it's not a drive path), treat it as Name:Path
    # - otherwise treat as Path
    if ":" in ref_str:
        name_candidate, path_candidate = ref_str.split(":", 1)
        name_candidate = name_candidate.strip()
        path_candidate = path_candidate.strip()

        # If the user accidentally passed something like ":" or "Name:", fall back to treating it as a path.
        if path_candidate:
            return (name_candidate or default_name), path_candidate

    return default_name, ref_str

def main():
    parser = argparse.ArgumentParser(description="Offline Turkish Meeting Diarization")
    
    parser.add_argument("--audio", help="Path to meeting audio file")
    # Support multiple references: --reference "Name:Path" or just "Path" (default name Unknown_N)
    parser.add_argument("--reference", action="append", help="Reference speaker audio. Format: 'Name:Path' or just 'Path'. Can be used multiple times.")
    parser.add_argument("--out_dir", default="output", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda)")
    parser.add_argument("--lang", default="tr", help="Language code")
    
    parser.add_argument("--accept_threshold", type=float, default=0.65, help="Acceptance threshold for speaker verification")
    parser.add_argument("--reject_threshold", type=float, default=0.45, help="Rejection threshold for speaker verification")
    parser.add_argument("--min_segment_sec", type=float, default=2.0, help="Minimum segment duration in seconds")
    parser.add_argument("--vad_presegment", action="store_true", help="Enable lightweight VAD pre-segmentation before ASR")
    parser.add_argument("--vad_min_speech_sec", type=float, default=0.4, help="Minimum speech chunk duration for VAD pre-segmentation")
    parser.add_argument("--vad_max_silence_sec", type=float, default=0.35, help="Max silence gap to keep chunk continuity for VAD pre-segmentation")
    parser.add_argument("--vad_padding_sec", type=float, default=0.15, help="Padding around detected VAD speech chunks")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum expected speaker count for diarization")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum expected speaker count for diarization")
    parser.add_argument("--num_speakers", type=int, default=None, help="Fixed expected speaker count for diarization")
    parser.add_argument("--model_size", default=None, help="ASR model size (e.g. small, medium, large-v2, large-v3)")
    parser.add_argument("--compute_type", default=None, help="ASR compute type (e.g. float16, float32, int8)")
    parser.add_argument("--batch_size", type=int, default=None, help="ASR transcription batch size")
    parser.add_argument("--profile", choices=["fast", "balanced", "accurate"], default=None, help="Preset for ASR model/compute/batch")
    parser.add_argument("--eval_manifest", default=None, help="Path to evaluation manifest JSON")
    parser.add_argument("--eval_out", default=None, help="Optional output path for evaluation report JSON")
    
    parser.add_argument("--self_check", action="store_true", help="Run self-check/smoke test")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if args.self_check:
        from . import checks
        ref_path = None
        if args.reference:
            ref_str = args.reference[0]
            _, ref_path = _parse_reference_arg(ref_str, default_name="Reference_1")
                
        success = checks.run_self_check(args.audio, ref_path, args.out_dir)
        sys.exit(0 if success else 1)

    if args.eval_manifest:
        try:
            from . import eval as eval_module

            report_out = args.eval_out or os.path.join(args.out_dir, "eval_report.json")
            report = eval_module.run_manifest_evaluation(
                manifest_path=args.eval_manifest,
                output_path=report_out,
            )
            logging.info(
                "Evaluation completed: samples=%s, output=%s",
                report.get("aggregate", {}).get("sample_count", 0),
                report_out,
            )
            sys.exit(0)
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            sys.exit(1)
        
    if not args.audio or not args.reference:
        parser.error("the following arguments are required: --audio, --reference (unless --self_check is used)")

    if args.min_speakers is not None and args.min_speakers <= 0:
        parser.error("--min_speakers must be a positive integer")
    if args.max_speakers is not None and args.max_speakers <= 0:
        parser.error("--max_speakers must be a positive integer")
    if args.num_speakers is not None and args.num_speakers <= 0:
        parser.error("--num_speakers must be a positive integer")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min_speakers cannot be greater than --max_speakers")
    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error("--num_speakers cannot be used together with --min_speakers/--max_speakers")

    profile_defaults = PROFILE_DEFAULTS.get(args.profile, {}) if args.profile else {}
    model_size = args.model_size or profile_defaults.get("model_size")
    compute_type = args.compute_type or profile_defaults.get("compute_type")
    batch_size = args.batch_size if args.batch_size is not None else profile_defaults.get("batch_size")

    if batch_size is not None and batch_size <= 0:
        parser.error("--batch_size must be a positive integer")
    
    # Parse references
    parsed_references = []
    for i, ref_str in enumerate(args.reference):
        default_name = f"Speaker_{i+1}"
        name, path = _parse_reference_arg(ref_str, default_name=default_name)
        parsed_references.append({"name": name, "path": path})

    try:
        from . import pipeline
        pipeline.run_pipeline(
            audio_path=args.audio,
            references=parsed_references,
            out_dir=args.out_dir,
            device=args.device,
            lang=args.lang,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
            min_segment_sec=args.min_segment_sec,
            vad_presegment=args.vad_presegment,
            vad_min_speech_sec=args.vad_min_speech_sec,
            vad_max_silence_sec=args.vad_max_silence_sec,
            vad_padding_sec=args.vad_padding_sec,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            num_speakers=args.num_speakers,
            model_size=model_size,
            compute_type=compute_type,
            batch_size=batch_size,
            profile=args.profile,
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
