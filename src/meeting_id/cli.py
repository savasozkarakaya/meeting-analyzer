import argparse
import logging
import sys
import os

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
        
    if not args.audio or not args.reference:
        parser.error("the following arguments are required: --audio, --reference (unless --self_check is used)")
    
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
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
