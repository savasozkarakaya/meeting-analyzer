import argparse
import logging
import sys
import os
from . import pipeline

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
        # For self check, we might need to adapt if it uses args.reference as string
        # But checks.run_self_check expects single paths. 
        # Let's handle simple case for self check: use first reference if available
        ref_path = None
        if args.reference:
            # Take the first one, split if needed
            ref_str = args.reference[0]
            if ":" in ref_str and not os.path.exists(ref_str):
                ref_path = ref_str.split(":", 1)[1]
            else:
                ref_path = ref_str
                
        success = checks.run_self_check(args.audio, ref_path, args.out_dir)
        sys.exit(0 if success else 1)
        
    if not args.audio or not args.reference:
        parser.error("the following arguments are required: --audio, --reference (unless --self_check is used)")
    
    # Parse references
    parsed_references = []
    for i, ref_str in enumerate(args.reference):
        if ":" in ref_str and not os.path.exists(ref_str):
            # Assumed format Name:Path
            name, path = ref_str.split(":", 1)
        else:
            name = f"Speaker_{i+1}"
            path = ref_str
        
        parsed_references.append({"name": name, "path": path})

    try:
        pipeline.run_pipeline(
            audio_path=args.audio,
            references=parsed_references,
            out_dir=args.out_dir,
            device=args.device,
            lang=args.lang,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
            min_segment_sec=args.min_segment_sec
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
