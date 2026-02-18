import sys
import os
import subprocess
import logging
import importlib.util
import json

logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Checks if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True, "FFmpeg is available."
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "FFmpeg not found. Please install ffmpeg and add it to PATH."

def check_python_version():
    """Checks if Python version is >= 3.10."""
    if sys.version_info >= (3, 10):
        return True, f"Python version {sys.version.split()[0]} is OK."
    return False, f"Python version {sys.version.split()[0]} is too old. Minimum 3.10 required."

def check_torch():
    """Checks Torch installation and device."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return True, f"Torch available. Device: {device}"
    except ImportError:
        return False, "Torch not installed."

def check_imports():
    """Checks critical imports."""
    missing = []
    
    # WhisperX
    if importlib.util.find_spec("whisperx") is None:
        missing.append("whisperx (pip install whisperx)")
        
    # SpeechBrain
    if importlib.util.find_spec("speechbrain") is None:
        missing.append("speechbrain (pip install speechbrain)")
        
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All critical packages importable."

def check_hf_token():
    """Checks for Hugging Face token."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return True, "HF_TOKEN found."
    return False, "HF_TOKEN not found. Diarization requires a Hugging Face token."

def check_output_dir(path):
    """Checks if output directory is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True, f"Output directory {path} is writable."
    except Exception as e:
        return False, f"Output directory {path} is not writable: {e}"

def check_audio_properties(path):
    """Checks audio properties using soundfile."""
    if not os.path.exists(path):
        return False, f"File not found: {path}"
        
    try:
        import soundfile as sf
        info = sf.info(path)
        # We don't enforce 16k mono here, as the pipeline converts it.
        # But we check if it's a valid audio file.
        if info.duration == 0:
            return False, "Audio duration is 0."
        return True, f"Audio valid: {info.samplerate}Hz, {info.channels}ch, {info.duration:.2f}s"
    except Exception as e:
        return False, f"Invalid audio file: {e}"

def validate_pipeline_output(out_dir):
    """Validates the output files of the pipeline."""
    segments_path = os.path.join(out_dir, "segments.json")
    transcript_path = os.path.join(out_dir, "speaker_attributed_transcript.txt")
    word_json_path = os.path.join(out_dir, "word_speaker_attribution.json")
    word_txt_path = os.path.join(out_dir, "word_speaker_attribution.txt")
    
    if not os.path.exists(segments_path):
        return False, "segments.json not found."
        
    if not os.path.exists(transcript_path):
        return False, "speaker_attributed_transcript.txt not found."

    if not os.path.exists(word_json_path):
        return False, "word_speaker_attribution.json not found."

    if not os.path.exists(word_txt_path):
        return False, "word_speaker_attribution.txt not found."
        
    try:
        with open(segments_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
            
        if not isinstance(segments, list):
            return False, "segments.json is not a list."
            
        required_keys = {
            "start",
            "end",
            "speaker",
            "words",
            "score",
            "decision",
            "flags",
            # Faz 1 identity schema extensions
            "candidate_speakers",
            "confidence",
            "evidence_flags",
            "embedding_model_version",
        }
        valid_decisions = {"accept", "reject", "uncertain"}
        
        for i, seg in enumerate(segments):
            keys = set(seg.keys())
            if not required_keys.issubset(keys):
                return False, f"Segment {i} missing keys: {required_keys - keys}"
                
            if seg["start"] >= seg["end"]:
                return False, f"Segment {i} has invalid time: start={seg['start']}, end={seg['end']}"
                
            if seg["decision"] not in valid_decisions:
                return False, f"Segment {i} has invalid decision: {seg['decision']}"
                
        return True, "Output files validated successfully."
        
    except Exception as e:
        return False, f"Failed to validate output: {e}"

def run_self_check(audio_path=None, ref_path=None, out_dir="test_output"):
    """Runs all checks."""
    print("Running Self-Check...")
    results = []
    
    # 1. Environment Checks
    results.append(check_ffmpeg())
    results.append(check_python_version())
    results.append(check_torch())
    results.append(check_imports())
    results.append(check_hf_token())
    results.append(check_output_dir(out_dir))
    
    # Print results so far
    failed = False
    for success, msg in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {msg}")
        if not success:
            failed = True
            
    if failed:
        print("Environment checks failed. Aborting pipeline test.")
        return False

    # 2. Pipeline Test (if audio provided)
    if audio_path and ref_path:
        print("\nRunning Pipeline Smoke Test...")
        
        # Audio checks
        s1, m1 = check_audio_properties(audio_path)
        print(f"[{'PASS' if s1 else 'FAIL'}] Input Audio: {m1}")
        s2, m2 = check_audio_properties(ref_path)
        print(f"[{'PASS' if s2 else 'FAIL'}] Ref Audio: {m2}")
        
        if not s1 or not s2:
            return False
            
        try:
            from . import pipeline
            # Run pipeline
            pipeline.run_pipeline(
                audio_path=audio_path,
                references=[{"name": "Reference_1", "path": ref_path}],
                out_dir=out_dir,
                device="auto", # Use auto to test device selection
                lang="tr",
                min_segment_sec=0.5 # Short segment for test
            )
            
            # Validate output
            v_success, v_msg = validate_pipeline_output(out_dir)
            print(f"[{'PASS' if v_success else 'FAIL'}] Output Validation: {v_msg}")
            
            if not v_success:
                return False
                
        except Exception as e:
            print(f"[FAIL] Pipeline crashed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    return True
