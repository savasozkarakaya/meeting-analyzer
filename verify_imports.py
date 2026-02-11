import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    print("Importing meeting_id...")
    import meeting_id
    print("Importing audio...")
    from meeting_id import audio
    print("Importing asr...")
    from meeting_id import asr
    print("Importing diarize...")
    from meeting_id import diarize
    print("Importing embed...")
    from meeting_id import embed
    print("Importing io...")
    from meeting_id import io
    print("Importing pipeline...")
    from meeting_id import pipeline
    print("Importing cli...")
    from meeting_id import cli
    print("Importing gui...")
    from meeting_id import gui
    print("All imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
