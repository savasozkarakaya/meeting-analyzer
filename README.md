# Meeting ID

Offline Turkish meeting diarization + target speaker detection.

## Prerequisites

- **Python 3.10+**
- **FFmpeg**: Must be installed and accessible in your system PATH.
- **Hugging Face Token**: Required for Pyannote (used by WhisperX). Set `HF_TOKEN` environment variable.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### First Run (Online)
The first time you run the tool, it needs to download the necessary models (Whisper, Pyannote, SpeechBrain). Ensure you have an internet connection.

### Offline Usage
After the initial download, the tool can run completely offline.

### CLI
```bash
python -m meeting_id.cli --audio path/to/meeting.wav --reference path/to/speaker.wav
```

Long recording / parity tuning example:
```bash
python -m meeting_id.cli \
  --audio path/to/meeting.wav \
  --reference "Alice:path/to/alice.wav" \
  --reference "Bob:path/to/bob.wav" \
  --vad_presegment \
  --min_speakers 2 \
  --max_speakers 6
```

### GUI
```bash
python -m meeting_id.gui
```

### Self-Check / Smoke Test
To verify the environment and installation:
```bash
python -m meeting_id.cli --self_check
```
To run a full end-to-end smoke test with audio files:
```bash
python -m meeting_id.cli --self_check --audio path/to/test.wav --reference path/to/ref.wav --out_dir test_output
```

## Configuration
- `HF_TOKEN`: Environment variable for Hugging Face token.
- `TORCH_HOME` / `HF_HOME`: Set these environment variables to control where models are cached.

## Outputs
- `segments.json`: Segment-level text, identity decision, and word-level attribution.
- `speaker_attributed_transcript.txt`: Readable segment transcript.
- `word_speaker_attribution.json`: Persisted word-level speaker attribution rows.
- `word_speaker_attribution.txt`: Readable word-level attribution timeline.
