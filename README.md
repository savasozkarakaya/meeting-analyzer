# Meeting ID

Offline Turkish meeting segmentation and speaker labeling with CLI + GUI workflows.

## Prerequisites

- Python `3.10+`
- `ffmpeg` in PATH
- Hugging Face token (`HF_TOKEN`) for required component downloads

## Quickstart (10-15 minutes)

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Export token and optional cache paths
```bash
# PowerShell
$env:HF_TOKEN = "hf_xxx"
$env:HF_HOME = "D:\hf_cache"
$env:TORCH_HOME = "D:\torch_cache"
```

3) Run a first online execution (downloads required components on first run)
```bash
python -m meeting_id.cli --audio path/to/meeting.wav --reference "Alice:path/to/alice.wav" --out_dir output
```

4) Verify generated outputs in `output/`
- `segments.json`
- `speaker_attributed_transcript.txt`
- `word_speaker_attribution.json`
- `word_speaker_attribution.txt`
- `run_report.json`

5) Optional: launch GUI for manual uncertain resolution
```bash
python -m meeting_id.gui
```

## Usage

### CLI: Basic
```bash
python -m meeting_id.cli --audio path/to/meeting.wav --reference path/to/speaker.wav
```

### CLI: Multi-speaker + long recording tuning
```bash
python -m meeting_id.cli \
  --audio path/to/meeting.wav \
  --reference "Alice:path/to/alice.wav" \
  --reference "Bob:path/to/bob.wav" \
  --profile balanced \
  --batch_size 16 \
  --vad_presegment \
  --min_speakers 2 \
  --max_speakers 6 \
  --accept_threshold 0.68 \
  --reject_threshold 0.42
```

### CLI: Fixed speaker count (`num_speakers`)
```bash
python -m meeting_id.cli \
  --audio path/to/meeting.wav \
  --reference "Alice:path/to/alice.wav" \
  --reference "Bob:path/to/bob.wav" \
  --num_speakers 2
```

Conflict rule:
- `--num_speakers` cannot be used with `--min_speakers` or `--max_speakers`
- invalid combinations fail fast with a clear CLI error

### CLI: ASR parameter presets and overrides
```bash
# Preset driven
python -m meeting_id.cli --audio path/to/meeting.wav --reference path/to/ref.wav --profile fast
```

```bash
# Explicit override
python -m meeting_id.cli \
  --audio path/to/meeting.wav \
  --reference path/to/ref.wav \
  --model_size large-v3 \
  --compute_type float16 \
  --batch_size 8
```

### CLI: Batch evaluation (manifest driven)
```bash
python -m meeting_id.cli --eval_manifest path/to/eval_manifest.json --eval_out output/eval_report.json
```

Minimal manifest schema:
```json
{
  "samples": [
    {
      "id": "sample-1",
      "reference_text": "ground truth transcript",
      "hypothesis_text": "predicted transcript",
      "reference_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
      "hypothesis_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
      "identity_scores": [0.91, 0.12],
      "identity_labels": [true, false]
    }
  ]
}
```

### GUI
```bash
python -m meeting_id.gui
```

The GUI supports manual review for `decision == uncertain` segments:
- unresolved segments are listed in a review modal
- you can assign a known speaker or set `UNKNOWN`
- reviewed decisions are written back to output files

### Self-check / smoke test
```bash
python -m meeting_id.cli --self_check
```

```bash
python -m meeting_id.cli --self_check --audio path/to/test.wav --reference path/to/ref.wav --out_dir test_output
```

## Offline Cache Management

First execution must be online because required runtime components are downloaded.

For controlled/offline deployments:

1) Set cache roots before first run
```bash
# PowerShell
$env:HF_HOME = "D:\meeting_id_cache\hf"
$env:TORCH_HOME = "D:\meeting_id_cache\torch"
```

2) Perform one successful online run to warm caches.

3) Archive cache folders and ship them to offline machine.

4) Restore to the same paths and run offline.

If offline run fails, check:
- token availability (`HF_TOKEN`) for first-time downloads
- cache path permissions
- `run_report.json` for failed step and error category

## Identity Calibration Guide (accept/reject/margin)

Identity decisions are score-based:
- `accept_threshold`: score above this tends to auto-accept
- `reject_threshold`: score below this tends to reject as `UNKNOWN`
- gray zone between thresholds becomes `uncertain`
- margin logic (best vs second-best candidate) reduces false positives

### Practical calibration flow

1) Start with defaults (`accept=0.65`, `reject=0.45`)

2) Run on a labeled sample set and inspect `segments.json`:
- too many false accepts -> increase `accept_threshold` (for example `0.65 -> 0.72`)
- too many unknown rejects -> decrease `reject_threshold` slightly (for example `0.45 -> 0.40`)
- many `uncertain` outputs -> reduce gap between thresholds carefully

3) Keep a stable target ratio:
- prioritize low false-accept for production identity workflows
- use GUI manual review for uncertain segments when confidence is moderate

4) Re-test with varied audio conditions:
- noisy room
- overlapped speech
- short utterances

### Example tuning commands

Conservative acceptance:
```bash
python -m meeting_id.cli --audio path/to/meeting.wav --reference "Alice:path/to/alice.wav" --accept_threshold 0.75 --reject_threshold 0.45
```

Higher recall (more accepts, more review load):
```bash
python -m meeting_id.cli --audio path/to/meeting.wav --reference "Alice:path/to/alice.wav" --accept_threshold 0.62 --reject_threshold 0.38
```

## Telemetry and Diagnostics

Each run emits:
- structured telemetry lines in logs (`run_id`, step start/end, durations)
- `run_report.json` with lifecycle summary and step timings
- categorized failure context (`input`, `model`, `runtime`, `dependency`) on errors

This makes CLI and GUI runs traceable and easier to troubleshoot.
