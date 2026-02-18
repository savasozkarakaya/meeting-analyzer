import json
import sys

import pytest

from meeting_id import cli


@pytest.mark.integration
def test_cli_eval_manifest_mode_generates_report(tmp_path, monkeypatch):
    manifest = {
        "samples": [
            {
                "id": "sample-1",
                "reference_text": "merhaba dunya",
                "hypothesis_text": "merhaba dunya",
                "reference_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
                "hypothesis_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
                "identity_scores": [0.9, 0.1],
                "identity_labels": [True, False],
                "true_identity_labels": ["A", "B"],
                "pred_identity_labels": ["A", "UNKNOWN"],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "eval_report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meeting_id.cli",
            "--eval_manifest",
            str(manifest_path),
            "--eval_out",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["sample_count"] == 1
    assert "asr" in report["aggregate"]
