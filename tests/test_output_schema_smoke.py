import json

from meeting_id import io


def test_segments_json_schema_smoke(tmp_path):
    sample_segments = [
        {
            "start": 0.0,
            "end": 1.25,
            "speaker": "Alice",
            "text": "Merhaba",
            "decision": "accept",
            "words": [{"word": "Merhaba", "start": 0.1, "end": 0.8, "speaker": "Alice"}],
        }
    ]

    output_path = tmp_path / "segments.json"
    io.write_segments(sample_segments, str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1

    segment = payload[0]
    required_keys = {"start", "end", "speaker", "text", "decision", "words"}
    assert required_keys.issubset(set(segment.keys()))
    assert isinstance(segment["words"], list)
