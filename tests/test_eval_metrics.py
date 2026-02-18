from meeting_id import eval as eval_module


def test_asr_metrics_basic():
    metrics = eval_module.compute_asr_metrics(
        reference_text="merhaba dunya",
        hypothesis_text="merhaba",
    )
    assert metrics["wer"] == 0.5
    assert metrics["cer"] > 0.0


def test_diarization_metrics_basic():
    ref = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]
    hyp = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "A"},
    ]
    metrics = eval_module.compute_diarization_metrics(ref, hyp)
    assert 0.0 <= metrics["der"] <= 1.0
    assert 0.0 <= metrics["jer"] <= 1.0
    assert metrics["der"] > 0.0


def test_identity_metrics_and_eer():
    scores = [0.9, 0.8, 0.3, 0.2]
    labels = [True, True, False, False]

    far_frr = eval_module.compute_far_frr(scores, labels, threshold=0.5)
    assert far_frr["far"] == 0.0
    assert far_frr["frr"] == 0.0

    eer = eval_module.compute_eer(scores, labels)
    assert 0.0 <= eer["eer"] <= 1.0
    assert "threshold" in eer


def test_confusion_matrix_shape():
    matrix = eval_module.compute_confusion_matrix(
        true_labels=["Alice", "Bob", "Alice"],
        predicted_labels=["Alice", "UNKNOWN", "Bob"],
    )
    assert matrix["Alice"]["Alice"] == 1
    assert matrix["Alice"]["Bob"] == 1
    assert matrix["Bob"]["UNKNOWN"] == 1
