import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _levenshtein_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr_row = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr_row.append(
                min(
                    prev_row[j] + 1,
                    curr_row[j - 1] + 1,
                    prev_row[j - 1] + cost,
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """
    Compute Word Error Rate (WER).

    Input schema:
    - reference_text: ground-truth text string.
    - hypothesis_text: predicted text string.
    """
    ref_tokens = _normalize_whitespace(reference_text).split()
    hyp_tokens = _normalize_whitespace(hypothesis_text).split()
    distance = _levenshtein_distance(ref_tokens, hyp_tokens)
    return _safe_div(distance, len(ref_tokens))


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    """
    Compute Character Error Rate (CER), excluding whitespace.

    Input schema:
    - reference_text: ground-truth text string.
    - hypothesis_text: predicted text string.
    """
    ref_chars = list((_normalize_whitespace(reference_text)).replace(" ", ""))
    hyp_chars = list((_normalize_whitespace(hypothesis_text)).replace(" ", ""))
    distance = _levenshtein_distance(ref_chars, hyp_chars)
    return _safe_div(distance, len(ref_chars))


def compute_asr_metrics(reference_text: str, hypothesis_text: str) -> Dict[str, float]:
    """
    Compute ASR metrics.

    Input schema:
    - reference_text: ground-truth text.
    - hypothesis_text: predicted text.
    """
    return {
        "wer": compute_wer(reference_text, hypothesis_text),
        "cer": compute_cer(reference_text, hypothesis_text),
    }


@dataclass(frozen=True)
class _Interval:
    start: float
    end: float
    label: str


def _prepare_intervals(segments: Iterable[Dict[str, Any]]) -> List[_Interval]:
    prepared: List[_Interval] = []
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        label = str(seg.get("speaker", "UNKNOWN"))
        if end > start:
            prepared.append(_Interval(start=start, end=end, label=label))
    return prepared


def _label_at_time(intervals: Sequence[_Interval], t: float) -> Optional[str]:
    for iv in intervals:
        if iv.start <= t < iv.end:
            return iv.label
    return None


def compute_diarization_metrics(
    reference_segments: Sequence[Dict[str, Any]],
    hypothesis_segments: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute basic diarization metrics (DER/JER approximation).

    Input schema:
    - reference_segments: [{"start": float, "end": float, "speaker": str}, ...]
    - hypothesis_segments: [{"start": float, "end": float, "speaker": str}, ...]
    """
    ref = _prepare_intervals(reference_segments)
    hyp = _prepare_intervals(hypothesis_segments)

    boundaries = set()
    for iv in ref:
        boundaries.add(iv.start)
        boundaries.add(iv.end)
    for iv in hyp:
        boundaries.add(iv.start)
        boundaries.add(iv.end)

    if not boundaries:
        return {"der": 0.0, "jer": 0.0}

    ordered = sorted(boundaries)
    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    ref_total = 0.0

    for i in range(len(ordered) - 1):
        left = ordered[i]
        right = ordered[i + 1]
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        duration = right - left
        ref_label = _label_at_time(ref, midpoint)
        hyp_label = _label_at_time(hyp, midpoint)

        if ref_label is not None:
            ref_total += duration

        if ref_label is None and hyp_label is None:
            continue
        if ref_label is not None and hyp_label is None:
            missed += duration
            continue
        if ref_label is None and hyp_label is not None:
            false_alarm += duration
            continue
        if ref_label != hyp_label:
            confusion += duration

    der = _safe_div(missed + false_alarm + confusion, ref_total)

    ref_speakers = sorted({iv.label for iv in ref})
    per_speaker_error: List[float] = []
    for speaker in ref_speakers:
        speaker_ref = [iv for iv in ref if iv.label == speaker]
        speaker_hyp = [iv for iv in hyp if iv.label == speaker]
        speaker_boundaries = sorted(
            {b for iv in speaker_ref + speaker_hyp for b in (iv.start, iv.end)}
        )
        if len(speaker_boundaries) < 2:
            per_speaker_error.append(0.0)
            continue

        intersection = 0.0
        union = 0.0
        for i in range(len(speaker_boundaries) - 1):
            left = speaker_boundaries[i]
            right = speaker_boundaries[i + 1]
            if right <= left:
                continue
            midpoint = (left + right) / 2.0
            duration = right - left
            in_ref = _label_at_time(speaker_ref, midpoint) is not None
            in_hyp = _label_at_time(speaker_hyp, midpoint) is not None
            if in_ref and in_hyp:
                intersection += duration
            if in_ref or in_hyp:
                union += duration
        speaker_jaccard = _safe_div(intersection, union)
        per_speaker_error.append(1.0 - speaker_jaccard)

    jer = float(sum(per_speaker_error) / len(per_speaker_error)) if per_speaker_error else 0.0
    return {"der": der, "jer": jer}


def compute_far_frr(
    scores: Sequence[float], labels: Sequence[bool], threshold: float
) -> Dict[str, float]:
    """
    Compute FAR/FRR at a fixed threshold.

    Input schema:
    - scores: similarity scores, higher means more likely genuine.
    - labels: True for genuine pair, False for impostor pair.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    genuine_total = 0
    impostor_total = 0
    false_rejects = 0
    false_accepts = 0

    for score, is_genuine in zip(scores, labels):
        accepted = float(score) >= float(threshold)
        if is_genuine:
            genuine_total += 1
            if not accepted:
                false_rejects += 1
        else:
            impostor_total += 1
            if accepted:
                false_accepts += 1

    return {
        "far": _safe_div(false_accepts, impostor_total),
        "frr": _safe_div(false_rejects, genuine_total),
        "false_accepts": float(false_accepts),
        "false_rejects": float(false_rejects),
        "impostor_total": float(impostor_total),
        "genuine_total": float(genuine_total),
    }


def compute_eer(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    """
    Compute Equal Error Rate (EER) by threshold scan.

    Input schema:
    - scores: similarity scores.
    - labels: True for genuine, False for impostor.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        return {"eer": 0.0, "threshold": 0.0, "far": 0.0, "frr": 0.0}

    unique_thresholds = sorted(set(float(s) for s in scores))
    candidate_thresholds = [unique_thresholds[0] - 1e-6] + unique_thresholds + [unique_thresholds[-1] + 1e-6]

    best = None
    for threshold in candidate_thresholds:
        metrics = compute_far_frr(scores, labels, threshold)
        gap = abs(metrics["far"] - metrics["frr"])
        if best is None or gap < best["gap"]:
            best = {
                "gap": gap,
                "threshold": threshold,
                "far": metrics["far"],
                "frr": metrics["frr"],
            }

    eer = (best["far"] + best["frr"]) / 2.0 if best else 0.0
    return {
        "eer": float(eer),
        "threshold": float(best["threshold"]) if best else 0.0,
        "far": float(best["far"]) if best else 0.0,
        "frr": float(best["frr"]) if best else 0.0,
    }


def compute_confusion_matrix(
    true_labels: Sequence[str], predicted_labels: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix in dict-of-dicts form.

    Input schema:
    - true_labels: ground-truth identity labels.
    - predicted_labels: predicted identity labels.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")

    label_set = sorted(set(str(x) for x in true_labels) | set(str(x) for x in predicted_labels))
    matrix: Dict[str, Dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in label_set} for true_label in label_set
    }
    for true_label, pred_label in zip(true_labels, predicted_labels):
        matrix[str(true_label)][str(pred_label)] += 1
    return matrix


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_segments(entry: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    inline = entry.get(key)
    file_key = f"{key}_path"
    if inline is not None:
        if not isinstance(inline, list):
            raise ValueError(f"{key} must be a list when provided inline")
        return inline
    if entry.get(file_key):
        data = _load_json(str(entry[file_key]))
        if not isinstance(data, list):
            raise ValueError(f"{file_key} must point to a JSON list")
        return data
    return []


def run_manifest_evaluation(
    manifest_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run multi-sample evaluation from manifest and compute aggregate metrics.

    Manifest schema:
    {
      "samples": [
        {
          "id": "sample-1",
          "reference_text": "...",
          "hypothesis_text": "...",
          "reference_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
          "hypothesis_segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
          "reference_segments_path": "path/to/ref_segments.json",
          "hypothesis_segments_path": "path/to/hyp_segments.json",
          "identity_scores": [0.91, 0.22],
          "identity_labels": [true, false],
          "identity_threshold": 0.5,
          "true_identity_labels": ["Alice", "Bob"],
          "pred_identity_labels": ["Alice", "UNKNOWN"]
        }
      ]
    }
    """
    payload = _load_json(manifest_path)
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    if not isinstance(samples, list):
        raise ValueError("manifest must contain a list or a {'samples': [...]} object")

    report_samples: List[Dict[str, Any]] = []
    all_wers: List[float] = []
    all_cers: List[float] = []
    all_ders: List[float] = []
    all_jers: List[float] = []
    aggregate_scores: List[float] = []
    aggregate_labels: List[bool] = []

    for idx, entry in enumerate(samples):
        if not isinstance(entry, dict):
            raise ValueError("Each sample entry must be an object")

        sample_id = str(entry.get("id", f"sample-{idx+1}"))
        sample_report: Dict[str, Any] = {"id": sample_id}

        ref_text = entry.get("reference_text")
        hyp_text = entry.get("hypothesis_text")
        if ref_text is not None and hyp_text is not None:
            asr_metrics = compute_asr_metrics(str(ref_text), str(hyp_text))
            sample_report["asr"] = asr_metrics
            all_wers.append(asr_metrics["wer"])
            all_cers.append(asr_metrics["cer"])

        ref_segments = _resolve_segments(entry, "reference_segments")
        hyp_segments = _resolve_segments(entry, "hypothesis_segments")
        if ref_segments or hyp_segments:
            diar_metrics = compute_diarization_metrics(ref_segments, hyp_segments)
            sample_report["diarization"] = diar_metrics
            all_ders.append(diar_metrics["der"])
            all_jers.append(diar_metrics["jer"])

        identity_scores = entry.get("identity_scores")
        identity_labels = entry.get("identity_labels")
        if identity_scores is not None and identity_labels is not None:
            threshold = float(entry.get("identity_threshold", 0.5))
            far_frr = compute_far_frr(identity_scores, identity_labels, threshold)
            eer = compute_eer(identity_scores, identity_labels)
            sample_report["identity"] = {
                "threshold_metrics": far_frr,
                "eer": eer,
            }
            aggregate_scores.extend(float(x) for x in identity_scores)
            aggregate_labels.extend(bool(x) for x in identity_labels)

        true_identity_labels = entry.get("true_identity_labels")
        pred_identity_labels = entry.get("pred_identity_labels")
        if true_identity_labels is not None and pred_identity_labels is not None:
            sample_report["confusion_matrix"] = compute_confusion_matrix(
                true_identity_labels, pred_identity_labels
            )

        report_samples.append(sample_report)

    aggregate: Dict[str, Any] = {
        "sample_count": len(report_samples),
        "asr": {"wer_mean": _mean(all_wers), "cer_mean": _mean(all_cers)},
        "diarization": {"der_mean": _mean(all_ders), "jer_mean": _mean(all_jers)},
    }
    if aggregate_scores and aggregate_labels:
        aggregate["identity"] = {"eer": compute_eer(aggregate_scores, aggregate_labels)}

    report = {"manifest_path": str(manifest_path), "aggregate": aggregate, "samples": report_samples}

    if output_path:
        output_parent = Path(output_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
