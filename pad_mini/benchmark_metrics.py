"""Dependency-free ISO-style PAD metrics for held-out benchmark scores."""

from collections import defaultdict

import numpy as np


def evaluate_pad_scores(records, threshold=50.0, selected_fprs=(0.1, 0.01, 0.001)):
    """Evaluate supported sample scores; abstentions remain separately visible."""
    records = list(records)
    supported = [
        item
        for item in records
        if item.get("score") is not None and bool(item.get("score_valid", True))
    ]
    attacks = [item for item in supported if item["label"] == "attack"]
    bona_fide = [item for item in supported if item["label"] == "bona_fide"]
    apcer = _rate(
        sum(float(item["score"]) < threshold for item in attacks),
        len(attacks),
    )
    bpcer = _rate(
        sum(float(item["score"]) >= threshold for item in bona_fide),
        len(bona_fide),
    )
    acer = (
        None if apcer is None or bpcer is None else 0.5 * (apcer + bpcer)
    )
    roc = _roc_points(supported)
    eer = _eer(roc)
    per_attack = _per_attack_confusion(attacks, threshold)
    abstentions = [
        item
        for item in records
        if item.get("classification")
        in {
            "INSUFFICIENT_QUALITY",
            "INSUFFICIENT_EVIDENCE",
            "UNSUPPORTED_CAPTURE",
        }
    ]
    unsupported = [
        item
        for item in records
        if item.get("classification") == "UNSUPPORTED_CAPTURE"
    ]
    latencies = [
        float(item["runtime_ms"])
        for item in records
        if item.get("runtime_ms") is not None
    ]
    return {
        "threshold": float(threshold),
        "sample_count": len(records),
        "supported_score_count": len(supported),
        "attack_count": len(attacks),
        "bona_fide_count": len(bona_fide),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "EER": eer,
        "abstention_rate": _rate(len(abstentions), len(records)),
        "unsupported_rate": _rate(len(unsupported), len(records)),
        "tpr_at_selected_fpr": {
            str(target): _tpr_at_fpr(roc, target) for target in selected_fprs
        },
        "per_attack_confusion": per_attack,
        "latency_ms": _latency_summary(latencies),
        "roc": roc,
    }


def _roc_points(records):
    if not records:
        return []
    attacks = sum(item["label"] == "attack" for item in records)
    bona_fide = sum(item["label"] == "bona_fide" for item in records)
    if attacks == 0 or bona_fide == 0:
        return []
    scores = sorted({float(item["score"]) for item in records}, reverse=True)
    thresholds = [scores[0] + 1.0] + scores + [scores[-1] - 1.0]
    points = []
    for threshold in thresholds:
        true_positive = sum(
            item["label"] == "attack" and float(item["score"]) >= threshold
            for item in records
        )
        false_positive = sum(
            item["label"] == "bona_fide" and float(item["score"]) >= threshold
            for item in records
        )
        points.append(
            {
                "threshold": threshold,
                "fpr": false_positive / bona_fide,
                "tpr": true_positive / attacks,
            }
        )
    return points


def _eer(roc):
    if not roc:
        return None
    point = min(roc, key=lambda item: abs(item["fpr"] - (1.0 - item["tpr"])))
    return 0.5 * (point["fpr"] + 1.0 - point["tpr"])


def _tpr_at_fpr(roc, target):
    eligible = [item["tpr"] for item in roc if item["fpr"] <= target]
    return max(eligible) if eligible else None


def _per_attack_confusion(attacks, threshold):
    groups = defaultdict(list)
    for item in attacks:
        groups[str(item.get("attack_species", "unspecified"))].append(item)
    return {
        name: {
            "count": len(items),
            "detected": sum(float(item["score"]) >= threshold for item in items),
            "missed": sum(float(item["score"]) < threshold for item in items),
            "APCER": _rate(
                sum(float(item["score"]) < threshold for item in items),
                len(items),
            ),
        }
        for name, items in sorted(groups.items())
    }


def _latency_summary(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator
