#!/usr/bin/env python3
"""Run the deterministic PreControl over an image/video benchmark manifest."""

import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_metrics import evaluate_pad_scores
from model_free_pre_control_application import ModelFreePreControlApplication


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument(
        "--split",
        default="test",
        help="Only rows with this split are evaluated (default: test)",
    )
    parser.add_argument("--maximum-frames", type=int, default=120)
    return parser.parse_args()


def load_manifest(path, split):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    required = {"path", "label", "attack_species", "capture_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest requires columns: " + ", ".join(sorted(required)))
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("no rows found for split: " + split)
    return selected


def evaluate_row(row, maximum_frames):
    path = Path(row["path"]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    application = ModelFreePreControlApplication()
    frame_count = 0
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            application.process_frame(
                image,
                frame_metadata={
                    "source_kind": "decoded_image",
                    "timestamp_basis": "benchmark_processing_monotonic",
                    "encoded_bytes_available": False,
                },
            )
            frame_count = 1
        else:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError("cannot decode benchmark input: " + str(path))
            nominal_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            while frame_count < maximum_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                application.process_frame(
                    frame,
                    frame_metadata={
                        "source_kind": "decoded_video",
                        "timestamp_basis": "container_position",
                        "source_position_ms": position_ms,
                        "acquisition_monotonic_s": time.monotonic(),
                        "nominal_fps": nominal_fps if nominal_fps > 0.0 else None,
                        "encoded_bytes_available": False,
                    },
                )
                frame_count += 1
            capture.release()
        decision = application.latest_precontrol_decision
        if decision is None:
            raise RuntimeError("PreControl did not produce a decision")
        result = decision.to_dict()
        return {
            "capture_id": row["capture_id"],
            "path": str(path),
            "label": row["label"],
            "attack_species": row["attack_species"],
            "frame_count": frame_count,
            "score": result["overall_risk_0_100"],
            "score_valid": result["score_valid"],
            "classification": result["classification"],
            "reliability": result["overall_reliability"],
            "runtime_ms": result["runtime_ms"],
            "decision": result,
            "method_runtime_ms": {
                name: method.runtime_ms
                for name, method in application.latest_pre_control_results.items()
            },
        }
    finally:
        application.shutdown()


def main():
    arguments = parse_arguments()
    rows = load_manifest(arguments.manifest, arguments.split)
    records = [evaluate_row(row, arguments.maximum_frames) for row in rows]
    metrics = evaluate_pad_scores(records, threshold=arguments.threshold)
    metrics["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "protocol": {
            "manifest": str(arguments.manifest.resolve()),
            "split": arguments.split,
            "threshold": arguments.threshold,
            "note": (
                "Threshold evaluation is offline benchmarking only; it does not "
                "override production abstention or calibration rules."
            ),
        },
        "metrics": metrics,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(str(arguments.output.resolve()))


if __name__ == "__main__":
    main()

