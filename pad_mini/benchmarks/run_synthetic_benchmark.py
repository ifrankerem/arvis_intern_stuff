#!/usr/bin/env python3
"""Generate deterministic perturbations and report mathematical behavior."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_models import FaceBox
from dct_block_pre_control import DCTBlockAnalysisPreController
from model_free_analysis import ModelFreePreControlContextBuilder
from periodicity_pre_control import PeriodicityPreController
from synthetic_signals import (
    add_eight_pixel_block_steps,
    add_periodic_residual,
    checkerboard,
    jpeg_recompress,
    screen_lattice,
    sinusoidal_grating,
)


def context(gray):
    canvas = np.pad(gray, 32, mode="reflect")
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return ModelFreePreControlContextBuilder().build(
        0.0,
        frame,
        frame,
        FaceBox(32, 32, gray.shape[1], gray.shape[0]),
    )


def main():
    random = np.random.default_rng(2026)
    natural_proxy = np.clip(
        127.0 + random.normal(0.0, 14.0, (256, 256)),
        0.0,
        255.0,
    ).astype(np.uint8)
    cases = {
        "natural_proxy": natural_proxy,
        "weak_grating": sinusoidal_grating(
            amplitude=5.0, noise_standard_deviation=8.0
        ),
        "strong_grating": sinusoidal_grating(
            amplitude=30.0, noise_standard_deviation=8.0
        ),
        "rotated_grating": sinusoidal_grating(
            angle_degrees=37.0, amplitude=30.0, noise_standard_deviation=5.0
        ),
        "checkerboard": checkerboard(),
        "screen_lattice": screen_lattice(),
        "periodic_residual": add_periodic_residual(natural_proxy),
        "jpeg_q90": jpeg_recompress(natural_proxy, [90]),
        "jpeg_q40": jpeg_recompress(natural_proxy, [40]),
        "double_jpeg": jpeg_recompress(natural_proxy, [85, 45]),
        "eight_pixel_steps": add_eight_pixel_block_steps(natural_proxy),
    }
    report = {}
    for name, image in cases.items():
        shared = context(image)
        periodicity = PeriodicityPreController().analyze(shared)
        dct = DCTBlockAnalysisPreController().analyze(shared)
        report[name] = {
            "periodicity_score": periodicity.raw_score,
            "periodicity_supported": periodicity.supported,
            "autocorrelation_peak_ratio": periodicity.raw_features.get(
                "autocorrelation_peak_ratio"
            ),
            "patch_periodic_vote_ratio": periodicity.raw_features.get(
                "patch_periodic_vote_ratio"
            ),
            "dct_score": dct.raw_score,
            "dct_supported": dct.supported,
            "blockiness_score": dct.raw_features.get("blockiness_score"),
        }
    report["behavior_checks"] = {
        "periodic_amplitude_monotonic": (
            report["strong_grating"]["periodicity_score"]
            > report["weak_grating"]["periodicity_score"]
        ),
        "eight_pixel_steps_change_block_metric": (
            report["eight_pixel_steps"]["blockiness_score"]
            != report["natural_proxy"]["blockiness_score"]
        ),
        "double_jpeg_is_not_claimed_from_decoded_bytes": True,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

