"""Deterministic autocorrelation and real-cepstrum periodicity evidence."""

from collections import deque
import math

import cv2
import numpy as np

import config
from deterministic_normalization import smoothstep
from model_free_analysis import ModelFreeAnalysisResult


class PeriodicityPreController:
    """Corroborate periodic texture in lag and quefrency domains.

    This method is deliberately fused inside the frequency family. An
    autocorrelation or cepstral peak is not an independent vote from FFT
    moire evidence and is never interpreted as proof of a replay by itself.
    """

    MODULE_NAME = "Autocorrelation / Cepstrum"

    def __init__(self):
        self.score_history = deque(
            maxlen=config.EXPERIMENTAL_PERIODICITY_HISTORY_SIZE
        )
        self.invalid_streak = 0
        self.debug_images = {}

    def analyze(self, context):
        self.debug_images = {}
        if context is None or not context.face_quality_valid:
            return self._unsupported(
                context.quality_reason
                if context is not None
                else "shared context unavailable"
            )

        roi = context.rois.get("raw_face") if context.rois else None
        source = roi.image if roi is not None and roi.valid else (
            context.original_high_resolution_face_crop
        )
        if source is None or source.size == 0:
            return self._unsupported("raw face ROI unavailable")
        if min(source.shape[:2]) < config.EXPERIMENTAL_PERIODICITY_MINIMUM_SIDE:
            return self._unsupported("raw face ROI too small")

        if source.ndim == 3:
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            gray = source.copy()
        gray = gray.astype(np.float32)

        clipped_ratio = float(
            np.mean((gray <= 2.0) | (gray >= 253.0))
        )
        if clipped_ratio >= config.EXPERIMENTAL_PERIODICITY_UNSUPPORTED_CLIPPING:
            return self._unsupported(
                "exposure clipping prevents periodicity interpretation",
                raw_features={"clipped_pixel_ratio": clipped_ratio},
            )

        working, resize_scale = self._bounded_image(gray)
        residual = working - cv2.GaussianBlur(
            working,
            (0, 0),
            config.EXPERIMENTAL_PERIODICITY_RESIDUAL_SIGMA,
        )
        residual_std = float(np.std(residual))
        if residual_std < config.EXPERIMENTAL_PERIODICITY_MINIMUM_RESIDUAL_STD:
            return self._unsupported(
                "residual contrast is insufficient",
                raw_features={
                    "clipped_pixel_ratio": clipped_ratio,
                    "residual_standard_deviation": residual_std,
                },
            )

        global_metrics, autocorrelation, cepstrum = self._domain_metrics(
            residual
        )
        patch_metrics, heatmap = self._patch_analysis(residual)
        metrics = {
            **global_metrics,
            **patch_metrics,
            "clipped_pixel_ratio": clipped_ratio,
            "residual_standard_deviation": residual_std,
            "source_width": int(gray.shape[1]),
            "source_height": int(gray.shape[0]),
            "analysis_width": int(working.shape[1]),
            "analysis_height": int(working.shape[0]),
            "resize_scale": resize_scale,
            "input_transform": (
                "raw_guide_grayscale_high_pass"
                if resize_scale == 1.0
                else "downscaled_raw_guide_grayscale_high_pass"
            ),
        }

        autocorrelation_score = smoothstep(
            metrics["autocorrelation_peak_ratio"],
            config.EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_START,
            config.EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_FULL,
        )
        cepstrum_score = smoothstep(
            metrics["cepstral_peak_prominence_z"],
            config.EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_START,
            config.EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_FULL,
        )
        vote_score = smoothstep(
            metrics["patch_periodic_vote_ratio"],
            config.EXPERIMENTAL_PERIODICITY_PATCH_VOTE_START,
            config.EXPERIMENTAL_PERIODICITY_PATCH_VOTE_FULL,
        )
        agreement_score = metrics["cross_domain_period_agreement"]
        raw_score = 100.0 * (
            config.EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_WEIGHT
            * autocorrelation_score
            + config.EXPERIMENTAL_PERIODICITY_CEPSTRUM_WEIGHT
            * cepstrum_score
            + config.EXPERIMENTAL_PERIODICITY_PATCH_VOTE_WEIGHT
            * vote_score
            + config.EXPERIMENTAL_PERIODICITY_AGREEMENT_WEIGHT
            * agreement_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 100.0))
        self.score_history.append(raw_score)
        self.invalid_streak = 0
        stabilized_score = float(np.median(self.score_history))

        patch_count = int(metrics["patch_count"])
        patch_coverage = min(
            1.0,
            patch_count / max(1.0, config.EXPERIMENTAL_PERIODICITY_TARGET_PATCH_COUNT),
        )
        exposure_reliability = 1.0 - min(
            1.0,
            clipped_ratio
            / config.EXPERIMENTAL_PERIODICITY_UNSUPPORTED_CLIPPING,
        )
        domain_reliability = 0.5 + 0.5 * agreement_score
        reliability = float(
            np.clip(
                patch_coverage * exposure_reliability * domain_reliability,
                0.0,
                config.EXPERIMENTAL_PERIODICITY_MAXIMUM_RELIABILITY,
            )
        )

        triggered = bool(
            stabilized_score >= config.EXPERIMENTAL_PERIODICITY_TRIGGER_SCORE
            and metrics["patch_periodic_vote_ratio"]
            >= config.EXPERIMENTAL_PERIODICITY_REQUIRED_PATCH_VOTE
            and agreement_score
            >= config.EXPERIMENTAL_PERIODICITY_REQUIRED_DOMAIN_AGREEMENT
        )
        if triggered:
            status = "Stable periodic recapture evidence"
            reason_codes = ["PERIODIC_LAG_AND_QUEFRENCY_PEAKS"]
            evidence = [
                "Autocorrelation and cepstrum agree on repeated spatial structure",
                "Periodic evidence is repeated across local patches",
            ]
        elif raw_score >= config.EXPERIMENTAL_PERIODICITY_SUPPORTING_SCORE:
            status = "Supporting periodic evidence"
            reason_codes = ["WEAK_PERIODIC_STRUCTURE"]
            evidence = [
                "Some periodic structure is present but cross-check requirements are incomplete"
            ]
        else:
            status = "No stable periodicity"
            reason_codes = []
            evidence = [
                "No stable cross-domain periodic structure was measured"
            ]

        metrics.update(
            {
                "autocorrelation_component": autocorrelation_score,
                "cepstrum_component": cepstrum_score,
                "patch_vote_component": vote_score,
                "agreement_component": agreement_score,
                "normalization_mode": "experimental_monotonic_smoothstep",
            }
        )
        self.debug_images = {
            "autocorrelation_map": self._normalize_map(autocorrelation),
            "cepstrum_map": self._normalize_map(cepstrum),
            "patch_periodicity_heatmap": cv2.applyColorMap(
                self._normalize_map(heatmap),
                cv2.COLORMAP_TURBO,
            ),
        }
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=metrics,
            raw_score=raw_score,
            stabilized_score=stabilized_score,
            confidence=reliability,
            status=status,
            evidence=evidence,
            warnings=[
                "Experimental thresholds require device-specific calibration",
                "Natural repeated texture can produce the same measurements",
            ],
            debug_data={
                "quality_status": "Sufficient",
                "possible_attack": "screen_or_print_recap" if triggered else "none",
                "autocorrelation_peak_coordinate": metrics[
                    "autocorrelation_peak_coordinate"
                ],
                "cepstral_peak_coordinate": metrics[
                    "cepstral_peak_coordinate"
                ],
            },
            calibrated=False,
            evidence_family="frequency",
            attack_targets=["replay_screen", "print_attack", "recapture"],
            reason_codes=reason_codes,
            human_explanation=" ".join(evidence),
            triggered=triggered,
        )

    def get_debug_images(self):
        return dict(self.debug_images)

    def reset(self):
        self.score_history.clear()
        self.invalid_streak = 0
        self.debug_images = {}

    def _unsupported(self, reason, raw_features=None):
        self.invalid_streak += 1
        if self.invalid_streak >= config.EXPERIMENTAL_PERIODICITY_INVALID_RESET_FRAMES:
            self.score_history.clear()
        result = ModelFreeAnalysisResult.unavailable(
            self.MODULE_NAME,
            reason,
            debug_data={
                "quality_status": "Insufficient",
                "possible_attack": "none",
            },
            calibrated=False,
        )
        result.raw_features = dict(raw_features or {})
        result.evidence_family = "frequency"
        result.attack_targets = ["replay_screen", "print_attack", "recapture"]
        result.reason_codes = ["PERIODICITY_UNSUPPORTED"]
        result.human_explanation = reason
        return result

    def _bounded_image(self, gray):
        maximum = int(config.EXPERIMENTAL_PERIODICITY_MAXIMUM_SIDE)
        height, width = gray.shape[:2]
        scale = min(1.0, maximum / float(max(height, width)))
        if scale == 1.0:
            return gray.copy(), 1.0
        resized = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, float(scale)

    def _domain_metrics(self, residual):
        height, width = residual.shape[:2]
        normalized = residual - float(np.mean(residual))
        deviation = float(np.std(normalized))
        if deviation > 1e-9:
            normalized /= deviation
        window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
        spectrum = np.fft.fft2(normalized * window)
        power = np.abs(spectrum) ** 2

        autocorrelation = np.fft.fftshift(np.fft.ifft2(power).real)
        center = (height // 2, width // 2)
        center_value = max(float(autocorrelation[center]), 1e-9)
        autocorrelation = autocorrelation / center_value

        log_magnitude = np.log1p(np.abs(spectrum))
        cepstrum = np.abs(
            np.fft.fftshift(np.fft.ifft2(log_magnitude).real)
        )
        mask = self._lag_mask(height, width)
        ac_peak = self._prominent_peak(autocorrelation, mask)
        cep_peak = self._prominent_peak(cepstrum, mask)

        ac_period = ac_peak["radius"]
        cep_period = cep_peak["radius"]
        period_scale = max(ac_period, cep_period, 1.0)
        agreement = math.exp(
            -abs(ac_period - cep_period)
            / (config.EXPERIMENTAL_PERIODICITY_PERIOD_TOLERANCE * period_scale)
        )
        metrics = {
            "autocorrelation_peak_ratio": max(0.0, ac_peak["value"]),
            "autocorrelation_peak_prominence_z": ac_peak["prominence_z"],
            "autocorrelation_peak_coordinate": ac_peak["coordinate"],
            "autocorrelation_dominant_period_pixels": ac_period,
            "cepstral_peak_ratio": cep_peak["ratio"],
            "cepstral_peak_prominence_z": cep_peak["prominence_z"],
            "cepstral_peak_coordinate": cep_peak["coordinate"],
            "cepstral_dominant_period_pixels": cep_period,
            "cross_domain_period_agreement": float(np.clip(agreement, 0.0, 1.0)),
        }
        return metrics, autocorrelation, cepstrum

    def _patch_analysis(self, residual):
        height, width = residual.shape[:2]
        minimum = min(height, width)
        patch_sizes = sorted(
            set(
                max(
                    48,
                    min(
                        minimum,
                        int(round((minimum * ratio) / 16.0)) * 16,
                    ),
                )
                for ratio in config.EXPERIMENTAL_PERIODICITY_PATCH_SCALE_RATIOS
            )
        )
        heat_sum = np.zeros((height, width), dtype=np.float32)
        heat_count = np.zeros((height, width), dtype=np.float32)
        entries = []
        for patch_size in patch_sizes:
            # Non-overlapping voting keeps the CPU path bounded. The final
            # edge-aligned patch is still included by _grid_positions.
            stride = patch_size
            y_positions = self._grid_positions(height, patch_size, stride)
            x_positions = self._grid_positions(width, patch_size, stride)
            for y in y_positions:
                for x in x_positions:
                    patch = residual[y : y + patch_size, x : x + patch_size]
                    metrics, _, _ = self._domain_metrics(patch)
                    ac_component = smoothstep(
                        metrics["autocorrelation_peak_ratio"],
                        config.EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_START,
                        config.EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_FULL,
                    )
                    cep_component = smoothstep(
                        metrics["cepstral_peak_prominence_z"],
                        config.EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_START,
                        config.EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_FULL,
                    )
                    score = float(
                        math.sqrt(max(0.0, ac_component * cep_component))
                        * metrics["cross_domain_period_agreement"]
                    )
                    periodic = bool(
                        metrics["autocorrelation_peak_ratio"]
                        >= config.EXPERIMENTAL_PERIODICITY_PATCH_AC_MINIMUM
                        and metrics["cepstral_peak_prominence_z"]
                        >= config.EXPERIMENTAL_PERIODICITY_PATCH_CEPSTRUM_Z_MINIMUM
                    )
                    entries.append(
                        {
                            "x": int(x),
                            "y": int(y),
                            "size": int(patch_size),
                            "score": score,
                            "periodic": periodic,
                            "period": metrics[
                                "autocorrelation_dominant_period_pixels"
                            ],
                        }
                    )
                    heat_sum[y : y + patch_size, x : x + patch_size] += score
                    heat_count[y : y + patch_size, x : x + patch_size] += 1.0

        heatmap = heat_sum / np.maximum(heat_count, 1.0)
        periodic_entries = [entry for entry in entries if entry["periodic"]]
        vote_ratio = len(periodic_entries) / max(1.0, float(len(entries)))
        periods = np.asarray(
            [entry["period"] for entry in periodic_entries],
            dtype=np.float64,
        )
        if periods.size >= 2 and float(np.mean(periods)) > 1e-9:
            period_cv = float(np.std(periods) / np.mean(periods))
            lattice_regularity = float(np.exp(-period_cv / 0.35))
        elif periods.size == 1:
            period_cv = 0.0
            lattice_regularity = 0.25
        else:
            period_cv = 0.0
            lattice_regularity = 0.0
        return {
            "patch_count": int(len(entries)),
            "periodic_patch_count": int(len(periodic_entries)),
            "patch_periodic_vote_ratio": float(vote_ratio),
            "patch_score_median": float(
                np.median([entry["score"] for entry in entries])
                if entries
                else 0.0
            ),
            "patch_period_coefficient_of_variation": period_cv,
            "lattice_regularity": lattice_regularity,
            "patch_measurements": entries,
        }, heatmap

    def _lag_mask(self, height, width):
        yy, xx = np.indices((height, width), dtype=np.float32)
        cy = height // 2
        cx = width // 2
        radius = np.hypot(xx - cx, yy - cy)
        maximum = min(height, width) * config.EXPERIMENTAL_PERIODICITY_MAXIMUM_LAG_RATIO
        return (
            (radius >= config.EXPERIMENTAL_PERIODICITY_MINIMUM_LAG_PIXELS)
            & (radius <= maximum)
        )

    def _prominent_peak(self, values, mask):
        finite_values = np.asarray(values, dtype=np.float64)
        background = finite_values[mask]
        median = float(np.median(background))
        mad = float(np.median(np.abs(background - median)))
        scale = max(1.4826 * mad, float(np.std(background)) * 0.05, 1e-9)
        local_maximum = finite_values >= cv2.dilate(
            finite_values.astype(np.float32),
            np.ones((5, 5), dtype=np.uint8),
        )
        candidates = np.argwhere(mask & local_maximum)
        if candidates.size == 0:
            coordinate = np.unravel_index(
                int(np.argmax(np.where(mask, finite_values, -np.inf))),
                finite_values.shape,
            )
        else:
            candidate_values = finite_values[candidates[:, 0], candidates[:, 1]]
            coordinate = tuple(candidates[int(np.argmax(candidate_values))])
        y, x = int(coordinate[0]), int(coordinate[1])
        value = float(finite_values[y, x])
        cy, cx = finite_values.shape[0] // 2, finite_values.shape[1] // 2
        return {
            "coordinate": [x, y],
            "radius": float(math.hypot(x - cx, y - cy)),
            "value": value,
            "ratio": float(max(0.0, (value - median) / (abs(median) + scale))),
            "prominence_z": float(max(0.0, (value - median) / scale)),
        }

    @staticmethod
    def _grid_positions(length, patch_size, stride):
        if patch_size >= length:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    @staticmethod
    def _normalize_map(values):
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0 or not np.any(np.isfinite(array)):
            return np.zeros((256, 256), dtype=np.uint8)
        finite = array[np.isfinite(array)]
        lower, upper = np.percentile(finite, [1.0, 99.5])
        if upper <= lower:
            return np.zeros(array.shape, dtype=np.uint8)
        normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
        return np.round(normalized * 255.0).astype(np.uint8)
