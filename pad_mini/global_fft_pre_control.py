"""Model veya MediaPipe kullanmayan Global FFT PreControl analizi."""

from collections import deque
import math
import time

import cv2
import numpy as np

import config
from model_free_analysis import (
    ModelFreeAnalysisResult,
    ModelFreePreControlContextBuilder,
)


class GlobalFFTPreController:
    """Ortak context'teki FFT'den genel frekans dagilimini aciklar.

    Bu modul yeni FFT veya resize islemi yapmaz ve real/fake siniflandirmasi
    uretmez. Mevcut heuristic feature profilleri deneysel ve kalibrasyonsuzdur.
    """

    MODULE_NAME = "Global FFT"

    def __init__(self):
        self.score_history = deque(
            maxlen=config.EXPERIMENTAL_FFT_HISTORY_SIZE
        )
        self.invalid_frame_streak = 0
        self.latest_context = None

        # Geriye donuk debug/kaydetme erisimi. Alanlar context dizilerine
        # referans verir; burada FFT hesaplanmaz.
        self.latest_face_crop = None
        self.latest_standardized_crop = None
        self.latest_fft_result = None
        self.latest_shifted_fft = None
        self.latest_magnitude_spectrum = None
        self.latest_power_spectrum = None
        self.latest_log_spectrum = None

        analysis_size = config.MODEL_FREE_ANALYSIS_IMAGE_SIZE
        self.radius_map = self._create_normalized_radius_map(
            analysis_size,
            analysis_size,
        )
        self.compatibility_context_builder = (
            ModelFreePreControlContextBuilder()
        )

    def analyze(self, context):
        """Global feature ve skoru ortak kare context'inden hesaplar."""
        self._set_latest_context(context)
        invalid_reason = self._context_invalid_reason(context)
        if invalid_reason is not None:
            self._register_invalid_frame()
            return self._unavailable_result(invalid_reason)

        features = self._extract_features(context.power_spectrum)
        if features is None:
            self._register_invalid_frame()
            return self._unavailable_result(
                "FFT analysis band contains insufficient energy"
            )

        scoring_mode = config.MODEL_FREE_GLOBAL_FFT_SCORING_MODE.lower()
        if scoring_mode != "experimental":
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                evidence=[
                    "Configured scoring mode requires calibration data"
                ],
                debug_data={
                    "quality_status": "Sufficient",
                    "scoring_mode": scoring_mode,
                    "possible_attack": "none",
                },
            )

        raw_score, feature_deviations = self._experimental_score(features)
        features["fft_global_score"] = raw_score
        return self._stabilize_result(
            raw_score,
            features,
            feature_deviations,
        )

    def analyze_face_box(self, frame, face_box):
        """Eski cagrilar icin ortak context olusturan uyumluluk girisi."""
        context = self.compatibility_context_builder.build(
            time.time(),
            frame,
            frame,
            face_box,
        )
        return self.analyze(context)

    def ft_pre_control(self, frame, detected_faces):
        """Mevcut tespit listesiyle geriye donuk uyumlu giris noktasi."""
        if len(detected_faces) != 1:
            self._clear_latest_context()
            self._register_invalid_frame()
            reason = "no face exists"
            if len(detected_faces) > 1:
                reason = "multiple faces detected"
            return self._unavailable_result(reason)
        return self.analyze_face_box(frame, detected_faces[0].box)

    def reset(self):
        self.score_history.clear()
        self.invalid_frame_streak = 0
        self._clear_latest_context()

    def create_frequency_band_overlay(self, log_magnitude_visualization):
        """Debug export icin FFT bant sinirlarini gorsellestirir."""
        if log_magnitude_visualization is None:
            return None
        if log_magnitude_visualization.ndim == 2:
            overlay = cv2.cvtColor(
                log_magnitude_visualization,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            overlay = log_magnitude_visualization.copy()

        height, width = overlay.shape[:2]
        center = (width // 2, height // 2)
        radius_scale = float(min(center))
        boundaries = (
            (
                config.EXPERIMENTAL_FFT_DC_EXCLUSION_RADIUS,
                (0, 255, 255),
                "DC excluded",
            ),
            (
                config.EXPERIMENTAL_FFT_LOW_OUTER_RADIUS,
                (0, 255, 0),
                "Low",
            ),
            (
                config.EXPERIMENTAL_FFT_MID_OUTER_RADIUS,
                (0, 165, 255),
                "Middle",
            ),
            (
                config.EXPERIMENTAL_FFT_HIGH_OUTER_RADIUS,
                (0, 0, 255),
                "High",
            ),
        )
        for normalized_radius, color, _label in boundaries:
            cv2.circle(
                overlay,
                center,
                max(1, round(normalized_radius * radius_scale)),
                color,
                1,
                cv2.LINE_AA,
            )

        for index, (_radius, color, label) in enumerate(boundaries):
            cv2.putText(
                overlay,
                label,
                (7, 16 + index * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        return overlay

    def get_frequency_band_configuration(self):
        return {
            "dc_exclusion_radius": (
                config.EXPERIMENTAL_FFT_DC_EXCLUSION_RADIUS
            ),
            "low": [
                config.EXPERIMENTAL_FFT_LOW_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_LOW_OUTER_RADIUS,
            ],
            "middle": [
                config.EXPERIMENTAL_FFT_MID_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_MID_OUTER_RADIUS,
            ],
            "high": [
                config.EXPERIMENTAL_FFT_HIGH_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_HIGH_OUTER_RADIUS,
            ],
        }

    def _set_latest_context(self, context):
        self.latest_context = context
        if not context.has_valid_fft:
            self._clear_latest_arrays()
            return
        self.latest_face_crop = context.original_high_resolution_face_crop
        self.latest_standardized_crop = context.standardized_analysis_crop
        self.latest_fft_result = context.fft_result
        self.latest_shifted_fft = context.shifted_fft_result
        self.latest_magnitude_spectrum = context.magnitude_spectrum
        self.latest_power_spectrum = context.power_spectrum
        self.latest_log_spectrum = context.log_power_spectrum

    def _clear_latest_context(self):
        self.latest_context = None
        self._clear_latest_arrays()

    def _clear_latest_arrays(self):
        self.latest_face_crop = None
        self.latest_standardized_crop = None
        self.latest_fft_result = None
        self.latest_shifted_fft = None
        self.latest_magnitude_spectrum = None
        self.latest_power_spectrum = None
        self.latest_log_spectrum = None

    def _context_invalid_reason(self, context):
        if not context.face_quality_valid:
            return context.quality_reason or "face quality gate failed"
        if context.aligned_face_crop is None:
            return "aligned face crop is unavailable"
        if context.aligned_face_crop.size == 0:
            return "aligned face crop is empty"
        if context.standardized_analysis_crop is None:
            return "standardized FFT input is unavailable"
        if context.standardized_analysis_crop.size == 0:
            return "standardized FFT input is empty"
        if not context.has_valid_fft:
            return "shared FFT is unavailable"

        arrays = (
            context.fft_result,
            context.shifted_fft_result,
            context.magnitude_spectrum,
            context.power_spectrum,
        )
        expected_shape = context.standardized_analysis_crop.shape
        for array in arrays:
            if array.shape != expected_shape:
                return "shared FFT shape is inconsistent"
            if not np.all(np.isfinite(array)):
                return "shared FFT spectrum contains invalid values"
        if np.any(context.power_spectrum < 0):
            return "shared FFT power spectrum contains negative values"
        return None

    def _extract_features(self, power_spectrum):
        masks = self._create_band_masks()
        analysis_values = power_spectrum[masks["analysis"]]
        total_energy = float(analysis_values.sum())
        if not math.isfinite(total_energy) or total_energy <= 1e-12:
            return None

        low_energy = float(power_spectrum[masks["low"]].sum())
        middle_energy = float(power_spectrum[masks["middle"]].sum())
        high_energy = float(power_spectrum[masks["high"]].sum())
        low_ratio = low_energy / total_energy
        middle_ratio = middle_energy / total_energy
        high_ratio = high_energy / total_energy

        selected_radii = self.radius_map[masks["analysis"]]
        spectral_centroid = float(
            np.sum(selected_radii * analysis_values) / total_energy
        )

        probabilities = analysis_values / total_energy
        positive_probabilities = probabilities[probabilities > 0]
        entropy = -float(
            np.sum(positive_probabilities * np.log(positive_probabilities))
        )
        maximum_entropy = math.log(max(2, analysis_values.size))
        normalized_entropy = entropy / maximum_entropy

        slope_fit = self._calculate_spectral_slope(power_spectrum)
        if slope_fit is None or not math.isfinite(slope_fit["slope"]):
            return None

        mean_power = float(np.mean(analysis_values))
        spectral_flatness = float(
            np.exp(np.mean(np.log(analysis_values + 1e-12)))
            / (mean_power + 1e-12)
        )
        centered_power = analysis_values - mean_power
        variance = float(np.mean(centered_power ** 2))
        spectral_kurtosis = float(
            np.mean(centered_power ** 4) / (variance ** 2 + 1e-12)
        )
        order = np.argsort(selected_radii)
        cumulative_energy = np.cumsum(analysis_values[order])
        rolloff_index = int(
            np.searchsorted(
                cumulative_energy,
                0.85 * total_energy,
                side="left",
            )
        )
        rolloff_index = min(rolloff_index, order.size - 1)
        spectral_rolloff_85 = float(selected_radii[order[rolloff_index]])
        peak_metrics = self._dominant_symmetric_peak_metrics(
            power_spectrum,
            masks["analysis"],
            total_energy,
        )

        return {
            "low_frequency_energy_ratio": float(low_ratio),
            "middle_frequency_energy_ratio": float(middle_ratio),
            "high_frequency_energy_ratio": float(high_ratio),
            "spectral_centroid": spectral_centroid,
            "spectral_entropy": float(normalized_entropy),
            "spectral_slope": slope_fit["slope"],
            "spectral_slope_fit_mad": slope_fit["fit_mad"],
            "spectral_flatness": spectral_flatness,
            "spectral_rolloff_85_radius": spectral_rolloff_85,
            "spectral_kurtosis_pearson": spectral_kurtosis,
            **peak_metrics,
            "total_spectral_energy": total_energy,
            "high_to_low_energy_ratio": float(
                high_energy / (low_energy + 1e-12)
            ),
        }

    def _calculate_spectral_slope(self, power_spectrum):
        inner_radius = config.EXPERIMENTAL_FFT_DC_EXCLUSION_RADIUS
        outer_radius = config.EXPERIMENTAL_FFT_ANALYSIS_OUTER_RADIUS
        bin_edges = np.linspace(
            inner_radius,
            outer_radius,
            config.EXPERIMENTAL_FFT_RADIAL_SLOPE_BIN_COUNT + 1,
        )
        radial_centers = []
        radial_means = []
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            bin_mask = (self.radius_map >= lower) & (self.radius_map < upper)
            if not np.any(bin_mask):
                continue
            mean_power = float(np.mean(power_spectrum[bin_mask]))
            if not math.isfinite(mean_power) or mean_power <= 0:
                continue
            radial_centers.append((lower + upper) / 2.0)
            radial_means.append(mean_power)

        if len(radial_centers) < (
            config.EXPERIMENTAL_FFT_MINIMUM_SLOPE_BIN_COUNT
        ):
            return None

        log_radius = np.log(np.asarray(radial_centers, dtype=np.float64))
        log_power = np.log(np.asarray(radial_means, dtype=np.float64))
        delta_x = log_radius[None, :] - log_radius[:, None]
        delta_y = log_power[None, :] - log_power[:, None]
        upper_triangle = np.triu(
            np.ones(delta_x.shape, dtype=bool),
            k=1,
        )
        valid = upper_triangle & (np.abs(delta_x) > 1e-12)
        if not np.any(valid):
            return None
        slope = float(np.median(delta_y[valid] / delta_x[valid]))
        intercept = float(np.median(log_power - slope * log_radius))
        residuals = log_power - (slope * log_radius + intercept)
        fit_mad = float(
            np.median(np.abs(residuals - np.median(residuals)))
        )
        return {"slope": slope, "fit_mad": fit_mad}

    def _dominant_symmetric_peak_metrics(
        self,
        power_spectrum,
        analysis_mask,
        total_energy,
    ):
        candidates = np.where(analysis_mask, power_spectrum, -np.inf)
        flat_index = int(np.argmax(candidates))
        peak_y, peak_x = np.unravel_index(
            flat_index,
            power_spectrum.shape,
        )
        peak_energy = float(power_spectrum[peak_y, peak_x])
        height, width = power_spectrum.shape
        symmetric_y = (2 * (height // 2) - peak_y) % height
        symmetric_x = (2 * (width // 2) - peak_x) % width
        symmetric_energy = float(
            power_spectrum[symmetric_y, symmetric_x]
        )
        amplitude_symmetry = min(peak_energy, symmetric_energy) / (
            max(peak_energy, symmetric_energy) + 1e-12
        )
        return {
            "dominant_non_dc_peak_ratio": peak_energy / total_energy,
            "dominant_non_dc_peak_coordinate": [int(peak_x), int(peak_y)],
            "dominant_symmetric_pair_energy_ratio": (
                peak_energy + symmetric_energy
            )
            / total_energy,
            "dominant_pair_amplitude_symmetry": float(amplitude_symmetry),
        }

    def _experimental_score(self, features):
        weighted_deviation = 0.0
        total_weight = 0.0
        feature_deviations = {}
        for feature_name, profile in (
            config.EXPERIMENTAL_FFT_FEATURE_PROFILES.items()
        ):
            value = features[feature_name]
            deviation = self._range_deviation(
                value,
                profile["minimum"],
                profile["maximum"],
                profile["deviation_scale"],
            )
            weight = float(profile["weight"])
            feature_deviations[feature_name] = deviation
            weighted_deviation += deviation * weight
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Global FFT feature weights must be positive")
        score = 100.0 * weighted_deviation / total_weight
        return float(np.clip(score, 0.0, 100.0)), feature_deviations

    def _stabilize_result(self, raw_score, features, feature_deviations):
        self.invalid_frame_streak = 0
        self.score_history.append(raw_score)
        stabilized_score = float(np.median(self.score_history))
        history_length = len(self.score_history)
        history_ready = (
            history_length >= config.EXPERIMENTAL_FFT_MINIMUM_VALID_FRAMES
        )

        warnings = []
        if not history_ready:
            status = "Analysis uncertain"
        elif stabilized_score >= config.EXPERIMENTAL_FFT_HIGH_ANOMALY_SCORE:
            status = "High spectral anomaly"
            warnings.append(
                "Global frequency distribution has a high experimental anomaly score"
            )
        elif stabilized_score >= config.EXPERIMENTAL_FFT_SUSPICIOUS_SCORE:
            status = "Suspicious frequency distribution"
            warnings.append(
                "Global frequency distribution is outside experimental ranges"
            )
        else:
            status = "Normal frequency structure"

        evidence = self._create_evidence(feature_deviations)
        evidence.append(
            "Experimental feature ranges are not a calibrated scientific baseline"
        )
        temporal_confidence = min(
            1.0,
            history_length / config.EXPERIMENTAL_FFT_MINIMUM_VALID_FRAMES,
        )
        confidence = (
            temporal_confidence
            * config.EXPERIMENTAL_FFT_MAXIMUM_CONFIDENCE
        )
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=features,
            raw_score=raw_score,
            stabilized_score=stabilized_score,
            confidence=confidence,
            status=status,
            evidence=evidence,
            warnings=warnings,
            debug_data={
                "possible_attack": "none",
                "quality_status": "Sufficient",
                "history_length": history_length,
                "scoring_mode": "experimental",
                "feature_deviations": feature_deviations,
                "frequency_bands": self.get_frequency_band_configuration(),
            },
            calibrated=False,
        )

    def _create_evidence(self, feature_deviations):
        labels = {
            "low_frequency_energy_ratio": "Low-frequency energy deviates",
            "middle_frequency_energy_ratio": "Middle-frequency energy deviates",
            "high_frequency_energy_ratio": "High-frequency energy deviates",
            "spectral_centroid": "Spectral centroid deviates",
            "spectral_entropy": "Spectral entropy deviates",
            "spectral_slope": "Radial spectral slope deviates",
            "high_to_low_energy_ratio": "High-to-low energy ratio deviates",
        }
        evidence = [
            labels[name]
            for name, deviation in feature_deviations.items()
            if deviation >= config.EXPERIMENTAL_FFT_EVIDENCE_DEVIATION
        ]
        if not evidence:
            evidence.append(
                "No strong deviation from provisional global FFT ranges"
            )
        return evidence

    def _create_band_masks(self):
        return {
            "analysis": (
                (
                    self.radius_map
                    >= config.EXPERIMENTAL_FFT_DC_EXCLUSION_RADIUS
                )
                & (
                    self.radius_map
                    <= config.EXPERIMENTAL_FFT_ANALYSIS_OUTER_RADIUS
                )
            ),
            "low": (
                (
                    self.radius_map
                    >= config.EXPERIMENTAL_FFT_LOW_INNER_RADIUS
                )
                & (
                    self.radius_map
                    < config.EXPERIMENTAL_FFT_LOW_OUTER_RADIUS
                )
            ),
            "middle": (
                (
                    self.radius_map
                    >= config.EXPERIMENTAL_FFT_MID_INNER_RADIUS
                )
                & (
                    self.radius_map
                    < config.EXPERIMENTAL_FFT_MID_OUTER_RADIUS
                )
            ),
            "high": (
                (
                    self.radius_map
                    >= config.EXPERIMENTAL_FFT_HIGH_INNER_RADIUS
                )
                & (
                    self.radius_map
                    <= config.EXPERIMENTAL_FFT_HIGH_OUTER_RADIUS
                )
            ),
        }

    def _create_normalized_radius_map(self, width, height):
        center_x = width // 2
        center_y = height // 2
        y_coordinates, x_coordinates = np.indices((height, width))
        radius_scale = float(max(1, min(center_x, center_y)))
        return np.hypot(
            x_coordinates - center_x,
            y_coordinates - center_y,
        ) / radius_scale

    def _register_invalid_frame(self):
        self.invalid_frame_streak += 1
        if (
            self.invalid_frame_streak
            >= config.EXPERIMENTAL_FFT_INVALID_RESET_FRAMES
        ):
            self.score_history.clear()
            self.invalid_frame_streak = 0

    def _unavailable_result(self, reason):
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=False,
            raw_features={},
            raw_score=None,
            stabilized_score=None,
            confidence=0.0,
            status="Analysis unavailable",
            evidence=[reason],
            warnings=[],
            debug_data={
                "possible_attack": "none",
                "quality_status": reason,
                "scoring_mode": (
                    config.MODEL_FREE_GLOBAL_FFT_SCORING_MODE
                ),
            },
            calibrated=False,
        )

    def _range_deviation(self, value, minimum, maximum, scale):
        if not math.isfinite(value):
            return 1.0
        if scale <= 0:
            raise ValueError("Global FFT deviation scale must be positive")
        if value < minimum:
            distance = minimum - value
        elif value > maximum:
            distance = value - maximum
        else:
            return 0.0
        return float(np.clip(distance / scale, 0.0, 1.0))
