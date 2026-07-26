"""Module 3: Ortak power spectrum icin radial ve angular analiz."""

from collections import deque
import json
import math

import cv2
import numpy as np

import config
from model_free_analysis import ModelFreeAnalysisResult


class RadialAngularSpectrumPreController:
    """Merkezlenmis ortak FFT power spectrum'unu 1B profillere donusturur.

    Bu modul FFT, fftshift veya resize hesaplamaz. Sac, sakal, kas, pose,
    aydinlatma gradyani, desenli arka plan ve crop sinirlari yon/radial profili
    etkileyebilir. Ortak Hann penceresi crop kenarlarini bastirir; fakat sonuc
    tek basina real/fake veya saldiri siniflandirmasi degildir.
    """

    MODULE_NAME = "Radial/Angular"

    def __init__(self):
        history_size = config.EXPERIMENTAL_RADIAL_ANGULAR_HISTORY_SIZE
        self.score_history = deque(maxlen=history_size)
        self.radial_score_history = deque(maxlen=history_size)
        self.angular_score_history = deque(maxlen=history_size)
        self.invalid_streak = 0
        self.previous_region = None

        analysis_size = config.MODEL_FREE_ANALYSIS_IMAGE_SIZE
        self.radius_map, self.angle_map = self._create_coordinate_maps(
            analysis_size,
            analysis_size,
        )
        self.calibration = self._load_calibration()

    def analyze(self, context):
        self._handle_region_change(context.face_bounding_box)
        invalid_reason = self._context_invalid_reason(context)
        if invalid_reason is not None:
            return self._register_unavailable(invalid_reason)

        radial_data = self._calculate_radial_profile(context.power_spectrum)
        angular_data = self._calculate_angular_profile(context.power_spectrum)
        if radial_data is None or angular_data is None:
            return self._register_unavailable(
                "radial or angular spectrum contains insufficient energy"
            )

        features = {}
        features.update(radial_data)
        features.update(angular_data)

        radial_score, radial_deviations = self._score_feature_profiles(
            features,
            config.EXPERIMENTAL_RADIAL_FEATURE_PROFILES,
        )
        angular_score, angular_deviations = self._score_feature_profiles(
            features,
            config.EXPERIMENTAL_ANGULAR_FEATURE_PROFILES,
        )

        calibrated = self.calibration is not None
        radial_profile_deviation = None
        angular_profile_deviation = None
        scoring_mode = config.RADIAL_ANGULAR_SCORING_MODE.lower()
        if calibrated and scoring_mode in ("auto", "calibrated"):
            radial_profile_deviation = self._profile_deviation(
                features["radial_normalized_energy_profile"],
                self.calibration["radial_profile"],
            )
            angular_profile_deviation = self._profile_deviation(
                features["angular_energy_profile"],
                self.calibration["angular_profile"],
            )
            radial_score = self._blend_calibrated_score(
                radial_score,
                radial_profile_deviation,
            )
            angular_score = self._blend_calibrated_score(
                angular_score,
                angular_profile_deviation,
            )
        elif scoring_mode == "calibrated":
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                evidence=[
                    "Compatible radial/angular calibration profiles are unavailable"
                ],
                debug_data={
                    "quality_status": "Sufficient",
                    "scoring_mode": scoring_mode,
                    "possible_attack": "none",
                },
            )
        else:
            calibrated = False

        combined_score = self._combine_scores(
            radial_score,
            angular_score,
        )
        features.update(
            {
                "radial_profile_deviation": radial_profile_deviation,
                "angular_profile_deviation": angular_profile_deviation,
                "radial_anomaly_score": radial_score,
                "angular_anisotropy_score": angular_score,
                "radial_angular_score": combined_score,
            }
        )
        return self._stabilize_result(
            context,
            features,
            radial_deviations,
            angular_deviations,
            radial_score,
            angular_score,
            combined_score,
            calibrated,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None

    def create_profile_image(self, values, title, color):
        """Debug export icin dependency gerektirmeyen basit profil grafigi."""
        width = 640
        height = 320
        margin_left = 52
        margin_right = 18
        margin_top = 42
        margin_bottom = 38
        image = np.full((height, width, 3), 20, dtype=np.uint8)
        cv2.putText(
            image,
            title,
            (margin_left, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        origin = (margin_left, height - margin_bottom)
        cv2.line(
            image,
            origin,
            (width - margin_right, origin[1]),
            (125, 125, 125),
            1,
        )
        cv2.line(
            image,
            origin,
            (origin[0], margin_top),
            (125, 125, 125),
            1,
        )

        profile = np.asarray(values, dtype=np.float64)
        if profile.size < 2 or not np.all(np.isfinite(profile)):
            return image
        maximum = float(profile.max())
        if maximum <= 1e-12:
            return image

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        points = []
        for index, value in enumerate(profile):
            x_position = margin_left + round(
                index * plot_width / max(1, profile.size - 1)
            )
            y_position = origin[1] - round(value / maximum * plot_height)
            points.append((x_position, y_position))
        cv2.polylines(
            image,
            [np.asarray(points, dtype=np.int32)],
            False,
            color,
            2,
            cv2.LINE_AA,
        )
        return image

    def create_direction_overlay(
        self,
        log_magnitude_visualization,
        dominant_frequency_angle_degrees,
    ):
        """Dominant frequency ve ona dik image-line yonunu isaretler."""
        if log_magnitude_visualization.ndim == 2:
            overlay = cv2.cvtColor(
                log_magnitude_visualization,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            overlay = log_magnitude_visualization.copy()
        height, width = overlay.shape[:2]
        center = (width // 2, height // 2)
        line_length = round(
            min(center)
            * config.EXPERIMENTAL_RADIAL_ANGULAR_OUTER_RADIUS
        )
        self._draw_angle_line(
            overlay,
            center,
            line_length,
            dominant_frequency_angle_degrees,
            (0, 255, 255),
        )
        image_line_angle = (dominant_frequency_angle_degrees + 90.0) % 180.0
        self._draw_angle_line(
            overlay,
            center,
            line_length,
            image_line_angle,
            (0, 255, 0),
        )
        cv2.putText(
            overlay,
            "Frequency direction: %.1f deg" % dominant_frequency_angle_degrees,
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Image-line direction: %.1f deg" % image_line_angle,
            (7, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def _calculate_radial_profile(self, power_spectrum):
        inner = config.EXPERIMENTAL_RADIAL_ANGULAR_INNER_RADIUS
        outer = config.EXPERIMENTAL_RADIAL_ANGULAR_OUTER_RADIUS
        edges = np.linspace(
            inner,
            outer,
            config.EXPERIMENTAL_RADIAL_BIN_COUNT + 1,
        )
        centers = []
        mean_power = []
        median_power = []
        bin_energy = []
        log_power = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (self.radius_map >= lower) & (self.radius_map < upper)
            values = power_spectrum[mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                centers.append(float((lower + upper) / 2.0))
                mean_power.append(0.0)
                median_power.append(0.0)
                bin_energy.append(0.0)
                log_power.append(0.0)
                continue
            current_mean = float(np.mean(values))
            centers.append(float((lower + upper) / 2.0))
            mean_power.append(current_mean)
            median_power.append(float(np.median(values)))
            bin_energy.append(float(np.sum(values)))
            log_power.append(float(np.log1p(current_mean)))

        total_energy = float(sum(bin_energy))
        if not math.isfinite(total_energy) or total_energy <= 1e-12:
            return None
        normalized_energy = [
            float(energy / total_energy) for energy in bin_energy
        ]
        slope, fit_error = self._fit_radial_slope(
            centers,
            mean_power,
        )
        if slope is None or fit_error is None:
            return None

        profile = np.asarray(normalized_energy, dtype=np.float64)
        positive_profile = profile[profile > 0]
        entropy = -float(
            np.sum(positive_profile * np.log(positive_profile))
        ) / math.log(max(2, profile.size))
        dominant_index = int(np.argmax(profile))
        dominant_ratio = float(profile[dominant_index])
        narrow_concentration = self._narrow_band_concentration(
            profile,
            dominant_index,
        )
        band_energies = self._radial_band_energies(
            power_spectrum,
            total_energy,
        )
        return {
            "radial_bin_centers": centers,
            "radial_mean_power_profile": mean_power,
            "radial_median_power_profile": median_power,
            "radial_normalized_energy_profile": normalized_energy,
            "radial_log_power_profile": log_power,
            "radial_spectral_slope": slope,
            "slope_fit_error": fit_error,
            "low_radial_energy": band_energies["low"],
            "middle_radial_energy": band_energies["middle"],
            "high_radial_energy": band_energies["high"],
            "radial_entropy": entropy,
            "dominant_radial_frequency": centers[dominant_index],
            "dominant_radial_energy_ratio": dominant_ratio,
            "narrow_band_energy_concentration": narrow_concentration,
        }

    def _calculate_angular_profile(self, power_spectrum):
        valid_mask = self._valid_annulus_mask()
        valid_energy = power_spectrum[valid_mask]
        total_energy = float(valid_energy.sum())
        if not math.isfinite(total_energy) or total_energy <= 1e-12:
            return None

        bin_count = config.EXPERIMENTAL_ANGULAR_BIN_COUNT
        edges = np.linspace(0.0, math.pi, bin_count + 1)
        centers_radians = (edges[:-1] + edges[1:]) / 2.0
        bin_energy = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = valid_mask & (self.angle_map >= lower) & (
                self.angle_map < upper
            )
            bin_energy.append(float(power_spectrum[mask].sum()))
        profile = np.asarray(bin_energy, dtype=np.float64) / total_energy
        if not np.all(np.isfinite(profile)):
            return None

        centers_degrees = np.degrees(centers_radians)
        dominant_index = int(np.argmax(profile))
        dominant_frequency_angle = float(centers_degrees[dominant_index])
        circular_cosine = float(
            np.sum(profile * np.cos(2.0 * centers_radians))
        )
        circular_sine = float(
            np.sum(profile * np.sin(2.0 * centers_radians))
        )
        resultant_strength = float(
            np.clip(
                math.hypot(circular_cosine, circular_sine),
                0.0,
                1.0,
            )
        )
        mean_angle = (
            0.5 * math.atan2(circular_sine, circular_cosine)
        ) % math.pi
        positive_profile = profile[profile > 0]
        entropy = -float(
            np.sum(positive_profile * np.log(positive_profile))
        ) / math.log(max(2, profile.size))
        uniform_profile = np.full(profile.shape, 1.0 / profile.size)
        maximum_total_variation = 1.0 - 1.0 / profile.size
        anisotropy = (
            0.5 * float(np.sum(np.abs(profile - uniform_profile)))
            / maximum_total_variation
        )

        horizontal = self._sector_energy(centers_degrees, profile, (0.0,))
        vertical = self._sector_energy(centers_degrees, profile, (90.0,))
        diagonal = self._sector_energy(
            centers_degrees,
            profile,
            (45.0, 135.0),
        )
        return {
            "angular_bin_centers_degrees": centers_degrees.tolist(),
            "angular_energy_profile": profile.tolist(),
            "dominant_frequency_angle_degrees": dominant_frequency_angle,
            "dominant_image_line_angle_degrees": float(
                (dominant_frequency_angle + 90.0) % 180.0
            ),
            "maximum_angular_energy": float(profile[dominant_index]),
            "angular_mean_degrees": float(math.degrees(mean_angle)),
            "angular_variance": float(1.0 - resultant_strength),
            "angular_entropy": entropy,
            "directional_anisotropy": float(
                np.clip(anisotropy, 0.0, 1.0)
            ),
            "horizontal_concentration": horizontal,
            "vertical_concentration": vertical,
            "diagonal_concentration": diagonal,
        }

    def _fit_radial_slope(self, centers, mean_power):
        centers = np.asarray(centers, dtype=np.float64)
        mean_power = np.asarray(mean_power, dtype=np.float64)
        valid = (
            np.isfinite(centers)
            & np.isfinite(mean_power)
            & (centers > 0)
            & (mean_power > 0)
        )
        if int(np.sum(valid)) < (
            config.EXPERIMENTAL_RADIAL_MINIMUM_FIT_BIN_COUNT
        ):
            return None, None
        x_values = np.log(centers[valid])
        y_values = np.log(mean_power[valid])
        centered_x = x_values - float(x_values.mean())
        denominator = float(np.sum(centered_x ** 2))
        if denominator <= 1e-12:
            return None, None
        slope = float(
            np.sum(centered_x * (y_values - float(y_values.mean())))
            / denominator
        )
        intercept = float(y_values.mean() - slope * x_values.mean())
        residuals = y_values - (slope * x_values + intercept)
        rmse = math.sqrt(float(np.mean(residuals ** 2)))
        fit_scale = max(float(np.std(y_values)), 1e-12)
        return slope, float(rmse / fit_scale)

    def _narrow_band_concentration(self, profile, dominant_index):
        neighbors = config.EXPERIMENTAL_RADIAL_NARROW_BAND_NEIGHBOR_COUNT
        left = max(0, dominant_index - neighbors)
        right = min(profile.size, dominant_index + neighbors + 1)
        window_energy = float(profile[left:right].sum())
        expected_uniform_energy = (right - left) / profile.size
        return float(window_energy / max(expected_uniform_energy, 1e-12))

    def _radial_band_energies(self, power_spectrum, total_energy):
        band_limits = {
            "low": (
                config.EXPERIMENTAL_FFT_LOW_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_LOW_OUTER_RADIUS,
            ),
            "middle": (
                config.EXPERIMENTAL_FFT_MID_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_MID_OUTER_RADIUS,
            ),
            "high": (
                config.EXPERIMENTAL_FFT_HIGH_INNER_RADIUS,
                config.EXPERIMENTAL_FFT_HIGH_OUTER_RADIUS,
            ),
        }
        ratios = {}
        for name, (inner, outer) in band_limits.items():
            mask = (self.radius_map >= inner) & (self.radius_map < outer)
            ratios[name] = float(power_spectrum[mask].sum() / total_energy)
        return ratios

    def _sector_energy(self, centers_degrees, profile, targets):
        half_width = (
            config.EXPERIMENTAL_ANGULAR_SECTOR_HALF_WIDTH_DEGREES
        )
        selected = np.zeros(profile.shape, dtype=bool)
        for target in targets:
            distance = np.abs(
                ((centers_degrees - target + 90.0) % 180.0) - 90.0
            )
            selected |= distance <= half_width
        return float(profile[selected].sum())

    def _score_feature_profiles(self, features, profiles):
        weighted_score = 0.0
        total_weight = 0.0
        deviations = {}
        for feature_name, profile in profiles.items():
            deviation = self._range_deviation(
                features[feature_name],
                profile["minimum"],
                profile["maximum"],
                profile["deviation_scale"],
            )
            weight = float(profile["weight"])
            deviations[feature_name] = deviation
            weighted_score += deviation * weight
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("Radial/angular feature weights must be positive")
        return 100.0 * weighted_score / total_weight, deviations

    def _blend_calibrated_score(self, experimental_score, deviation):
        calibration_score = 100.0 * float(
            np.clip(
                deviation
                / config.EXPERIMENTAL_CALIBRATED_PROFILE_DEVIATION_FULL,
                0.0,
                1.0,
            )
        )
        weight = config.EXPERIMENTAL_CALIBRATED_PROFILE_WEIGHT
        return float(
            (1.0 - weight) * experimental_score
            + weight * calibration_score
        )

    def _combine_scores(self, radial_score, angular_score):
        radial_weight = config.EXPERIMENTAL_RADIAL_SCORE_WEIGHT
        angular_weight = config.EXPERIMENTAL_ANGULAR_SCORE_WEIGHT
        total_weight = radial_weight + angular_weight
        if total_weight <= 0:
            raise ValueError("Radial/angular score weights must be positive")
        score = (
            radial_weight * radial_score + angular_weight * angular_score
        ) / total_weight
        return float(np.clip(score, 0.0, 100.0))

    def _stabilize_result(
        self,
        context,
        features,
        radial_deviations,
        angular_deviations,
        radial_score,
        angular_score,
        combined_score,
        calibrated,
    ):
        self.invalid_streak = 0
        self.radial_score_history.append(radial_score)
        self.angular_score_history.append(angular_score)
        self.score_history.append(combined_score)
        stable_radial = float(np.median(self.radial_score_history))
        stable_angular = float(np.median(self.angular_score_history))
        stable_combined = float(np.median(self.score_history))
        history_length = len(self.score_history)
        history_ready = (
            history_length
            >= config.EXPERIMENTAL_RADIAL_ANGULAR_MINIMUM_HISTORY
        )

        warnings = []
        if not history_ready:
            status = "Analysis uncertain"
        elif (
            stable_angular
            >= config.EXPERIMENTAL_RADIAL_DIRECTIONAL_STATUS_SCORE
            and stable_angular >= stable_radial
        ):
            status = "Directional concentration detected"
            warnings.append(
                "Directional frequency concentration detected; image-space lines are perpendicular"
            )
        elif (
            stable_radial
            >= config.EXPERIMENTAL_RADIAL_NARROW_BAND_STATUS_SCORE
            and radial_deviations.get(
                "narrow_band_energy_concentration",
                0.0,
            )
            >= config.EXPERIMENTAL_RADIAL_ANGULAR_EVIDENCE_DEVIATION
        ):
            status = "Narrow-band spectral anomaly"
            warnings.append("Narrow radial frequency concentration detected")
        elif stable_combined >= (
            config.EXPERIMENTAL_RADIAL_ANGULAR_SUSPICIOUS_SCORE
        ):
            status = "Suspicious radial/angular structure"
            warnings.append("Radial/angular spectrum deviates from references")
        else:
            status = "Normal spectral distribution"

        evidence = self._create_evidence(
            context,
            radial_deviations,
            angular_deviations,
            calibrated,
        )
        history_confidence = min(
            1.0,
            history_length
            / config.EXPERIMENTAL_RADIAL_ANGULAR_MINIMUM_HISTORY,
        )
        confidence_limit = (
            1.0
            if calibrated
            else config.EXPERIMENTAL_RADIAL_ANGULAR_MAXIMUM_CONFIDENCE
        )
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=features,
            raw_score=combined_score,
            stabilized_score=stable_combined,
            confidence=history_confidence * confidence_limit,
            status=status,
            evidence=evidence,
            warnings=warnings,
            debug_data={
                "possible_attack": "none",
                "quality_status": "Sufficient",
                "scoring_mode": (
                    "calibrated" if calibrated else "experimental"
                ),
                "history_length": history_length,
                "raw_radial_score": radial_score,
                "raw_angular_score": angular_score,
                "stabilized_radial_score": stable_radial,
                "stabilized_angular_score": stable_angular,
                "radial_feature_deviations": radial_deviations,
                "angular_feature_deviations": angular_deviations,
                "false_positive_factors": [
                    "face pose",
                    "hair",
                    "beard",
                    "eyebrows",
                    "image blur",
                    "lighting gradients",
                    "crop borders",
                    "patterned background",
                ],
            },
            calibrated=calibrated,
        )

    def _create_evidence(
        self,
        context,
        radial_deviations,
        angular_deviations,
        calibrated,
    ):
        threshold = config.EXPERIMENTAL_RADIAL_ANGULAR_EVIDENCE_DEVIATION
        evidence = []
        for name, deviation in radial_deviations.items():
            if deviation >= threshold:
                evidence.append("Radial feature deviates: " + name)
        for name, deviation in angular_deviations.items():
            if deviation >= threshold:
                evidence.append("Angular feature deviates: " + name)
        if not evidence:
            evidence.append("No strong radial/angular deviation detected")
        if not calibrated:
            evidence.append(
                "Experimental ranges; bona-fide profile calibration unavailable"
            )
        if context.pose_alignment_valid is None:
            evidence.append(
                "Pose validity unavailable; directional evidence is conservative"
            )
        if not context.alignment_applied:
            evidence.append(
                "No geometric face alignment; Hann window reduces crop-border influence"
            )
        return evidence

    def _context_invalid_reason(self, context):
        if not context.face_quality_valid:
            return context.quality_reason or "face quality gate failed"
        if context.aligned_face_crop is None or context.aligned_face_crop.size == 0:
            return "aligned face crop is unavailable"
        if not context.has_valid_fft or context.power_spectrum is None:
            return "shared centered power spectrum is unavailable"
        if context.power_spectrum.ndim != 2:
            return "shared power spectrum must be two-dimensional"
        if context.power_spectrum.shape != self.radius_map.shape:
            return "shared power spectrum shape is inconsistent"
        if not np.all(np.isfinite(context.power_spectrum)):
            return "shared power spectrum contains invalid values"
        if np.any(context.power_spectrum < 0):
            return "shared power spectrum contains negative values"
        return None

    def _valid_annulus_mask(self):
        return (
            self.radius_map
            >= config.EXPERIMENTAL_RADIAL_ANGULAR_INNER_RADIUS
        ) & (
            self.radius_map
            <= config.EXPERIMENTAL_RADIAL_ANGULAR_OUTER_RADIUS
        )

    def _create_coordinate_maps(self, width, height):
        center_x = width // 2
        center_y = height // 2
        y_coordinates, x_coordinates = np.indices((height, width))
        delta_x = x_coordinates - center_x
        delta_y = y_coordinates - center_y
        radius_scale = float(max(1, min(center_x, center_y)))
        radius_map = np.hypot(delta_x, delta_y) / radius_scale
        angle_map = np.mod(np.arctan2(delta_y, delta_x), math.pi)
        return radius_map, angle_map

    def _draw_angle_line(self, image, center, length, angle_degrees, color):
        angle_radians = math.radians(angle_degrees)
        offset_x = round(math.cos(angle_radians) * length)
        offset_y = round(math.sin(angle_radians) * length)
        cv2.line(
            image,
            (center[0] - offset_x, center[1] - offset_y),
            (center[0] + offset_x, center[1] + offset_y),
            color,
            1,
            cv2.LINE_AA,
        )

    def _range_deviation(self, value, minimum, maximum, scale):
        if not math.isfinite(value):
            return 1.0
        if scale <= 0:
            raise ValueError("Radial/angular deviation scale must be positive")
        if value < minimum:
            distance = minimum - value
        elif value > maximum:
            distance = value - maximum
        else:
            return 0.0
        return float(np.clip(distance / scale, 0.0, 1.0))

    def _profile_deviation(self, current_profile, reference_profile):
        current = np.asarray(current_profile, dtype=np.float64)
        reference = np.asarray(reference_profile, dtype=np.float64)
        return 0.5 * float(np.sum(np.abs(current - reference)))

    def _load_calibration(self):
        path = config.MODEL_FREE_CALIBRATION_FILE_PATH
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as calibration_file:
                document = json.load(calibration_file)
            section = document.get("radial_angular", {})
            radial = self._validated_reference_profile(
                section.get("radial_profile"),
                config.EXPERIMENTAL_RADIAL_BIN_COUNT,
            )
            angular = self._validated_reference_profile(
                section.get("angular_profile"),
                config.EXPERIMENTAL_ANGULAR_BIN_COUNT,
            )
            if radial is None or angular is None:
                print(
                    "Radial/angular calibration profiles are missing or incompatible."
                )
                return None
            return {
                "radial_profile": radial,
                "angular_profile": angular,
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print("Radial/angular calibration could not be loaded: " + str(error))
            return None

    def _validated_reference_profile(self, values, expected_size):
        if not isinstance(values, list) or len(values) != expected_size:
            return None
        profile = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(profile)) or np.any(profile < 0):
            return None
        total = float(profile.sum())
        if total <= 1e-12:
            return None
        return (profile / total).tolist()

    def _handle_region_change(self, face_box):
        current_region = self._region_tuple(face_box)
        if current_region is None:
            return
        if self.previous_region is not None:
            overlap = self._intersection_over_union(
                self.previous_region,
                current_region,
            )
            if overlap < (
                config.EXPERIMENTAL_RADIAL_ANGULAR_REGION_IOU_RESET_THRESHOLD
            ):
                self._reset_temporal_state()
        self.previous_region = current_region

    def _register_unavailable(self, reason):
        self.invalid_streak += 1
        if self.invalid_streak >= (
            config.EXPERIMENTAL_RADIAL_ANGULAR_INVALID_RESET_FRAMES
        ):
            self._reset_temporal_state()
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
                    "calibrated" if self.calibration else "experimental"
                ),
            },
            calibrated=self.calibration is not None,
        )

    def _reset_temporal_state(self):
        self.score_history.clear()
        self.radial_score_history.clear()
        self.angular_score_history.clear()
        self.invalid_streak = 0

    def _region_tuple(self, face_box):
        if face_box is None:
            return None
        return (
            int(face_box.x),
            int(face_box.y),
            int(face_box.width),
            int(face_box.height),
        )

    def _intersection_over_union(self, first, second):
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = (
            first_width * first_height
            + second_width * second_height
            - intersection
        )
        if union <= 0:
            return 0.0
        return intersection / union
