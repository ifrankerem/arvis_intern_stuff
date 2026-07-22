"""Mevcut FFT spektrumundan Moire/periyodik desen suphe sinyali uretir.

Bu modul FFT hesaplamaz. ``GlobalFFTPreController`` tarafindan uretilmis,
merkezi zaten ``fftshift`` ile kaydirilmis sayisal guc ve log spektrumlarini
kullanir. Sonuc kesin bir canlilik veya fake karari degildir.
"""

from collections import deque
import math

import cv2
import numpy as np

import config


class MoireAnalysisResult:
    """UI ve PreControl koleksiyonu icin aciklanabilir Moire sonucu."""

    display_name = "Moiré"

    def __init__(
        self,
        available,
        score,
        periodic_peak_score,
        symmetric_peak_score,
        directional_concentration_score,
        status,
        possible_attack,
        evidence,
        warning,
        quality_status,
        metrics=None,
    ):
        self.available = available
        self.moire_available = available
        self.score = score
        self.moire_score = score
        self.periodic_peak_score = periodic_peak_score
        self.symmetric_peak_score = symmetric_peak_score
        self.directional_concentration_score = (
            directional_concentration_score
        )
        self.status = status
        self.moire_status = status
        self.possible_attack = possible_attack
        self.attack_type = possible_attack
        self.evidence = evidence
        self.warning = warning
        self.quality_status = quality_status
        self.passed = available and status == "Normal"
        self.metrics = metrics or {}


class MoirePeriodicPatternPreController:
    """FFT peak, merkez simetrisi ve yon yogunlasmasini inceler.

    Sac/sakal, cizgili kiyafet, desenli arka plan, panjur, JPEG sikistirmasi,
    kamera keskinlestirmesi ve dijital yeniden boyutlandirma da periyodik peak
    uretebilir. Bu nedenle sinyal konservatiftir, tek kareyle warning vermez ve
    gerceklik/fake siniflandirmasi olarak kullanilmamalidir. Mevcut FFT girisi
    Hann pencereli ROI'den geldigi icin crop kenarlari zaten bastirilmistir.
    """

    def __init__(self):
        self.score_history = deque(
            maxlen=config.EXPERIMENTAL_MOIRE_HISTORY_SIZE
        )
        self.suspicious_streak = 0
        self.normal_streak = 0
        self.invalid_streak = 0
        self.warning_is_active = False
        self.previous_region = None

    def analyze(
        self,
        power_spectrum,
        log_spectrum,
        face_box=None,
        unavailable_reason="FFT spectrum unavailable",
    ):
        self._handle_region_change(face_box)

        if power_spectrum is None or log_spectrum is None:
            return self.register_unavailable(unavailable_reason, face_box)

        if not self._spectrum_is_valid(power_spectrum, log_spectrum):
            return self.register_unavailable(
                "FFT spectrum contains invalid values",
                face_box,
            )

        metrics = self._calculate_metrics(power_spectrum, log_spectrum)
        if metrics is None:
            return self.register_unavailable(
                "selected FFT band contains insufficient energy",
                face_box,
            )

        self.invalid_streak = 0
        evidence = self._create_evidence(metrics)
        return self._stabilize(metrics, evidence)

    def register_unavailable(self, reason, face_box=None):
        self._handle_region_change(face_box)
        self.invalid_streak += 1
        if (
            self.invalid_streak
            >= config.EXPERIMENTAL_MOIRE_INVALID_RESET_FRAMES
        ):
            self._reset_temporal_state()

        return MoireAnalysisResult(
            False,
            None,
            0.0,
            0.0,
            0.0,
            "Unavailable",
            "none",
            [reason],
            "",
            reason,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None

    def _calculate_metrics(self, power_spectrum, log_spectrum):
        height, width = power_spectrum.shape
        radius_map = self._create_radius_map(width, height)
        analysis_mask = (
            (radius_map >= config.EXPERIMENTAL_MOIRE_INNER_RADIUS)
            & (radius_map <= config.EXPERIMENTAL_MOIRE_OUTER_RADIUS)
        )

        band_power = power_spectrum[analysis_mask]
        total_band_energy = float(band_power.sum())
        if not math.isfinite(total_band_energy) or total_band_energy <= 1e-9:
            return None

        background = cv2.GaussianBlur(
            log_spectrum,
            (0, 0),
            config.EXPERIMENTAL_MOIRE_BACKGROUND_SIGMA,
        )
        residual = log_spectrum - background
        residual_z = self._robust_z_score(residual, analysis_mask)

        contrast_background = cv2.blur(
            log_spectrum,
            (
                config.EXPERIMENTAL_MOIRE_LOCAL_CONTRAST_SIZE,
                config.EXPERIMENTAL_MOIRE_LOCAL_CONTRAST_SIZE,
            ),
        )
        local_contrast = log_spectrum - contrast_background
        contrast_z = self._robust_z_score(local_contrast, analysis_mask)

        maximum_size = config.EXPERIMENTAL_MOIRE_LOCAL_MAXIMUM_SIZE
        local_maximum = residual_z >= cv2.dilate(
            residual_z,
            np.ones((maximum_size, maximum_size), dtype=np.uint8),
        )
        candidate_mask = (
            analysis_mask
            & local_maximum
            & (
                residual_z
                >= config.EXPERIMENTAL_MOIRE_MINIMUM_PEAK_Z_SCORE
            )
            & (
                contrast_z
                >= config.EXPERIMENTAL_MOIRE_MINIMUM_CONTRAST_Z_SCORE
            )
        )

        candidates = self._select_candidates(
            candidate_mask,
            residual_z,
            power_spectrum,
            total_band_energy,
        )
        periodic_score, peak_energy_share = self._periodic_peak_score(
            candidates
        )
        symmetry_score, symmetric_pair_count = self._symmetry_score(
            candidates,
            width,
            height,
        )
        (
            directional_score,
            dominant_direction,
            dominant_direction_share,
        ) = self._directional_score(candidates, width, height)

        symmetry_evidence = periodic_score * symmetry_score
        direction_evidence = periodic_score * directional_score
        raw_moire_score = 100.0 * (
            config.EXPERIMENTAL_MOIRE_PERIODIC_WEIGHT * periodic_score
            + config.EXPERIMENTAL_MOIRE_SYMMETRY_WEIGHT
            * symmetry_evidence
            + config.EXPERIMENTAL_MOIRE_DIRECTION_WEIGHT
            * direction_evidence
        )

        return {
            "raw_moire_score": float(np.clip(raw_moire_score, 0.0, 100.0)),
            "periodic_peak_score": periodic_score,
            "symmetric_peak_score": symmetry_score,
            "directional_concentration_score": directional_score,
            "candidate_peak_count": len(candidates),
            "symmetric_pair_count": symmetric_pair_count,
            "peak_energy_share": peak_energy_share,
            "dominant_direction": dominant_direction,
            "dominant_direction_share": dominant_direction_share,
        }

    def _select_candidates(
        self,
        candidate_mask,
        residual_z,
        power_spectrum,
        total_band_energy,
    ):
        coordinates = np.argwhere(candidate_mask)
        candidates = []
        patch_radius = config.EXPERIMENTAL_MOIRE_PEAK_PATCH_RADIUS
        height, width = power_spectrum.shape

        for y, x in coordinates:
            top = max(0, int(y) - patch_radius)
            bottom = min(height, int(y) + patch_radius + 1)
            left = max(0, int(x) - patch_radius)
            right = min(width, int(x) + patch_radius + 1)
            patch_energy = float(
                power_spectrum[top:bottom, left:right].sum()
            )
            energy_share = patch_energy / (total_band_energy + 1e-9)
            if (
                energy_share
                < config.EXPERIMENTAL_MOIRE_MINIMUM_PEAK_ENERGY_SHARE
            ):
                continue

            candidates.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "z": float(residual_z[y, x]),
                    "power": float(power_spectrum[y, x]),
                    "energy_share": energy_share,
                }
            )

        candidates.sort(
            key=lambda candidate: (
                candidate["z"] * candidate["energy_share"]
            ),
            reverse=True,
        )

        # Ayni genis tepenin komsu piksellerini farkli periyodik peak'ler
        # olarak saymamak icin basit bir non-maximum suppression uygula.
        selected = []
        minimum_distance = (
            config.EXPERIMENTAL_MOIRE_PEAK_MINIMUM_DISTANCE
        )
        for candidate in candidates:
            overlaps_selected_peak = any(
                math.hypot(
                    candidate["x"] - selected_peak["x"],
                    candidate["y"] - selected_peak["y"],
                ) < minimum_distance
                for selected_peak in selected
            )
            if overlaps_selected_peak:
                continue
            selected.append(candidate)
            if (
                len(selected)
                >= config.EXPERIMENTAL_MOIRE_MAXIMUM_PEAK_COUNT
            ):
                break
        return selected

    def _periodic_peak_score(self, candidates):
        if not candidates:
            return 0.0, 0.0

        strongest = candidates[:8]
        mean_peak_z = float(
            np.mean([candidate["z"] for candidate in strongest])
        )
        prominence_score = self._scale(
            mean_peak_z,
            config.EXPERIMENTAL_MOIRE_MINIMUM_PEAK_Z_SCORE,
            config.EXPERIMENTAL_MOIRE_STRONG_PEAK_Z_SCORE,
        )
        count_score = min(
            1.0,
            len(candidates) / config.EXPERIMENTAL_MOIRE_TARGET_PEAK_COUNT,
        )
        peak_energy_share = min(
            1.0,
            sum(candidate["energy_share"] for candidate in candidates),
        )
        energy_score = self._scale(
            peak_energy_share,
            config.EXPERIMENTAL_MOIRE_ENERGY_CONCENTRATION_START,
            config.EXPERIMENTAL_MOIRE_ENERGY_CONCENTRATION_FULL,
        )
        score = (
            0.55 * prominence_score
            + 0.20 * count_score
            + 0.25 * energy_score
        )
        score *= min(1.0, len(candidates) / 2.0)
        return float(np.clip(score, 0.0, 1.0)), peak_energy_share

    def _symmetry_score(self, candidates, width, height):
        if len(candidates) < 2:
            return 0.0, 0

        center_x = width // 2
        center_y = height // 2
        tolerance = config.EXPERIMENTAL_MOIRE_SYMMETRY_TOLERANCE_PIXELS
        half_plane = [
            candidate
            for candidate in candidates
            if candidate["y"] < center_y
            or (
                candidate["y"] == center_y
                and candidate["x"] < center_x
            )
        ]
        if not half_plane:
            return 0.0, 0

        match_scores = []
        for candidate in half_plane:
            target_x = 2 * center_x - candidate["x"]
            target_y = 2 * center_y - candidate["y"]
            best_match = None
            best_distance = float("inf")

            for counterpart in candidates:
                distance = math.hypot(
                    counterpart["x"] - target_x,
                    counterpart["y"] - target_y,
                )
                if distance <= tolerance and distance < best_distance:
                    best_match = counterpart
                    best_distance = distance

            if best_match is None:
                continue

            larger_power = max(candidate["power"], best_match["power"])
            smaller_power = min(candidate["power"], best_match["power"])
            amplitude_ratio = smaller_power / (larger_power + 1e-9)
            if (
                amplitude_ratio
                < config.EXPERIMENTAL_MOIRE_MINIMUM_SYMMETRY_AMPLITUDE_RATIO
            ):
                continue

            distance_score = 1.0 - min(1.0, best_distance / tolerance)
            match_scores.append(amplitude_ratio * (0.7 + 0.3 * distance_score))

        if not match_scores:
            return 0.0, 0

        coverage = len(match_scores) / len(half_plane)
        symmetry_score = float(np.mean(match_scores)) * coverage
        return float(np.clip(symmetry_score, 0.0, 1.0)), len(match_scores)

    def _directional_score(self, candidates, width, height):
        if len(candidates) < 2:
            return 0.0, "none", 0.0

        center_x = width // 2
        center_y = height // 2
        half_plane = [
            candidate
            for candidate in candidates
            if candidate["y"] < center_y
            or (
                candidate["y"] == center_y
                and candidate["x"] < center_x
            )
        ]
        if not half_plane:
            return 0.0, "none", 0.0

        angles = []
        weights = []
        for candidate in half_plane:
            angle = math.atan2(
                candidate["y"] - center_y,
                candidate["x"] - center_x,
            ) % math.pi
            angles.append(angle)
            weights.append(candidate["energy_share"])

        histogram, edges = np.histogram(
            angles,
            bins=config.EXPERIMENTAL_MOIRE_DIRECTION_BIN_COUNT,
            range=(0.0, math.pi),
            weights=weights,
        )
        total_weight = float(histogram.sum())
        if total_weight <= 1e-12:
            return 0.0, "none", 0.0

        dominant_index = int(np.argmax(histogram))
        dominant_share = float(histogram[dominant_index] / total_weight)
        direction_score = self._scale(
            dominant_share,
            config.EXPERIMENTAL_MOIRE_DIRECTION_SHARE_START,
            config.EXPERIMENTAL_MOIRE_DIRECTION_SHARE_FULL,
        )
        direction_angle = float(
            (edges[dominant_index] + edges[dominant_index + 1]) / 2.0
        )
        direction_name = self._direction_name(direction_angle)
        return direction_score, direction_name, dominant_share

    def _stabilize(self, metrics, evidence):
        raw_score = metrics["raw_moire_score"]
        self.score_history.append(raw_score)

        if raw_score >= config.EXPERIMENTAL_MOIRE_SUSPICIOUS_SCORE:
            self.suspicious_streak += 1
            self.normal_streak = 0
        elif raw_score <= config.EXPERIMENTAL_MOIRE_RELEASE_SCORE:
            self.normal_streak += 1
            self.suspicious_streak = 0
        else:
            self.suspicious_streak = 0
            self.normal_streak = 0

        history_ready = (
            len(self.score_history)
            >= config.EXPERIMENTAL_MOIRE_MINIMUM_HISTORY
        )
        stable_score = float(np.median(self.score_history))

        if (
            history_ready
            and self.suspicious_streak
            >= config.EXPERIMENTAL_MOIRE_REQUIRED_SUSPICIOUS_FRAMES
            and stable_score >= config.EXPERIMENTAL_MOIRE_SUSPICIOUS_SCORE
        ):
            self.warning_is_active = True

        if (
            self.warning_is_active
            and self.normal_streak
            >= config.EXPERIMENTAL_MOIRE_REQUIRED_RELEASE_FRAMES
        ):
            recent_scores = list(self.score_history)[
                -config.EXPERIMENTAL_MOIRE_REQUIRED_RELEASE_FRAMES :
            ]
            if float(np.median(recent_scores)) <= (
                config.EXPERIMENTAL_MOIRE_RELEASE_SCORE
            ):
                self.warning_is_active = False

        possible_attack = "none"
        warning = ""
        if self.warning_is_active:
            if (
                stable_score
                >= config.EXPERIMENTAL_MOIRE_SCREEN_REPLAY_SCORE
                and metrics["symmetric_peak_score"] >= 0.45
                and metrics["directional_concentration_score"] >= 0.35
            ):
                status = "Possible Screen Replay"
                possible_attack = "screen_replay"
            else:
                status = "Suspicious"
                possible_attack = "periodic_pattern"
            warning = (
                "WARNING: Possible screen replay / periodic display pattern"
            )
        elif not history_ready or raw_score >= (
            config.EXPERIMENTAL_MOIRE_SUSPICIOUS_SCORE
        ):
            status = "Analysis Uncertain"
        else:
            status = "Normal"

        metrics = dict(metrics)
        metrics["stable_moire_score"] = stable_score
        metrics["history_length"] = len(self.score_history)
        return MoireAnalysisResult(
            True,
            stable_score,
            metrics["periodic_peak_score"],
            metrics["symmetric_peak_score"],
            metrics["directional_concentration_score"],
            status,
            possible_attack,
            evidence,
            warning,
            "Sufficient",
            metrics,
        )

    def _create_evidence(self, metrics):
        evidence = []
        if metrics["periodic_peak_score"] >= 0.55:
            evidence.append("Strong periodic peaks detected")
        if (
            metrics["periodic_peak_score"] >= 0.45
            and metrics["symmetric_peak_score"] >= 0.55
        ):
            evidence.append("Symmetric frequency peaks found")
        if (
            metrics["periodic_peak_score"] >= 0.45
            and metrics["directional_concentration_score"] >= 0.50
        ):
            evidence.append(
                "Energy concentrated in a narrow %s frequency direction"
                % metrics["dominant_direction"]
            )
        if not evidence:
            evidence.append("No stable periodic display evidence")
        return evidence

    def _handle_region_change(self, face_box):
        current_region = self._region_tuple(face_box)
        if current_region is None:
            return

        if self.previous_region is not None:
            overlap = self._intersection_over_union(
                self.previous_region,
                current_region,
            )
            if overlap < config.EXPERIMENTAL_MOIRE_REGION_IOU_RESET_THRESHOLD:
                self._reset_temporal_state()

        self.previous_region = current_region

    def _reset_temporal_state(self):
        self.score_history.clear()
        self.suspicious_streak = 0
        self.normal_streak = 0
        self.invalid_streak = 0
        self.warning_is_active = False

    def _spectrum_is_valid(self, power_spectrum, log_spectrum):
        return (
            power_spectrum.ndim == 2
            and log_spectrum.ndim == 2
            and power_spectrum.shape == log_spectrum.shape
            and power_spectrum.size > 0
            and np.all(np.isfinite(power_spectrum))
            and np.all(np.isfinite(log_spectrum))
            and np.all(power_spectrum >= 0)
        )

    def _robust_z_score(self, values, mask):
        selected = values[mask]
        median = float(np.median(selected))
        median_deviation = float(np.median(np.abs(selected - median)))
        robust_scale = max(1.4826 * median_deviation, 1e-6)
        return (values - median) / robust_scale

    def _create_radius_map(self, width, height):
        x_coordinates = np.linspace(-1.0, 1.0, width)
        y_coordinates = np.linspace(-1.0, 1.0, height)
        grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
        return np.sqrt(grid_x * grid_x + grid_y * grid_y)

    def _direction_name(self, angle):
        degrees = math.degrees(angle) % 180.0
        if degrees <= 22.5 or degrees >= 157.5:
            return "horizontal"
        if 67.5 <= degrees <= 112.5:
            return "vertical"
        return "diagonal"

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

    def _scale(self, value, minimum, maximum):
        if maximum <= minimum:
            return 0.0
        return float(np.clip((value - minimum) / (maximum - minimum), 0, 1))
