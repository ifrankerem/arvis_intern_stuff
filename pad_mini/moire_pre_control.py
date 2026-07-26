"""Ortak FFT context'inden Moire/periyodik desen suphe sinyali uretir.

Bu modul FFT hesaplamaz. ``ModelFreePreControlContext`` icindeki merkezi zaten
kaydirilmis sayisal power ve analitik log-power spektrumlarini kullanir.
"""

from collections import deque
import math

import cv2
import numpy as np

import config
from model_free_analysis import ModelFreeAnalysisResult


class MoirePeriodicPatternPreController:
    """FFT peak, merkez simetrisi ve yon yogunlasmasini inceler.

    Sac/sakal, cizgili kiyafet, desenli arka plan, panjur, JPEG sikistirmasi,
    kamera keskinlestirmesi ve dijital yeniden boyutlandirma da periyodik peak
    uretebilir. Bu nedenle sinyal konservatiftir, tek kareyle warning vermez ve
    gerceklik/fake siniflandirmasi olarak kullanilmamalidir. Mevcut FFT girisi
    Hann pencereli ROI'den geldigi icin crop kenarlari zaten bastirilmistir.
    """

    MODULE_NAME = "Moiré"

    def __init__(self):
        self.score_history = deque(
            maxlen=config.EXPERIMENTAL_MOIRE_HISTORY_SIZE
        )
        self.suspicious_streak = 0
        self.normal_streak = 0
        self.invalid_streak = 0
        self.warning_is_active = False
        self.previous_region = None
        self.latest_candidate_peaks = []
        self.latest_metrics = None
        self.latest_local_heatmap = None

    def analyze(self, context):
        face_box = context.face_bounding_box
        self._handle_region_change(face_box)

        if not context.face_quality_valid:
            return self.register_unavailable(
                context.quality_reason or "face quality gate failed",
                face_box,
            )

        power_spectrum = context.power_spectrum
        log_spectrum = context.log_power_spectrum
        if power_spectrum is None or log_spectrum is None:
            return self.register_unavailable(
                "shared FFT spectrum unavailable",
                face_box,
            )

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

        global_candidate_peaks = list(self.latest_candidate_peaks)
        local_metrics = self._calculate_local_patch_metrics(
            context.grayscale_crop
        )
        self.latest_candidate_peaks = global_candidate_peaks
        metrics["global_moire_score"] = metrics["raw_moire_score"]
        metrics.update(local_metrics)
        metrics["raw_moire_score"] = max(
            metrics["global_moire_score"],
            local_metrics["local_patch_moire_score"],
        )
        self.latest_metrics = dict(metrics)

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

        self.latest_candidate_peaks = []
        self.latest_metrics = None
        self.latest_local_heatmap = None

        return ModelFreeAnalysisResult.unavailable(
            self.MODULE_NAME,
            reason,
            debug_data={
                "possible_attack": "none",
                "quality_status": reason,
            },
            calibrated=False,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None
        self.latest_candidate_peaks = []
        self.latest_metrics = None
        self.latest_local_heatmap = None

    def get_local_heatmap(self):
        if self.latest_local_heatmap is None:
            return None
        return self.latest_local_heatmap.copy()

    def create_debug_visualization(self, log_magnitude_visualization):
        """Draw only the periodic-peak candidates measured by this module."""
        if log_magnitude_visualization is None:
            return None
        visualization = cv2.cvtColor(
            log_magnitude_visualization.copy(),
            cv2.COLOR_GRAY2BGR,
        )
        height, width = visualization.shape[:2]
        cv2.drawMarker(
            visualization,
            (width // 2, height // 2),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )
        for candidate in self.latest_candidate_peaks:
            cv2.circle(
                visualization,
                (int(candidate["x"]), int(candidate["y"])),
                4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        metrics = self.latest_metrics or {}
        label = "Peaks: %d  Symmetric pairs: %d" % (
            int(metrics.get("candidate_peak_count", 0)),
            int(metrics.get("symmetric_pair_count", 0)),
        )
        cv2.putText(
            visualization,
            label,
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            label,
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return visualization

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
        self.latest_candidate_peaks = [dict(candidate) for candidate in candidates]
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

        metrics = {
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
        self.latest_metrics = dict(metrics)
        return metrics

    def _calculate_local_patch_metrics(self, grayscale_crop):
        empty = {
            "local_patch_moire_score": 0.0,
            "local_patch_count": 0,
            "local_patch_strong_count": 0,
            "local_patch_vote_ratio": 0.0,
            "local_patch_top_score_mean": 0.0,
            "local_patch_orientation_consistency": 0.0,
            "local_patch_dominant_direction": "none",
            "local_patch_measurements": [],
        }
        self.latest_local_heatmap = None
        if grayscale_crop is None or grayscale_crop.size == 0:
            return empty
        if grayscale_crop.ndim == 3:
            grayscale = cv2.cvtColor(grayscale_crop, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = grayscale_crop
        height, width = grayscale.shape[:2]
        minimum_side = min(height, width)
        patch_size = int(round(
            minimum_side * config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_SIZE_RATIO
        ))
        patch_size = int(np.clip(
            patch_size,
            config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_SIZE,
            config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_SIZE,
        ))
        patch_size += patch_size % 2
        if width < patch_size * 2 or height < patch_size * 2:
            return empty

        window = np.outer(
            np.hanning(patch_size),
            np.hanning(patch_size),
        ).astype(np.float32)
        margin = config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_INNER_MARGIN_RATIO
        measurements = []
        heat_sum = np.zeros((height, width), dtype=np.float32)
        heat_weight = np.zeros((height, width), dtype=np.float32)
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                center_x = x + patch_size / 2.0
                center_y = y + patch_size / 2.0
                if not (
                    margin * width
                    <= center_x
                    <= config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_X_RATIO
                    * width
                    and margin * height
                    <= center_y
                    <= config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_Y_RATIO
                    * height
                ):
                    continue
                patch = grayscale[
                    y : y + patch_size,
                    x : x + patch_size,
                ].astype(np.float32)
                patch_std = float(patch.std())
                if patch_std < config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_STD:
                    continue
                patch = (patch - float(patch.mean())) / max(patch_std, 1e-6)
                patch *= window
                shifted_fft = np.fft.fftshift(np.fft.fft2(patch))
                patch_power = np.abs(shifted_fft) ** 2
                patch_log_power = np.log1p(patch_power).astype(np.float32)
                patch_metrics = self._calculate_metrics(
                    patch_power,
                    patch_log_power,
                )
                if patch_metrics is None:
                    continue
                score = float(patch_metrics["raw_moire_score"])
                is_strong = bool(
                    score
                    >= config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_STRONG_SCORE
                    and patch_metrics["candidate_peak_count"] >= 2
                    and patch_metrics["symmetric_pair_count"] >= 1
                )
                measurement = {
                    "x": int(x),
                    "y": int(y),
                    "size": int(patch_size),
                    "score": score,
                    "strong": is_strong,
                    "candidate_peak_count": int(
                        patch_metrics["candidate_peak_count"]
                    ),
                    "symmetric_pair_count": int(
                        patch_metrics["symmetric_pair_count"]
                    ),
                    "dominant_direction": patch_metrics[
                        "dominant_direction"
                    ],
                }
                measurements.append(measurement)
                heat_sum[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += score
                heat_weight[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += 1.0

        if not measurements:
            return empty
        heat = np.divide(
            heat_sum,
            heat_weight,
            out=np.zeros_like(heat_sum),
            where=heat_weight > 0.0,
        )
        heat_uint8 = np.clip(heat * 2.55, 0.0, 255.0).astype(np.uint8)
        self.latest_local_heatmap = cv2.applyColorMap(
            heat_uint8,
            cv2.COLORMAP_TURBO,
        )

        strong = [item for item in measurements if item["strong"]]
        strong.sort(key=lambda item: item["score"], reverse=True)
        vote_ratio = len(strong) / len(measurements)
        if len(strong) < config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_VOTES:
            local_score = 0.0
            top_mean = 0.0
            orientation_consistency = 0.0
            dominant_direction = "none"
        else:
            top = strong[: config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_TOP_COUNT]
            top_mean = float(np.mean([item["score"] for item in top]))
            directions = [item["dominant_direction"] for item in strong]
            direction_counts = {
                direction: directions.count(direction)
                for direction in set(directions)
                if direction != "none"
            }
            if direction_counts:
                dominant_direction = max(
                    direction_counts,
                    key=direction_counts.get,
                )
                orientation_consistency = (
                    direction_counts[dominant_direction] / len(strong)
                )
            else:
                dominant_direction = "none"
                orientation_consistency = 0.0
            vote_support = min(
                1.0,
                len(strong)
                / config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_VOTES,
            )
            local_score = (
                config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_PEAK_WEIGHT
                * top_mean
                + 100.0
                * config.EXPERIMENTAL_MOIRE_LOCAL_PATCH_CONSISTENCY_WEIGHT
                * orientation_consistency
                * vote_support
            )

        return {
            "local_patch_moire_score": float(
                np.clip(local_score, 0.0, 100.0)
            ),
            "local_patch_count": len(measurements),
            "local_patch_strong_count": len(strong),
            "local_patch_vote_ratio": float(vote_ratio),
            "local_patch_top_score_mean": float(top_mean),
            "local_patch_orientation_consistency": float(
                orientation_consistency
            ),
            "local_patch_dominant_direction": dominant_direction,
            "local_patch_measurements": measurements,
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

        strongest = candidates[
            : config.EXPERIMENTAL_MOIRE_TARGET_PEAK_COUNT
        ]
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
            config.EXPERIMENTAL_MOIRE_PROMINENCE_WEIGHT
            * prominence_score
            + config.EXPERIMENTAL_MOIRE_PEAK_COUNT_WEIGHT * count_score
            + config.EXPERIMENTAL_MOIRE_PEAK_ENERGY_WEIGHT * energy_score
        )
        score *= min(
            1.0,
            len(candidates)
            / config.EXPERIMENTAL_MOIRE_MINIMUM_PERIODIC_PEAK_COUNT,
        )
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
            match_scores.append(
                amplitude_ratio
                * (
                    config.EXPERIMENTAL_MOIRE_SYMMETRY_AMPLITUDE_WEIGHT
                    + config.EXPERIMENTAL_MOIRE_SYMMETRY_DISTANCE_WEIGHT
                    * distance_score
                )
            )

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
                and metrics["symmetric_peak_score"]
                >= config.EXPERIMENTAL_MOIRE_SCREEN_REPLAY_SYMMETRY_SCORE
                and metrics["directional_concentration_score"]
                >= config.EXPERIMENTAL_MOIRE_SCREEN_REPLAY_DIRECTION_SCORE
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
            config.EXPERIMENTAL_MOIRE_LOCAL_SUPPORTING_SCORE
        ):
            status = "Analysis Uncertain"
        else:
            status = "Normal"

        confidence = min(
            1.0,
            len(self.score_history)
            / config.EXPERIMENTAL_MOIRE_MINIMUM_HISTORY,
        )
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=dict(metrics),
            raw_score=raw_score,
            stabilized_score=stable_score,
            confidence=confidence,
            status=status,
            evidence=evidence,
            warnings=[warning] if warning else [],
            debug_data={
                "possible_attack": possible_attack,
                "quality_status": "Sufficient",
                "history_length": len(self.score_history),
            },
            calibrated=False,
        )

    def _create_evidence(self, metrics):
        evidence = []
        if metrics["periodic_peak_score"] >= (
            config.EXPERIMENTAL_MOIRE_PERIODIC_EVIDENCE_SCORE
        ):
            evidence.append("Strong periodic peaks detected")
        if (
            metrics["periodic_peak_score"]
            >= config.EXPERIMENTAL_MOIRE_SUPPORTING_PERIODIC_SCORE
            and metrics["symmetric_peak_score"]
            >= config.EXPERIMENTAL_MOIRE_SYMMETRY_EVIDENCE_SCORE
        ):
            evidence.append("Symmetric frequency peaks found")
        if (
            metrics["periodic_peak_score"]
            >= config.EXPERIMENTAL_MOIRE_SUPPORTING_PERIODIC_SCORE
            and metrics["directional_concentration_score"]
            >= config.EXPERIMENTAL_MOIRE_DIRECTION_EVIDENCE_SCORE
        ):
            evidence.append(
                "Energy concentrated in a narrow %s frequency direction"
                % metrics["dominant_direction"]
            )
        if metrics.get("local_patch_moire_score", 0.0) >= (
            config.EXPERIMENTAL_MOIRE_LOCAL_SUPPORTING_SCORE
        ):
            evidence.append(
                "Consistent symmetric periodic peaks detected in %d local "
                "face patches"
                % metrics.get("local_patch_strong_count", 0)
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
