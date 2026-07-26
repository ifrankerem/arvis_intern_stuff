"""Module 5: model-free, cok olcekli wavelet yuz-dokusu analizi."""

from collections import deque
import json
import math

import cv2
import numpy as np

import config
from model_free_analysis import ModelFreeAnalysisResult

try:
    import pywt
except ImportError as import_error:
    pywt = None
    PYWAVELETS_IMPORT_ERROR = str(import_error)
else:
    PYWAVELETS_IMPORT_ERROR = None


PYWAVELETS_AVAILABLE = pywt is not None


class WaveletAnalysisPreController:
    """Standardize hizalanmis crop'ta aciklanabilir wavelet analizi yapar.

    Modul bir sinir agi kullanmaz. Heatmap, patch feature sapmalarinin yuz
    koordinatlarina geri yerlestirilmis aciklayici bir gorselidir; attention
    map degildir ve tek basina karar uretmez.
    """

    MODULE_NAME = "Wavelet"
    DETAIL_BANDS = (
        ("horizontal", "LH"),
        ("vertical", "HL"),
        ("diagonal", "HH"),
    )

    def __init__(self):
        history_size = config.EXPERIMENTAL_WAVELET_HISTORY_SIZE
        self.score_history = deque(maxlen=history_size)
        self.component_histories = {
            name: deque(maxlen=history_size)
            for name in config.EXPERIMENTAL_WAVELET_COMPONENT_WEIGHTS
        }
        self.invalid_streak = 0
        self.previous_region = None
        self.calibration = self._load_calibration()

        self.latest_analysis_crop = None
        self.latest_decomposition = None
        self.latest_anomaly_values = None
        self.latest_anomaly_heatmap = None
        self.latest_feature_report = None

    def analyze(self, context):
        self._handle_region_change(context.face_bounding_box)
        if not PYWAVELETS_AVAILABLE:
            return self._register_unavailable(
                "PyWavelets dependency is not installed",
                dependency_missing=True,
            )

        invalid_reason = self._context_invalid_reason(context)
        if invalid_reason is not None:
            return self._register_unavailable(invalid_reason)

        crop = np.ascontiguousarray(
            context.standardized_aligned_face_crop,
            dtype=np.float32,
        )
        inner_mask = self._create_inner_face_mask(crop.shape)
        quality_features = self._quality_features(
            context,
            crop,
            inner_mask,
        )
        if quality_features["clipped_pixel_ratio"] >= (
            config.WAVELET_UNAVAILABLE_CLIPPING_RATIO
        ):
            return self._register_unavailable(
                "severe intensity clipping makes wavelet analysis unreliable",
                raw_features=quality_features,
            )

        decomposition_error = self._decomposition_invalid_reason(crop)
        if decomposition_error is not None:
            return self._register_unavailable(
                decomposition_error,
                raw_features=quality_features,
            )

        try:
            levels = self._decompose(crop)
        except (TypeError, ValueError) as error:
            return self._register_unavailable(
                "wavelet decomposition failed: " + str(error),
                raw_features=quality_features,
            )

        features = dict(quality_features)
        global_features = self._extract_global_features(
            levels,
            inner_mask,
        )
        local_features, anomaly_values = self._analyze_local_detail(
            levels,
            inner_mask,
            crop.shape,
        )
        features.update(global_features)
        features.update(local_features)

        energy_score, energy_deviations = self._score_feature_profiles(
            features,
            config.EXPERIMENTAL_WAVELET_ENERGY_FEATURE_PROFILES,
        )
        directional_score = self._directional_score(features)
        local_score = float(features["local_wavelet_inconsistency_score"])
        component_scores = {
            "wavelet_energy_score": energy_score,
            "directional_wavelet_score": directional_score,
            "local_wavelet_inconsistency_score": local_score,
        }
        features.update(component_scores)
        final_score = self._weighted_component_score(component_scores)

        scoring_mode = config.WAVELET_SCORING_MODE.lower()
        calibrated = (
            self.calibration is not None
            and scoring_mode in ("auto", "calibrated")
        )
        calibrated_score = None
        calibration_deviations = {}
        if calibrated:
            calibrated_score, calibration_deviations = (
                self._score_feature_profiles(
                    features,
                    self.calibration["feature_profiles"],
                )
            )
            blend = config.EXPERIMENTAL_WAVELET_CALIBRATION_BLEND_WEIGHT
            final_score = (
                (1.0 - blend) * final_score
                + blend * calibrated_score
            )
        final_score = float(np.clip(final_score, 0.0, 100.0))
        features["calibrated_feature_anomaly_score"] = calibrated_score
        features["final_wavelet_score"] = final_score

        self._update_debug_outputs(
            crop,
            levels,
            anomaly_values,
            features,
        )

        if scoring_mode == "calibrated" and not calibrated:
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=[
                    "Compatible wavelet calibration data is unavailable"
                ],
                debug_data=self._debug_data(
                    context,
                    "calibration-required",
                    energy_deviations,
                    calibration_deviations,
                ),
            )
        if scoring_mode not in ("auto", "experimental", "calibrated"):
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=["Unsupported wavelet scoring mode"],
                debug_data={"scoring_mode": scoring_mode},
            )

        return self._stabilize_result(
            context,
            features,
            component_scores,
            final_score,
            calibrated,
            energy_deviations,
            calibration_deviations,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None
        self._clear_debug_outputs()

    def get_debug_subbands(self):
        if self.latest_decomposition is None:
            return None
        return {
            level_number: {
                name: values.copy()
                for name, values in level_data.items()
            }
            for level_number, level_data in self.latest_decomposition.items()
        }

    def get_anomaly_heatmap(self):
        if self.latest_anomaly_values is None:
            return None
        if self.latest_anomaly_heatmap is None:
            values = np.clip(
                self.latest_anomaly_values,
                0.0,
                100.0,
            )
            normalized = np.rint(values * 2.55).astype(np.uint8)
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            cv2.putText(
                heatmap,
                "Explanatory wavelet patch anomaly",
                (7, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            self.latest_anomaly_heatmap = heatmap
        return self.latest_anomaly_heatmap.copy()

    def get_feature_report(self):
        if self.latest_feature_report is None:
            return None
        return dict(self.latest_feature_report)

    def create_normalized_subband_visualization(
        self,
        values,
        is_detail,
    ):
        values = np.asarray(values, dtype=np.float64)
        if is_detail:
            maximum = float(np.percentile(np.abs(values), 99.0))
            if maximum <= 1e-12:
                return np.zeros(values.shape, dtype=np.uint8)
            return np.rint(
                255.0 * np.clip(np.abs(values) / maximum, 0.0, 1.0)
            ).astype(np.uint8)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum <= 1e-12:
            return np.zeros(values.shape, dtype=np.uint8)
        return np.rint(
            255.0 * (values - minimum) / (maximum - minimum)
        ).astype(np.uint8)

    def _context_invalid_reason(self, context):
        if not context.face_quality_valid:
            return context.quality_reason or "face quality gate failed"
        if context.aligned_face_crop is None:
            return "aligned face crop is unavailable"
        if context.aligned_face_crop.size == 0:
            return "aligned face crop is empty"
        if context.standardized_aligned_face_crop is None:
            return "standardized aligned face crop is unavailable"
        crop = context.standardized_aligned_face_crop
        if crop.ndim != 2:
            return "standardized aligned face crop must be grayscale"
        if not np.all(np.isfinite(crop)):
            return "standardized aligned face crop contains invalid values"
        if context.face_dimensions is None:
            return "source face dimensions are unavailable"
        if min(context.face_dimensions) < config.WAVELET_MINIMUM_SOURCE_SIDE:
            return "source face resolution is too low for wavelet analysis"
        if float(np.std(crop)) <= 1e-6:
            return "aligned face crop has insufficient intensity variation"
        return None

    def _decomposition_invalid_reason(self, crop):
        levels = int(config.WAVELET_DECOMPOSITION_LEVELS)
        if levels not in (1, 2):
            return "wavelet decomposition level must be one or two"
        if min(crop.shape) < 2 ** (levels + 2):
            return "aligned face crop is too small for wavelet decomposition"
        try:
            wavelet = pywt.Wavelet(config.WAVELET_NAME)
            maximum_level = pywt.dwt_max_level(
                min(crop.shape),
                wavelet.dec_len,
            )
        except (AttributeError, TypeError, ValueError) as error:
            return "invalid wavelet configuration: " + str(error)
        if maximum_level < levels:
            return (
                "invalid decomposition size for wavelet %s at level %d"
                % (config.WAVELET_NAME, levels)
            )
        return None

    def _decompose(self, crop):
        current = crop
        result = {}
        for level_number in range(
            1,
            int(config.WAVELET_DECOMPOSITION_LEVELS) + 1,
        ):
            approximation, details = pywt.dwt2(
                current,
                config.WAVELET_NAME,
                mode=config.WAVELET_BOUNDARY_MODE,
            )
            horizontal, vertical, diagonal = details
            arrays = (
                approximation,
                horizontal,
                vertical,
                diagonal,
            )
            if any(array.size == 0 for array in arrays):
                raise ValueError("empty wavelet subband")
            if any(not np.all(np.isfinite(array)) for array in arrays):
                raise ValueError("wavelet subband contains invalid values")
            result[level_number] = {
                "LL": np.asarray(approximation, dtype=np.float32),
                "LH": np.asarray(horizontal, dtype=np.float32),
                "HL": np.asarray(vertical, dtype=np.float32),
                "HH": np.asarray(diagonal, dtype=np.float32),
            }
            current = approximation
        return result

    def _create_inner_face_mask(self, shape):
        if not config.WAVELET_USE_INNER_FACE_MASK:
            return np.ones(shape, dtype=bool)
        height, width = shape
        y_coordinates, x_coordinates = np.indices(shape)
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius_x = max(
            1.0,
            width * config.WAVELET_INNER_MASK_HORIZONTAL_RADIUS_RATIO,
        )
        radius_y = max(
            1.0,
            height * config.WAVELET_INNER_MASK_VERTICAL_RADIUS_RATIO,
        )
        return (
            ((x_coordinates - center_x) / radius_x) ** 2
            + ((y_coordinates - center_y) / radius_y) ** 2
            <= 1.0
        )

    def _mask_for_shape(self, mask, shape):
        resized = cv2.resize(
            mask.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)

    def _quality_features(self, context, crop, inner_mask):
        values = crop[inner_mask]
        clipped = (
            (values <= config.WAVELET_CLIPPED_PIXEL_LOW)
            | (values >= config.WAVELET_CLIPPED_PIXEL_HIGH)
        )
        source_side = min(context.face_dimensions)
        analysis_side = min(crop.shape)
        return {
            "analysis_width": int(crop.shape[1]),
            "analysis_height": int(crop.shape[0]),
            "source_face_width": int(context.face_dimensions[0]),
            "source_face_height": int(context.face_dimensions[1]),
            "standardization_upscale_ratio": (
                analysis_side / max(float(source_side), 1.0)
            ),
            "clipped_pixel_ratio": float(np.mean(clipped)),
            "inner_face_mask_ratio": float(np.mean(inner_mask)),
            "wavelet_name": config.WAVELET_NAME,
            "wavelet_decomposition_levels": int(
                config.WAVELET_DECOMPOSITION_LEVELS
            ),
            "wavelet_boundary_mode": config.WAVELET_BOUNDARY_MODE,
        }

    def _extract_global_features(self, levels, inner_mask):
        features = {}
        entropies = []
        sparsities = []
        anisotropies = []
        dominances = []
        for level_number, level_data in levels.items():
            mask = self._mask_for_shape(
                inner_mask,
                level_data["LL"].shape,
            )
            approximation_values = level_data["LL"][mask]
            approximation_energy = float(
                np.sum(approximation_values.astype(np.float64) ** 2)
            )
            prefix = "level_%d_" % level_number
            features[prefix + "approximation_energy"] = approximation_energy
            features[prefix + "approximation_mean"] = float(
                np.mean(approximation_values)
            )
            features[prefix + "approximation_variance"] = float(
                np.var(approximation_values)
            )

            detail_energies = {}
            detail_statistics = {}
            for direction, band_name in self.DETAIL_BANDS:
                values = level_data[band_name][mask].astype(np.float64)
                statistics = self._detail_statistics(values)
                detail_statistics[direction] = statistics
                detail_energies[direction] = statistics["energy"]
                for statistic_name, statistic_value in statistics.items():
                    features[
                        prefix + direction + "_" + statistic_name
                    ] = statistic_value
                entropies.append(statistics["entropy"])
                sparsities.append(statistics["sparsity_ratio"])

            total_detail_energy = float(sum(detail_energies.values()))
            features[prefix + "total_detail_energy"] = total_detail_energy
            features[
                prefix + "detail_to_approximation_energy_ratio"
            ] = total_detail_energy / max(approximation_energy, 1e-12)

            direction_ratios = {}
            for direction, energy in detail_energies.items():
                ratio = energy / max(total_detail_energy, 1e-12)
                direction_ratios[direction] = ratio
                features[
                    prefix + direction + "_normalized_energy_ratio"
                ] = ratio

            ratio_values = list(direction_ratios.values())
            anisotropy = max(ratio_values) - min(ratio_values)
            dominance = max(ratio_values)
            dominant_direction = max(
                direction_ratios,
                key=direction_ratios.get,
            )
            features[prefix + "directional_energy_anisotropy"] = float(
                anisotropy
            )
            features[prefix + "maximum_directional_energy_ratio"] = float(
                dominance
            )
            features[prefix + "dominant_detail_direction"] = (
                dominant_direction
            )
            features[prefix + "horizontal_to_vertical_energy_ratio"] = (
                detail_energies["horizontal"]
                / max(detail_energies["vertical"], 1e-12)
            )
            anisotropies.append(anisotropy)
            dominances.append(dominance)

        features["global_detail_entropy_mean"] = float(np.mean(entropies))
        features["global_detail_entropy_std"] = float(np.std(entropies))
        features["global_detail_sparsity_mean"] = float(
            np.mean(sparsities)
        )
        features["global_detail_sparsity_std"] = float(np.std(sparsities))
        features["global_directional_anisotropy_maximum"] = float(
            np.max(anisotropies)
        )
        features["global_directional_dominance_maximum"] = float(
            np.max(dominances)
        )
        return features

    def _detail_statistics(self, values):
        values = np.asarray(values, dtype=np.float64)
        absolute = np.abs(values)
        median = float(np.median(values))
        return {
            "energy": float(np.sum(values ** 2)),
            "mean_absolute_coefficient": float(np.mean(absolute)),
            "variance": float(np.var(values)),
            "median_absolute_deviation": float(
                np.median(np.abs(values - median))
            ),
            "entropy": self._coefficient_entropy(values),
            "kurtosis": self._coefficient_kurtosis(values),
            "sparsity_ratio": float(
                np.mean(
                    absolute <= config.WAVELET_DETAIL_NEAR_ZERO_THRESHOLD
                )
            ),
        }

    def _coefficient_entropy(self, values):
        magnitudes = np.abs(np.asarray(values, dtype=np.float64))
        if magnitudes.size <= 1:
            return 0.0
        total = float(np.sum(magnitudes))
        if total <= 1e-12:
            return 0.0
        probabilities = magnitudes / total
        nonzero = probabilities > 0
        entropy = -float(
            np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
        )
        return entropy / math.log2(magnitudes.size)

    def _coefficient_kurtosis(self, values):
        values = np.asarray(values, dtype=np.float64)
        centered = values - float(np.mean(values))
        second_moment = float(np.mean(centered ** 2))
        if second_moment <= 1e-12:
            return 0.0
        fourth_moment = float(np.mean(centered ** 4))
        return fourth_moment / (second_moment ** 2)

    def _analyze_local_detail(self, levels, inner_mask, face_shape):
        heatmap_sum = np.zeros(face_shape, dtype=np.float32)
        heatmap_count = np.zeros(face_shape, dtype=np.float32)
        level_scores = []
        features = {}
        for level_number, level_data in levels.items():
            level_features, patch_records = self._analyze_level_patches(
                level_number,
                level_data,
                inner_mask,
            )
            features.update(level_features)
            level_score = level_features[
                "level_%d_local_inconsistency_score" % level_number
            ]
            level_scores.append(level_score)
            self._map_patch_scores_to_face(
                patch_records,
                level_number,
                face_shape,
                heatmap_sum,
                heatmap_count,
            )

        if level_scores:
            final_local_score = float(
                0.60 * np.max(level_scores)
                + 0.40 * np.mean(level_scores)
            )
        else:
            final_local_score = 0.0
        anomaly_values = np.divide(
            heatmap_sum,
            heatmap_count,
            out=np.zeros_like(heatmap_sum),
            where=heatmap_count > 0,
        )
        anomaly_values[~inner_mask] = 0.0
        features["local_wavelet_inconsistency_score"] = float(
            np.clip(final_local_score, 0.0, 100.0)
        )
        features["wavelet_heatmap_maximum"] = float(
            np.max(anomaly_values)
        )
        features["wavelet_heatmap_mean_inside_face"] = float(
            np.mean(anomaly_values[inner_mask])
        )
        return features, anomaly_values

    def _analyze_level_patches(
        self,
        level_number,
        level_data,
        inner_mask,
    ):
        shape = level_data["LH"].shape
        level_mask = self._mask_for_shape(inner_mask, shape)
        patch_size = max(
            4,
            int(
                round(
                    config.WAVELET_LOCAL_PATCH_FACE_SIZE
                    / float(2 ** level_number)
                )
            ),
        )
        patch_rows = shape[0] // patch_size
        patch_columns = shape[1] // patch_size
        records = []
        record_grid = {}
        for row in range(patch_rows):
            top = row * patch_size
            for column in range(patch_columns):
                left = column * patch_size
                patch_mask = level_mask[
                    top : top + patch_size,
                    left : left + patch_size,
                ]
                coverage = float(np.mean(patch_mask))
                if coverage < config.WAVELET_LOCAL_MINIMUM_MASK_COVERAGE:
                    continue
                energies = []
                for _direction, band_name in self.DETAIL_BANDS:
                    patch = level_data[band_name][
                        top : top + patch_size,
                        left : left + patch_size,
                    ]
                    values = patch[patch_mask].astype(np.float64)
                    energies.append(float(np.mean(values ** 2)))
                total_energy = float(sum(energies))
                shares = np.asarray(energies, dtype=np.float64) / max(
                    total_energy,
                    1e-12,
                )
                record = {
                    "row": row,
                    "column": column,
                    "top": top,
                    "left": left,
                    "patch_size": patch_size,
                    "energy_log": math.log1p(total_energy),
                    "direction_shares": shares,
                }
                records.append(record)
                record_grid[(row, column)] = record

        prefix = "level_%d_" % level_number
        if len(records) < 4:
            return self._empty_local_level_features(prefix), records

        energy_logs = np.asarray(
            [record["energy_log"] for record in records],
            dtype=np.float64,
        )
        direction_shares = np.stack(
            [record["direction_shares"] for record in records],
            axis=0,
        )
        energy_center = float(np.median(energy_logs))
        energy_scale = max(
            float(np.median(np.abs(energy_logs - energy_center))) * 1.4826,
            0.35,
        )
        direction_center = np.median(direction_shares, axis=0)
        direction_scale = np.maximum(
            np.median(
                np.abs(direction_shares - direction_center),
                axis=0,
            )
            * 1.4826,
            0.10,
        )

        neighbor_differences = []
        for record in records:
            record["neighbor_difference"] = 0.0
            for offset in ((0, 1), (1, 0)):
                neighbor = record_grid.get(
                    (
                        record["row"] + offset[0],
                        record["column"] + offset[1],
                    )
                )
                if neighbor is None:
                    continue
                difference = abs(
                    record["energy_log"] - neighbor["energy_log"]
                ) / energy_scale
                neighbor_differences.append(difference)
                record["neighbor_difference"] = max(
                    record["neighbor_difference"],
                    difference,
                )
                neighbor["neighbor_difference"] = max(
                    neighbor.get("neighbor_difference", 0.0),
                    difference,
                )

        energy_z_scores = []
        direction_distances = []
        patch_scores = []
        for record in records:
            energy_z = (
                record["energy_log"] - energy_center
            ) / energy_scale
            direction_z = (
                record["direction_shares"] - direction_center
            ) / direction_scale
            direction_distance = float(
                np.sqrt(np.mean(np.minimum(direction_z ** 2, 400.0)))
            )
            energy_score = self._linear_score(
                abs(energy_z),
                config.EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_START,
                config.EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_FULL,
            )
            direction_score = self._linear_score(
                direction_distance,
                config.EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_START,
                config.EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_FULL,
            )
            neighbor_score = self._linear_score(
                record["neighbor_difference"],
                config.EXPERIMENTAL_WAVELET_NEIGHBOR_DIFFERENCE_START,
                config.EXPERIMENTAL_WAVELET_NEIGHBOR_DIFFERENCE_FULL,
            )
            patch_score = float(
                np.clip(
                    0.50 * energy_score
                    + 0.30 * direction_score
                    + 0.20 * neighbor_score,
                    0.0,
                    100.0,
                )
            )
            record["energy_z_score"] = float(energy_z)
            record["direction_distance"] = direction_distance
            record["anomaly_score"] = patch_score
            energy_z_scores.append(energy_z)
            direction_distances.append(direction_distance)
            patch_scores.append(patch_score)

        energy_z_scores = np.asarray(energy_z_scores, dtype=np.float64)
        direction_distances = np.asarray(
            direction_distances,
            dtype=np.float64,
        )
        patch_scores = np.asarray(patch_scores, dtype=np.float64)
        neighbor_array = np.asarray(
            neighbor_differences or [0.0],
            dtype=np.float64,
        )
        high_ratio = float(
            np.mean(
                energy_z_scores
                >= config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE
            )
        )
        smooth_ratio = float(
            np.mean(
                energy_z_scores
                <= -config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE
            )
        )
        directional_ratio = float(
            np.mean(
                direction_distances
                >= config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE
            )
        )
        outlier_ratio = float(
            np.mean(
                (np.abs(energy_z_scores)
                 >= config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE)
                | (direction_distances
                   >= config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE)
            )
        )
        outlier_score = float(
            np.clip(
                100.0
                * outlier_ratio
                / config.EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_RATIO_FULL,
                0.0,
                100.0,
            )
        )
        level_score = float(
            np.clip(
                0.65 * np.percentile(patch_scores, 95)
                + 0.35 * outlier_score,
                0.0,
                100.0,
            )
        )
        features = {
            prefix + "local_patch_count": len(records),
            prefix + "local_energy_log_median": energy_center,
            prefix + "local_energy_robust_distance_p95": float(
                np.percentile(np.abs(energy_z_scores), 95)
            ),
            prefix + "isolated_high_frequency_patch_ratio": high_ratio,
            prefix + "unusually_smooth_patch_ratio": smooth_ratio,
            prefix + "directional_patch_anomaly_ratio": directional_ratio,
            prefix + "directional_patch_distance_p95": float(
                np.percentile(direction_distances, 95)
            ),
            prefix + "neighbor_patch_difference_mean": float(
                np.mean(neighbor_array)
            ),
            prefix + "neighbor_patch_difference_p95": float(
                np.percentile(neighbor_array, 95)
            ),
            prefix + "local_patch_anomaly_p90": float(
                np.percentile(patch_scores, 90)
            ),
            prefix + "local_patch_anomaly_p95": float(
                np.percentile(patch_scores, 95)
            ),
            prefix + "local_outlier_patch_ratio": outlier_ratio,
            prefix + "local_inconsistency_score": level_score,
        }
        return features, records

    def _empty_local_level_features(self, prefix):
        return {
            prefix + "local_patch_count": 0,
            prefix + "local_energy_log_median": 0.0,
            prefix + "local_energy_robust_distance_p95": 0.0,
            prefix + "isolated_high_frequency_patch_ratio": 0.0,
            prefix + "unusually_smooth_patch_ratio": 0.0,
            prefix + "directional_patch_anomaly_ratio": 0.0,
            prefix + "directional_patch_distance_p95": 0.0,
            prefix + "neighbor_patch_difference_mean": 0.0,
            prefix + "neighbor_patch_difference_p95": 0.0,
            prefix + "local_patch_anomaly_p90": 0.0,
            prefix + "local_patch_anomaly_p95": 0.0,
            prefix + "local_outlier_patch_ratio": 0.0,
            prefix + "local_inconsistency_score": 0.0,
        }

    def _map_patch_scores_to_face(
        self,
        records,
        level_number,
        face_shape,
        heatmap_sum,
        heatmap_count,
    ):
        scale = 2 ** level_number
        for record in records:
            if "anomaly_score" not in record:
                continue
            top = min(face_shape[0], record["top"] * scale)
            left = min(face_shape[1], record["left"] * scale)
            bottom = min(
                face_shape[0],
                (record["top"] + record["patch_size"]) * scale,
            )
            right = min(
                face_shape[1],
                (record["left"] + record["patch_size"]) * scale,
            )
            heatmap_sum[top:bottom, left:right] += record["anomaly_score"]
            heatmap_count[top:bottom, left:right] += 1.0

    def _directional_score(self, features):
        scores = []
        for level_number in range(
            1,
            int(config.WAVELET_DECOMPOSITION_LEVELS) + 1,
        ):
            prefix = "level_%d_" % level_number
            dominance_score = self._linear_score(
                features[prefix + "maximum_directional_energy_ratio"],
                config.EXPERIMENTAL_WAVELET_DIRECTION_DOMINANCE_START,
                config.EXPERIMENTAL_WAVELET_DIRECTION_DOMINANCE_FULL,
            )
            anisotropy_score = self._linear_score(
                features[prefix + "directional_energy_anisotropy"],
                config.EXPERIMENTAL_WAVELET_ANISOTROPY_START,
                config.EXPERIMENTAL_WAVELET_ANISOTROPY_FULL,
            )
            scores.append(0.55 * dominance_score + 0.45 * anisotropy_score)
        return float(np.clip(np.max(scores), 0.0, 100.0))

    def _score_feature_profiles(self, features, profiles):
        weighted_sum = 0.0
        total_weight = 0.0
        deviations = {}
        for name, profile in profiles.items():
            if name not in features or features[name] is None:
                continue
            value = features[name]
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            deviation = self._range_deviation(
                float(value),
                float(profile["minimum"]),
                float(profile["maximum"]),
                float(profile["deviation_scale"]),
            )
            weight = float(profile.get("weight", 1.0))
            deviations[name] = deviation
            weighted_sum += deviation * weight
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("wavelet feature profiles contain no usable weight")
        return (
            float(np.clip(100.0 * weighted_sum / total_weight, 0.0, 100.0)),
            deviations,
        )

    def _weighted_component_score(self, component_scores):
        weighted_sum = 0.0
        total_weight = 0.0
        for name, weight in (
            config.EXPERIMENTAL_WAVELET_COMPONENT_WEIGHTS.items()
        ):
            weighted_sum += float(component_scores[name]) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            raise ValueError("wavelet component weights must be positive")
        return weighted_sum / total_weight

    def _stabilize_result(
        self,
        context,
        features,
        component_scores,
        final_score,
        calibrated,
        energy_deviations,
        calibration_deviations,
    ):
        self.invalid_streak = 0
        self.score_history.append(final_score)
        for name, score in component_scores.items():
            self.component_histories[name].append(score)

        stable_score = float(np.median(self.score_history))
        stable_components = {
            name: float(np.median(history))
            for name, history in self.component_histories.items()
        }
        history_length = len(self.score_history)
        history_ready = (
            history_length >= config.EXPERIMENTAL_WAVELET_MINIMUM_HISTORY
        )
        uncertainty_reasons = self._quality_uncertainty_reasons(
            context,
            features,
        )

        warnings = []
        if uncertainty_reasons:
            status = "Analysis uncertain"
            warnings.append("; ".join(uncertainty_reasons))
        elif not history_ready:
            status = "Analysis uncertain" if calibrated else "Uncalibrated"
        elif (
            stable_score >= config.EXPERIMENTAL_WAVELET_SUSPICIOUS_SCORE
            and (
                calibrated
                or sum(score >= 50.0 for score in stable_components.values())
                >= 2
            )
        ):
            status = "Suspicious multi-scale texture"
            warnings.append(
                "Multiple wavelet signals are elevated; this is not definitive fraud evidence"
            )
        elif stable_components["local_wavelet_inconsistency_score"] >= (
            config.EXPERIMENTAL_WAVELET_LOCAL_STATUS_SCORE
        ):
            status = "Local detail inconsistency"
            warnings.append(
                "Several local detail patches differ from neighboring inner-face regions"
            )
        elif stable_components["directional_wavelet_score"] >= (
            config.EXPERIMENTAL_WAVELET_DIRECTIONAL_STATUS_SCORE
        ):
            status = "Directional wavelet anomaly"
            warnings.append(
                "Directional detail energy is unusually concentrated"
            )
        else:
            status = "Normal multi-scale texture"

        evidence = self._create_evidence(
            stable_components,
            calibrated,
            energy_deviations,
            calibration_deviations,
        )
        temporal_confidence = min(
            1.0,
            history_length / config.EXPERIMENTAL_WAVELET_MINIMUM_HISTORY,
        )
        confidence_limit = (
            1.0
            if calibrated
            else config.EXPERIMENTAL_WAVELET_MAXIMUM_CONFIDENCE
        )
        confidence = temporal_confidence * confidence_limit
        if uncertainty_reasons:
            confidence *= 0.35

        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=features,
            raw_score=final_score,
            stabilized_score=stable_score,
            confidence=confidence,
            status=status,
            evidence=evidence,
            warnings=warnings,
            debug_data=self._debug_data(
                context,
                "calibrated" if calibrated else "experimental-uncalibrated",
                energy_deviations,
                calibration_deviations,
                history_length,
                stable_components,
                uncertainty_reasons,
            ),
            calibrated=calibrated,
        )

    def _quality_uncertainty_reasons(self, context, features):
        reasons = []
        if context.blur_value is not None and context.blur_value < (
            config.WAVELET_UNCERTAIN_BLUR_SCORE
        ):
            reasons.append("severe blur reduces wavelet reliability")
        if min(context.face_dimensions) < config.WAVELET_UNCERTAIN_SOURCE_SIDE:
            reasons.append("source face resolution is marginal")
        if features["standardization_upscale_ratio"] > (
            config.WAVELET_UNCERTAIN_UPSCALE_RATIO
        ):
            reasons.append("standardization requires substantial upscaling")
        if features["clipped_pixel_ratio"] >= (
            config.WAVELET_UNCERTAIN_CLIPPING_RATIO
        ):
            reasons.append("intensity clipping affects wavelet coefficients")
        return reasons

    def _create_evidence(
        self,
        stable_components,
        calibrated,
        energy_deviations,
        calibration_deviations,
    ):
        threshold = config.EXPERIMENTAL_WAVELET_EVIDENCE_SCORE
        labels = {
            "wavelet_energy_score": "Multi-scale detail energy deviation",
            "directional_wavelet_score": "Directional detail concentration",
            "local_wavelet_inconsistency_score": (
                "Repeated local wavelet patch inconsistency"
            ),
        }
        evidence = [
            labels[name]
            for name, score in stable_components.items()
            if score >= threshold
        ]
        if not evidence:
            evidence.append("No strong multi-scale texture anomaly")
        if energy_deviations:
            evidence.append("Raw wavelet energy deviations retained")
        if calibrated and calibration_deviations:
            evidence.append("Compatible wavelet calibration profiles applied")
        else:
            evidence.append("Experimental score; calibration unavailable")
        evidence.append(
            "A single high-energy patch is not interpreted as definitive fraud"
        )
        return evidence

    def _debug_data(
        self,
        context,
        scoring_mode,
        energy_deviations,
        calibration_deviations,
        history_length=0,
        stable_components=None,
        uncertainty_reasons=None,
    ):
        return {
            "possible_attack": "none",
            "quality_status": (
                "Uncertain" if uncertainty_reasons else "Sufficient"
            ),
            "scoring_mode": scoring_mode,
            "history_length": history_length,
            "stabilized_component_scores": stable_components or {},
            "energy_feature_deviations": energy_deviations,
            "calibration_feature_deviations": calibration_deviations,
            "uncertainty_reasons": uncertainty_reasons or [],
            "wavelet_name": config.WAVELET_NAME,
            "decomposition_levels": config.WAVELET_DECOMPOSITION_LEVELS,
            "boundary_mode": config.WAVELET_BOUNDARY_MODE,
            "inner_face_mask_used": config.WAVELET_USE_INNER_FACE_MASK,
            "alignment_applied": context.alignment_applied,
            "pose_alignment_valid": context.pose_alignment_valid,
            "heatmap_interpretation": (
                "Explanatory patch anomaly map; not neural-network attention"
            ),
            "false_positive_factors": [
                "normal camera or video compression",
                "blur or denoising",
                "sharpening",
                "lighting gradients",
                "facial hair and natural texture",
                "resizing",
            ],
        }

    def _update_debug_outputs(
        self,
        crop,
        levels,
        anomaly_values,
        features,
    ):
        self.latest_analysis_crop = crop.copy()
        self.latest_decomposition = {
            level_number: {
                name: values.copy()
                for name, values in level_data.items()
            }
            for level_number, level_data in levels.items()
        }
        self.latest_anomaly_values = anomaly_values.copy()
        self.latest_anomaly_heatmap = None
        self.latest_feature_report = dict(features)

    def _load_calibration(self):
        path = config.MODEL_FREE_CALIBRATION_FILE_PATH
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as calibration_file:
                document = json.load(calibration_file)
            section = None
            for key in ("wavelet_analysis", "wavelet"):
                candidate = document.get(key)
                if isinstance(candidate, dict):
                    section = candidate
                    break
            if section is None:
                return None
            if section.get("wavelet_name", config.WAVELET_NAME) != (
                config.WAVELET_NAME
            ):
                return None
            if int(
                section.get(
                    "decomposition_levels",
                    config.WAVELET_DECOMPOSITION_LEVELS,
                )
            ) != int(config.WAVELET_DECOMPOSITION_LEVELS):
                return None
            raw_profiles = section.get("feature_profiles")
            if not isinstance(raw_profiles, dict):
                return None
            profiles = {}
            for name, raw_profile in raw_profiles.items():
                profile = self._validated_calibration_profile(raw_profile)
                if profile is not None:
                    profiles[str(name)] = profile
            if not profiles:
                return None
            return {"feature_profiles": profiles}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print("Wavelet calibration could not be loaded: " + str(error))
            return None

    def _validated_calibration_profile(self, profile):
        if not isinstance(profile, dict):
            return None
        try:
            if all(
                key in profile
                for key in ("minimum", "maximum", "deviation_scale")
            ):
                minimum = float(profile["minimum"])
                maximum = float(profile["maximum"])
                scale = float(profile["deviation_scale"])
            elif "mean" in profile and "standard_deviation" in profile:
                mean = float(profile["mean"])
                standard_deviation = float(profile["standard_deviation"])
                minimum = mean - 2.0 * standard_deviation
                maximum = mean + 2.0 * standard_deviation
                scale = max(2.0 * standard_deviation, 1e-9)
            else:
                return None
            weight = float(profile.get("weight", 1.0))
        except (TypeError, ValueError):
            return None
        values = (minimum, maximum, scale, weight)
        if not all(math.isfinite(value) for value in values):
            return None
        if maximum < minimum or scale <= 0 or weight <= 0:
            return None
        return {
            "minimum": minimum,
            "maximum": maximum,
            "deviation_scale": scale,
            "weight": weight,
        }

    def _register_unavailable(
        self,
        reason,
        raw_features=None,
        dependency_missing=False,
    ):
        self.invalid_streak += 1
        self._clear_debug_outputs()
        if self.invalid_streak >= (
            config.EXPERIMENTAL_WAVELET_INVALID_RESET_FRAMES
        ):
            self._reset_temporal_state()
        debug_data = {
            "possible_attack": "none",
            "quality_status": reason,
            "scoring_mode": (
                "calibrated" if self.calibration else "experimental-uncalibrated"
            ),
            "dependency": "PyWavelets",
            "dependency_available": PYWAVELETS_AVAILABLE,
        }
        if dependency_missing:
            debug_data["dependency_install_hint"] = (
                "Install the optional PyWavelets dependency to enable Module 5"
            )
            debug_data["import_error"] = PYWAVELETS_IMPORT_ERROR
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=False,
            raw_features=dict(raw_features or {}),
            raw_score=None,
            stabilized_score=None,
            confidence=0.0,
            status="Analysis unavailable",
            evidence=[reason],
            warnings=[reason] if dependency_missing else [],
            debug_data=debug_data,
            calibrated=False,
        )

    def _handle_region_change(self, face_box):
        current = self._region_tuple(face_box)
        if current is None:
            return
        if self.previous_region is not None:
            overlap = self._intersection_over_union(
                self.previous_region,
                current,
            )
            if overlap < (
                config.EXPERIMENTAL_WAVELET_REGION_IOU_RESET_THRESHOLD
            ):
                self._reset_temporal_state()
        self.previous_region = current

    def _reset_temporal_state(self):
        self.score_history.clear()
        for history in self.component_histories.values():
            history.clear()
        self.invalid_streak = 0

    def _clear_debug_outputs(self):
        self.latest_analysis_crop = None
        self.latest_decomposition = None
        self.latest_anomaly_values = None
        self.latest_anomaly_heatmap = None
        self.latest_feature_report = None

    def _range_deviation(self, value, minimum, maximum, scale):
        if not math.isfinite(value):
            return 1.0
        if value < minimum:
            distance = minimum - value
        elif value > maximum:
            distance = value - maximum
        else:
            return 0.0
        return float(np.clip(distance / scale, 0.0, 1.0))

    def _linear_score(self, value, start, full):
        if full <= start:
            raise ValueError("wavelet score interval must be increasing")
        return float(
            np.clip(100.0 * (value - start) / (full - start), 0.0, 100.0)
        )

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
        return intersection / union if union > 0 else 0.0


WaveletPreController = WaveletAnalysisPreController
