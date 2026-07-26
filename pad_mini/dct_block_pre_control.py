"""Module 4: decoded face pixels icin model-free DCT / blok analizi."""

from collections import deque
import json
import math

import cv2
import numpy as np

import config
from model_free_analysis import ModelFreeAnalysisResult


class DCTBlockAnalysisPreController:
    """8x8 DCT dagilimlarini ve blok siniri sureksizliklerini olcer.

    Girdi decode edilmis kamera karesidir. Bu nedenle modul JPEG quantization
    tablosu, JPEG kalite faktoru veya double-JPEG gecmisi cikarmaya calismaz.
    Yalnizca mevcut standardize yuz piksellerindeki matematiksel yapilari
    raporlar; normal video/kamera sikistirmasi da ayni yapilari uretebilir.
    """

    MODULE_NAME = "DCT / Block Analysis"

    def __init__(self):
        history_size = config.EXPERIMENTAL_DCT_HISTORY_SIZE
        self.score_history = deque(maxlen=history_size)
        self.component_histories = {
            name: deque(maxlen=history_size)
            for name in config.EXPERIMENTAL_DCT_COMPONENT_WEIGHTS
        }
        self.invalid_streak = 0
        self.previous_region = None
        self.calibration = self._load_calibration()

        self.block_size = int(config.DCT_BLOCK_SIZE)
        self.low_mask, self.middle_mask, self.high_mask = (
            self._create_frequency_masks()
        )
        self.ac_mask = self.low_mask | self.middle_mask | self.high_mask

        self.latest_analysis_crop = None
        self.latest_band_energy_map = None
        self.latest_boundary_visualization = None
        self.latest_blockiness_heatmap = None
        self.latest_coefficient_report = None
        self.latest_block_maps = None
        self.latest_blockiness_maps = None

    def analyze(self, context):
        """Ortak context'teki standardize hizalanmis crop'u analiz eder."""
        self._handle_region_change(context.face_bounding_box)
        invalid_reason = self._context_invalid_reason(context)
        if invalid_reason is not None:
            return self._register_unavailable(invalid_reason)

        crop = self._valid_block_crop(
            context.standardized_aligned_face_crop
        )
        block_rows = crop.shape[0] // self.block_size
        block_columns = crop.shape[1] // self.block_size
        block_count = block_rows * block_columns
        if block_count < config.DCT_MINIMUM_BLOCK_COUNT:
            return self._register_unavailable(
                "too few complete 8x8 blocks in aligned face crop"
            )

        coefficients = self._calculate_block_dct(crop)
        features, block_maps = self._extract_coefficient_features(
            coefficients
        )
        blockiness_features, blockiness_maps = self._analyze_block_boundaries(
            crop
        )
        local_features = self._analyze_local_consistency(block_maps)
        features.update(blockiness_features)
        features.update(local_features)
        features.update(
            {
                "block_size": self.block_size,
                "block_rows": block_rows,
                "block_columns": block_columns,
                "complete_block_count": block_count,
                "analysis_width": int(crop.shape[1]),
                "analysis_height": int(crop.shape[0]),
                "source_face_width": int(context.face_dimensions[0]),
                "source_face_height": int(context.face_dimensions[1]),
                "near_zero_threshold": float(
                    config.DCT_NEAR_ZERO_COEFFICIENT_THRESHOLD
                ),
                "standardization_upscale_ratio": (
                    min(crop.shape[:2])
                    / max(float(min(context.face_dimensions)), 1.0)
                ),
            }
        )

        band_score, band_deviations = self._score_feature_profiles(
            features,
            config.EXPERIMENTAL_DCT_BAND_FEATURE_PROFILES,
        )
        sparsity_score, sparsity_deviations = self._score_feature_profiles(
            features,
            config.EXPERIMENTAL_DCT_SPARSITY_FEATURE_PROFILES,
        )
        blockiness_score = float(features["blockiness_score"])
        local_score = float(features["local_dct_inconsistency_score"])

        component_scores = {
            "dct_band_anomaly_score": band_score,
            "coefficient_sparsity_score": sparsity_score,
            "blockiness_score": blockiness_score,
            "local_dct_inconsistency_score": local_score,
        }
        final_score = self._weighted_component_score(component_scores)
        features.update(component_scores)

        scoring_mode = config.DCT_SCORING_MODE.lower()
        calibrated = (
            self.calibration is not None
            and scoring_mode in ("auto", "calibrated")
        )
        calibrated_score = None
        if calibrated:
            calibrated_score, calibration_deviations = (
                self._score_feature_profiles(
                    features,
                    self.calibration["feature_profiles"],
                )
            )
            blend = config.EXPERIMENTAL_DCT_CALIBRATION_BLEND_WEIGHT
            final_score = (
                (1.0 - blend) * final_score
                + blend * calibrated_score
            )
        else:
            calibration_deviations = {}

        final_score = float(np.clip(final_score, 0.0, 100.0))
        features["calibrated_feature_anomaly_score"] = calibrated_score
        features["final_dct_score"] = final_score

        self._update_debug_outputs(
            crop,
            block_maps,
            blockiness_maps,
            features,
        )

        if scoring_mode == "calibrated" and not calibrated:
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=[
                    "Compatible DCT/block calibration data is unavailable"
                ],
                debug_data=self._debug_data(
                    context,
                    "calibration-required",
                    band_deviations,
                    sparsity_deviations,
                    calibration_deviations,
                ),
            )
        if scoring_mode not in ("auto", "experimental", "calibrated"):
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=["Unsupported DCT scoring mode"],
                debug_data={"scoring_mode": scoring_mode},
            )

        return self._stabilize_result(
            context,
            features,
            component_scores,
            final_score,
            calibrated,
            band_deviations,
            sparsity_deviations,
            calibration_deviations,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None
        self._clear_debug_outputs()

    def get_debug_images(self):
        """Son gecerli karenin istenen dort debug varligindan gorselleri."""
        if (
            self.latest_analysis_crop is None
            or self.latest_block_maps is None
            or self.latest_blockiness_maps is None
        ):
            return None
        if self.latest_band_energy_map is None:
            self.latest_band_energy_map = self._create_band_energy_map(
                self.latest_block_maps
            )
        if self.latest_boundary_visualization is None:
            self.latest_boundary_visualization = (
                self._create_boundary_visualization(
                    self.latest_analysis_crop,
                    self.latest_blockiness_maps,
                )
            )
        if self.latest_blockiness_heatmap is None:
            self.latest_blockiness_heatmap = self._create_blockiness_heatmap(
                self.latest_blockiness_maps["block_heatmap"],
                self.latest_analysis_crop.shape,
            )
        images = {
            "dct_band_energy_map": self.latest_band_energy_map,
            "block_boundary_visualization": (
                self.latest_boundary_visualization
            ),
            "blockiness_heatmap": self.latest_blockiness_heatmap,
        }
        if any(image is None for image in images.values()):
            return None
        return {name: image.copy() for name, image in images.items()}

    def get_coefficient_statistics_report(self):
        if self.latest_coefficient_report is None:
            return None
        return dict(self.latest_coefficient_report)

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
        if min(context.face_dimensions) < config.DCT_MINIMUM_SOURCE_SIDE:
            return "source face resolution is too low for DCT analysis"
        if float(np.std(crop)) <= 1e-6:
            return "aligned face crop has insufficient intensity variation"
        return None

    def _valid_block_crop(self, crop):
        height, width = crop.shape[:2]
        valid_height = height - (height % self.block_size)
        valid_width = width - (width % self.block_size)
        return np.ascontiguousarray(
            crop[:valid_height, :valid_width],
            dtype=np.float32,
        )

    def _calculate_block_dct(self, crop):
        block_rows = crop.shape[0] // self.block_size
        block_columns = crop.shape[1] // self.block_size
        coefficients = np.empty(
            (
                block_rows,
                block_columns,
                self.block_size,
                self.block_size,
            ),
            dtype=np.float32,
        )
        for row in range(block_rows):
            top = row * self.block_size
            for column in range(block_columns):
                left = column * self.block_size
                block = crop[
                    top : top + self.block_size,
                    left : left + self.block_size,
                ]
                coefficients[row, column] = cv2.dct(block)
        return coefficients

    def _create_frequency_masks(self):
        coordinates = np.indices((self.block_size, self.block_size))
        index_sum = coordinates[0] + coordinates[1]
        non_dc = index_sum > 0
        low = non_dc & (
            index_sum <= config.DCT_LOW_FREQUENCY_MAX_INDEX_SUM
        )
        middle = (
            index_sum > config.DCT_LOW_FREQUENCY_MAX_INDEX_SUM
        ) & (
            index_sum <= config.DCT_MIDDLE_FREQUENCY_MAX_INDEX_SUM
        )
        high = index_sum > config.DCT_MIDDLE_FREQUENCY_MAX_INDEX_SUM
        return low, middle, high

    def _extract_coefficient_features(self, coefficients):
        epsilon = 1e-12
        dc = coefficients[:, :, 0, 0].astype(np.float64)
        ac = coefficients[:, :, self.ac_mask].astype(np.float64)
        low_energy = np.sum(
            coefficients[:, :, self.low_mask].astype(np.float64) ** 2,
            axis=2,
        )
        middle_energy = np.sum(
            coefficients[:, :, self.middle_mask].astype(np.float64) ** 2,
            axis=2,
        )
        high_energy = np.sum(
            coefficients[:, :, self.high_mask].astype(np.float64) ** 2,
            axis=2,
        )
        total_ac_energy = low_energy + middle_energy + high_energy
        dc_energy = dc ** 2
        low_ratio = low_energy / (total_ac_energy + epsilon)
        middle_ratio = middle_energy / (total_ac_energy + epsilon)
        high_ratio = high_energy / (total_ac_energy + epsilon)
        ac_to_dc = total_ac_energy / (dc_energy + epsilon)

        threshold = config.DCT_NEAR_ZERO_COEFFICIENT_THRESHOLD
        sparsity_map = np.mean(np.abs(ac) <= threshold, axis=2)
        entropy_map = self._coefficient_entropy_map(ac)
        centered_ac = ac - np.mean(ac, axis=2, keepdims=True)
        second_moment = np.mean(centered_ac ** 2, axis=2)
        fourth_moment = np.mean(centered_ac ** 4, axis=2)
        kurtosis_map = np.divide(
            fourth_moment,
            second_moment ** 2,
            out=np.zeros_like(fourth_moment),
            where=second_moment > 1e-12,
        )

        descriptors = np.stack(
            [
                low_ratio,
                middle_ratio,
                high_ratio,
                sparsity_map,
                entropy_map,
                np.log1p(total_ac_energy) / 10.0,
            ],
            axis=2,
        )
        neighbor = self._neighbor_variation(descriptors)
        global_low = float(np.sum(low_energy))
        global_middle = float(np.sum(middle_energy))
        global_high = float(np.sum(high_energy))
        global_ac = global_low + global_middle + global_high

        features = {
            "dc_coefficient_mean": float(np.mean(dc)),
            "dc_coefficient_median": float(np.median(dc)),
            "dc_coefficient_std": float(np.std(dc)),
            "dc_coefficient_minimum": float(np.min(dc)),
            "dc_coefficient_maximum": float(np.max(dc)),
            "dc_energy_mean": float(np.mean(dc_energy)),
            "low_frequency_ac_energy_mean": float(np.mean(low_energy)),
            "low_frequency_ac_energy_median": float(
                np.median(low_energy)
            ),
            "low_frequency_ac_energy_std": float(np.std(low_energy)),
            "middle_frequency_ac_energy_mean": float(
                np.mean(middle_energy)
            ),
            "middle_frequency_ac_energy_median": float(
                np.median(middle_energy)
            ),
            "middle_frequency_ac_energy_std": float(
                np.std(middle_energy)
            ),
            "high_frequency_ac_energy_mean": float(np.mean(high_energy)),
            "high_frequency_ac_energy_median": float(
                np.median(high_energy)
            ),
            "high_frequency_ac_energy_std": float(np.std(high_energy)),
            "total_ac_energy_mean": float(np.mean(total_ac_energy)),
            "low_frequency_ac_energy_ratio": global_low / (
                global_ac + epsilon
            ),
            "middle_frequency_ac_energy_ratio": global_middle / (
                global_ac + epsilon
            ),
            "high_frequency_ac_energy_ratio": global_high / (
                global_ac + epsilon
            ),
            "ac_to_dc_ratio_mean": float(np.mean(ac_to_dc)),
            "ac_to_dc_ratio_median": float(np.median(ac_to_dc)),
            "ac_to_dc_ratio_std": float(np.std(ac_to_dc)),
            "near_zero_ac_coefficient_ratio": float(
                np.mean(np.abs(ac) <= threshold)
            ),
            "zero_ac_coefficient_ratio": float(
                np.mean(np.abs(ac) <= 1e-12)
            ),
            "near_zero_all_coefficient_ratio": float(
                np.mean(np.abs(coefficients) <= threshold)
            ),
            "near_zero_ratio_block_mean": float(np.mean(sparsity_map)),
            "near_zero_ratio_block_std": float(np.std(sparsity_map)),
            "coefficient_entropy_global": self._coefficient_entropy(
                ac.ravel()
            ),
            "coefficient_entropy_definition": (
                "normalized Shannon entropy of absolute AC coefficient mass"
            ),
            "coefficient_entropy_mean": float(np.mean(entropy_map)),
            "coefficient_entropy_median": float(np.median(entropy_map)),
            "coefficient_entropy_std": float(np.std(entropy_map)),
            "coefficient_kurtosis_global": self._coefficient_kurtosis(
                ac.ravel()
            ),
            "coefficient_kurtosis_mean": float(np.mean(kurtosis_map)),
            "coefficient_kurtosis_median": float(
                np.median(kurtosis_map)
            ),
            "coefficient_kurtosis_std": float(np.std(kurtosis_map)),
        }
        features.update(neighbor["features"])
        block_maps = {
            "low_energy": low_energy,
            "middle_energy": middle_energy,
            "high_energy": high_energy,
            "low_ratio": low_ratio,
            "middle_ratio": middle_ratio,
            "high_ratio": high_ratio,
            "sparsity": sparsity_map,
            "entropy": entropy_map,
            "kurtosis": kurtosis_map,
            "total_ac_energy": total_ac_energy,
            "descriptors": descriptors,
        }
        return features, block_maps

    def _neighbor_variation(self, descriptors):
        horizontal = np.mean(
            np.abs(descriptors[:, 1:] - descriptors[:, :-1]),
            axis=2,
        )
        vertical = np.mean(
            np.abs(descriptors[1:] - descriptors[:-1]),
            axis=2,
        )
        values = np.concatenate((horizontal.ravel(), vertical.ravel()))
        return {
            "features": {
                "neighboring_block_variation_mean": float(np.mean(values)),
                "neighboring_block_variation_std": float(np.std(values)),
                "neighboring_block_variation_p95": float(
                    np.percentile(values, 95)
                ),
                "horizontal_neighbor_variation_mean": float(
                    np.mean(horizontal)
                ),
                "vertical_neighbor_variation_mean": float(
                    np.mean(vertical)
                ),
            }
        }

    def _coefficient_entropy(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size <= 1:
            return 0.0
        magnitudes = np.abs(values)
        total = float(np.sum(magnitudes))
        if total <= 1e-12:
            return 0.0
        probabilities = magnitudes / total
        nonzero = probabilities > 0
        entropy = -float(
            np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
        )
        maximum_entropy = math.log2(values.size)
        return entropy / maximum_entropy if maximum_entropy > 0 else 0.0

    def _coefficient_entropy_map(self, values):
        magnitudes = np.abs(np.asarray(values, dtype=np.float64))
        totals = np.sum(magnitudes, axis=2, keepdims=True)
        probabilities = np.divide(
            magnitudes,
            totals,
            out=np.zeros_like(magnitudes),
            where=totals > 1e-12,
        )
        log_probabilities = np.zeros_like(probabilities)
        np.log2(
            probabilities,
            out=log_probabilities,
            where=probabilities > 0,
        )
        entropy = -np.sum(probabilities * log_probabilities, axis=2)
        return entropy / math.log2(values.shape[2])

    def _coefficient_kurtosis(self, values):
        values = np.asarray(values, dtype=np.float64)
        centered = values - float(np.mean(values))
        second_moment = float(np.mean(centered ** 2))
        if second_moment <= 1e-12:
            return 0.0
        fourth_moment = float(np.mean(centered ** 4))
        return fourth_moment / (second_moment ** 2)

    def _analyze_block_boundaries(self, crop):
        horizontal_differences = np.abs(np.diff(crop, axis=1))
        vertical_differences = np.abs(np.diff(crop, axis=0))
        vertical_stats = self._axis_boundary_statistics(
            horizontal_differences,
            axis=1,
        )
        horizontal_stats = self._axis_boundary_statistics(
            vertical_differences,
            axis=0,
        )
        horizontal_score = self._blockiness_ratio_score(
            horizontal_stats["ratio"]
        )
        vertical_score = self._blockiness_ratio_score(
            vertical_stats["ratio"]
        )
        periodicity_score = float(
            np.clip(
                0.5 * (horizontal_score + vertical_score),
                0.0,
                100.0,
            )
        )

        features = {
            "horizontal_block_boundary_discontinuity_mean": (
                horizontal_stats["boundary_mean"]
            ),
            "horizontal_nearby_non_boundary_discontinuity_mean": (
                horizontal_stats["nearby_mean"]
            ),
            "horizontal_block_boundary_ratio": horizontal_stats["ratio"],
            "horizontal_blockiness_score": horizontal_score,
            "vertical_block_boundary_discontinuity_mean": (
                vertical_stats["boundary_mean"]
            ),
            "vertical_nearby_non_boundary_discontinuity_mean": (
                vertical_stats["nearby_mean"]
            ),
            "vertical_block_boundary_ratio": vertical_stats["ratio"],
            "vertical_blockiness_score": vertical_score,
            "horizontal_discontinuity_phase_profile": (
                horizontal_stats["phase_profile"]
            ),
            "vertical_discontinuity_phase_profile": (
                vertical_stats["phase_profile"]
            ),
            "horizontal_8px_phase_prominence": (
                horizontal_stats["phase_prominence"]
            ),
            "vertical_8px_phase_prominence": (
                vertical_stats["phase_prominence"]
            ),
            "combined_8x8_periodicity_score": periodicity_score,
            "blockiness_score": periodicity_score,
        }
        maps = {
            "horizontal_line_scores": horizontal_stats["line_scores"],
            "vertical_line_scores": vertical_stats["line_scores"],
            "block_heatmap": self._block_boundary_heatmap(
                crop,
                horizontal_stats,
                vertical_stats,
            ),
        }
        return features, maps

    def _axis_boundary_statistics(self, differences, axis):
        length = differences.shape[axis]
        boundary_indices = np.arange(
            self.block_size - 1,
            length,
            self.block_size,
            dtype=np.int32,
        )
        line_means = np.mean(differences, axis=1 - axis)
        boundary_values = line_means[boundary_indices]
        nearby_indices = []
        for index in boundary_indices:
            for offset in (-2, -1, 1, 2):
                candidate = int(index + offset)
                if 0 <= candidate < length:
                    if candidate % self.block_size != self.block_size - 1:
                        nearby_indices.append(candidate)
        nearby_indices = np.asarray(sorted(set(nearby_indices)), dtype=int)
        nearby_values = line_means[nearby_indices]
        boundary_mean = float(np.mean(boundary_values))
        nearby_mean = float(np.mean(nearby_values))
        ratio = boundary_mean / max(nearby_mean, 1e-6)

        phase_profile = []
        for phase in range(self.block_size):
            indices = np.arange(phase, length, self.block_size)
            phase_profile.append(float(np.mean(line_means[indices])))
        non_boundary_phases = phase_profile[: self.block_size - 1]
        phase_prominence = phase_profile[-1] / max(
            float(np.mean(non_boundary_phases)),
            1e-6,
        )
        line_scores = [
            self._blockiness_ratio_score(
                float(line_means[index]) / max(nearby_mean, 1e-6)
            )
            for index in boundary_indices
        ]
        return {
            "boundary_indices": boundary_indices.tolist(),
            "boundary_mean": boundary_mean,
            "nearby_mean": nearby_mean,
            "ratio": float(ratio),
            "phase_profile": phase_profile,
            "phase_prominence": float(phase_prominence),
            "line_scores": line_scores,
        }

    def _blockiness_ratio_score(self, ratio):
        start = config.EXPERIMENTAL_DCT_BLOCKINESS_RATIO_START
        full = config.EXPERIMENTAL_DCT_BLOCKINESS_RATIO_FULL
        return self._linear_score(ratio, start, full)

    def _block_boundary_heatmap(
        self,
        crop,
        horizontal_stats,
        vertical_stats,
    ):
        rows = crop.shape[0] // self.block_size
        columns = crop.shape[1] // self.block_size
        heatmap = np.zeros((rows, columns), dtype=np.float32)
        counts = np.zeros((rows, columns), dtype=np.float32)
        horizontal_scores = np.asarray(
            horizontal_stats["line_scores"],
            dtype=np.float32,
        )
        vertical_scores = np.asarray(
            vertical_stats["line_scores"],
            dtype=np.float32,
        )
        if horizontal_scores.size:
            heatmap[:-1] += horizontal_scores[:, None]
            heatmap[1:] += horizontal_scores[:, None]
            counts[:-1] += 1.0
            counts[1:] += 1.0
        if vertical_scores.size:
            heatmap[:, :-1] += vertical_scores[None, :]
            heatmap[:, 1:] += vertical_scores[None, :]
            counts[:, :-1] += 1.0
            counts[:, 1:] += 1.0
        return np.divide(
            heatmap,
            counts,
            out=np.zeros_like(heatmap),
            where=counts > 0,
        )

    def _analyze_local_consistency(self, block_maps):
        descriptors = block_maps["descriptors"]
        rows, columns = descriptors.shape[:2]
        row_margin = max(1, int(round(rows * config.DCT_INNER_FACE_MARGIN_RATIO)))
        column_margin = max(
            1,
            int(round(columns * config.DCT_INNER_FACE_MARGIN_RATIO)),
        )
        inner = descriptors[
            row_margin : rows - row_margin,
            column_margin : columns - column_margin,
        ]
        patch_size = int(config.DCT_LOCAL_PATCH_BLOCK_SIZE)
        patch_rows = inner.shape[0] // patch_size
        patch_columns = inner.shape[1] // patch_size
        if patch_rows < 2 or patch_columns < 2:
            return {
                "inner_face_patch_count": 0,
                "local_dct_distance_mean": 0.0,
                "local_dct_distance_p95": 0.0,
                "local_dct_distance_maximum": 0.0,
                "local_dct_outlier_ratio": 0.0,
                "local_patch_neighbor_variation_mean": 0.0,
                "local_dct_inconsistency_score": 0.0,
            }

        patch_grid = np.empty(
            (patch_rows, patch_columns, descriptors.shape[2]),
            dtype=np.float64,
        )
        for row in range(patch_rows):
            for column in range(patch_columns):
                patch = inner[
                    row * patch_size : (row + 1) * patch_size,
                    column * patch_size : (column + 1) * patch_size,
                ]
                patch_grid[row, column] = np.median(
                    patch.reshape(-1, descriptors.shape[2]),
                    axis=0,
                )

        patch_vectors = patch_grid.reshape(-1, patch_grid.shape[2])
        center = np.median(patch_vectors, axis=0)
        mad = np.median(np.abs(patch_vectors - center), axis=0) * 1.4826
        scale_floor = np.asarray(
            [0.08, 0.06, 0.04, 0.12, 0.12, 0.70],
            dtype=np.float64,
        )
        scale = np.maximum(mad, scale_floor)
        z_scores = (patch_vectors - center) / scale
        distances = np.sqrt(np.mean(np.minimum(z_scores ** 2, 400.0), axis=1))
        distance_grid = distances.reshape(patch_rows, patch_columns)
        neighbor_values = np.concatenate(
            (
                np.abs(np.diff(distance_grid, axis=0)).ravel(),
                np.abs(np.diff(distance_grid, axis=1)).ravel(),
            )
        )
        mean_distance = float(np.mean(distances))
        p95_distance = float(np.percentile(distances, 95))
        maximum_distance = float(np.max(distances))
        outlier_ratio = float(
            np.mean(
                distances
                >= config.EXPERIMENTAL_DCT_LOCAL_OUTLIER_DISTANCE
            )
        )
        neighbor_variation = float(np.mean(neighbor_values))

        distance_score = self._linear_score(
            p95_distance,
            config.EXPERIMENTAL_DCT_LOCAL_DISTANCE_START,
            config.EXPERIMENTAL_DCT_LOCAL_DISTANCE_FULL,
        )
        maximum_score = self._linear_score(
            maximum_distance,
            config.EXPERIMENTAL_DCT_LOCAL_OUTLIER_DISTANCE,
            config.EXPERIMENTAL_DCT_LOCAL_DISTANCE_FULL * 1.5,
        )
        outlier_score = float(
            np.clip(
                100.0
                * outlier_ratio
                / config.EXPERIMENTAL_DCT_LOCAL_OUTLIER_RATIO_FULL,
                0.0,
                100.0,
            )
        )
        local_score = float(
            np.clip(
                0.55 * distance_score
                + 0.25 * maximum_score
                + 0.20 * outlier_score,
                0.0,
                100.0,
            )
        )
        return {
            "inner_face_patch_count": int(distances.size),
            "local_dct_distance_mean": mean_distance,
            "local_dct_distance_p95": p95_distance,
            "local_dct_distance_maximum": maximum_distance,
            "local_dct_outlier_ratio": outlier_ratio,
            "local_patch_neighbor_variation_mean": neighbor_variation,
            "local_dct_inconsistency_score": local_score,
        }

    def _score_feature_profiles(self, features, profiles):
        weighted_sum = 0.0
        total_weight = 0.0
        deviations = {}
        for name, profile in profiles.items():
            if name not in features or features[name] is None:
                continue
            deviation = self._range_deviation(
                float(features[name]),
                float(profile["minimum"]),
                float(profile["maximum"]),
                float(profile["deviation_scale"]),
            )
            weight = float(profile.get("weight", 1.0))
            deviations[name] = deviation
            weighted_sum += deviation * weight
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("DCT feature profiles contain no usable weight")
        return (
            float(np.clip(100.0 * weighted_sum / total_weight, 0.0, 100.0)),
            deviations,
        )

    def _weighted_component_score(self, component_scores):
        weighted_sum = 0.0
        total_weight = 0.0
        for name, weight in config.EXPERIMENTAL_DCT_COMPONENT_WEIGHTS.items():
            weighted_sum += float(component_scores[name]) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            raise ValueError("DCT component weights must be positive")
        return weighted_sum / total_weight

    def _stabilize_result(
        self,
        context,
        features,
        component_scores,
        final_score,
        calibrated,
        band_deviations,
        sparsity_deviations,
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
            history_length >= config.EXPERIMENTAL_DCT_MINIMUM_HISTORY
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
            stable_score >= config.EXPERIMENTAL_DCT_SUSPICIOUS_SCORE
            and (
                calibrated
                or sum(
                    score >= 50.0
                    for score in stable_components.values()
                )
                >= 2
            )
        ):
            status = "Suspicious block-frequency evidence"
            warnings.append(
                "Multiple experimental DCT/block signals are elevated; this is not proof of fraud"
            )
        elif stable_components["blockiness_score"] >= (
            config.EXPERIMENTAL_DCT_BLOCK_STRUCTURE_SCORE
        ):
            status = "Compression-like block structure detected"
            warnings.append(
                "8x8-aligned discontinuities can also come from normal camera/video compression"
            )
        elif stable_components["local_dct_inconsistency_score"] >= (
            config.EXPERIMENTAL_DCT_LOCAL_INCONSISTENCY_SCORE
        ):
            status = "Local DCT inconsistency"
            warnings.append(
                "Inner-face DCT patches are inconsistent; texture and lighting remain possible causes"
            )
        else:
            status = "No strong block anomaly"

        evidence = self._create_evidence(
            stable_components,
            calibrated,
            band_deviations,
            sparsity_deviations,
            calibration_deviations,
        )
        temporal_confidence = min(
            1.0,
            history_length / config.EXPERIMENTAL_DCT_MINIMUM_HISTORY,
        )
        confidence_limit = (
            1.0
            if calibrated
            else config.EXPERIMENTAL_DCT_MAXIMUM_CONFIDENCE
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
                band_deviations,
                sparsity_deviations,
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
            config.DCT_UNCERTAIN_BLUR_SCORE
        ):
            reasons.append("blur reduces DCT/block reliability")
        source_side = min(context.face_dimensions)
        if source_side < config.DCT_UNCERTAIN_SOURCE_SIDE:
            reasons.append("source face resolution is marginal")
        upscale_ratio = float(features["standardization_upscale_ratio"])
        if upscale_ratio > config.DCT_UNCERTAIN_UPSCALE_RATIO:
            reasons.append("standardization requires substantial upscaling")
        if features["near_zero_ac_coefficient_ratio"] >= (
            config.DCT_EXTREME_SPARSITY_RATIO
        ):
            reasons.append(
                "extreme smoothing or video compression makes coefficients unreliable"
            )
        return reasons

    def _create_evidence(
        self,
        stable_components,
        calibrated,
        band_deviations,
        sparsity_deviations,
        calibration_deviations,
    ):
        threshold = config.EXPERIMENTAL_DCT_EVIDENCE_SCORE
        labels = {
            "dct_band_anomaly_score": "DCT band distribution anomaly",
            "coefficient_sparsity_score": "Unusual AC coefficient sparsity",
            "blockiness_score": "8x8 boundary periodicity",
            "local_dct_inconsistency_score": "Inner-face local DCT variation",
        }
        evidence = [
            labels[name]
            for name, score in stable_components.items()
            if score >= threshold
        ]
        if not evidence:
            evidence.append("No strong block-frequency evidence")
        if band_deviations:
            evidence.append("Raw DCT band deviations retained in debug data")
        if sparsity_deviations:
            evidence.append("Raw coefficient deviations retained in debug data")
        if calibrated and calibration_deviations:
            evidence.append("Compatible DCT calibration profiles applied")
        else:
            evidence.append("Experimental score; calibration unavailable")
        evidence.append(
            "Decoded pixels cannot reveal original JPEG tables or definitive double-JPEG history"
        )
        return evidence

    def _debug_data(
        self,
        context,
        scoring_mode,
        band_deviations,
        sparsity_deviations,
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
            "band_feature_deviations": band_deviations,
            "sparsity_feature_deviations": sparsity_deviations,
            "calibration_feature_deviations": calibration_deviations,
            "uncertainty_reasons": uncertainty_reasons or [],
            "alignment_applied": context.alignment_applied,
            "pose_alignment_valid": context.pose_alignment_valid,
            "encoded_jpeg_bytes_available": False,
            "excluded_claims": [
                "original JPEG quantization tables",
                "definitive JPEG quality factor",
                "definitive double-JPEG history",
            ],
            "false_positive_factors": [
                "normal camera or video compression",
                "resizing",
                "severe blur",
                "denoising or smoothing",
                "sharpening",
                "natural facial texture and lighting",
            ],
        }

    def _update_debug_outputs(
        self,
        crop,
        block_maps,
        blockiness_maps,
        features,
    ):
        self.latest_analysis_crop = crop.copy()
        self.latest_block_maps = block_maps
        self.latest_blockiness_maps = blockiness_maps
        self.latest_band_energy_map = None
        self.latest_boundary_visualization = None
        self.latest_blockiness_heatmap = None
        self.latest_coefficient_report = dict(features)

    def _create_band_energy_map(self, block_maps):
        red = self._normalize_log_map(block_maps["low_energy"])
        green = self._normalize_log_map(block_maps["middle_energy"])
        blue = self._normalize_log_map(block_maps["high_energy"])
        block_image = cv2.merge((blue, green, red))
        height = block_image.shape[0] * self.block_size
        width = block_image.shape[1] * self.block_size
        image = cv2.resize(
            block_image,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(
            image,
            "DCT energy: low=R mid=G high=B",
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def _create_boundary_visualization(self, crop, blockiness_maps):
        base = np.clip(crop, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        for index, score in enumerate(
            blockiness_maps["vertical_line_scores"],
            start=1,
        ):
            x = index * self.block_size
            color = self._score_color(score)
            cv2.line(image, (x, 0), (x, height - 1), color, 1)
        for index, score in enumerate(
            blockiness_maps["horizontal_line_scores"],
            start=1,
        ):
            y = index * self.block_size
            color = self._score_color(score)
            cv2.line(image, (0, y), (width - 1, y), color, 1)
        return image

    def _create_blockiness_heatmap(self, block_heatmap, crop_shape):
        values = np.clip(block_heatmap, 0, 100).astype(np.uint8)
        heatmap = cv2.applyColorMap(
            np.rint(values * 2.55).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        return cv2.resize(
            heatmap,
            (crop_shape[1], crop_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def _normalize_log_map(self, values):
        logged = np.log1p(np.asarray(values, dtype=np.float64))
        minimum = float(np.min(logged))
        maximum = float(np.max(logged))
        if maximum - minimum <= 1e-12:
            return np.zeros(logged.shape, dtype=np.uint8)
        return np.rint(
            255.0 * (logged - minimum) / (maximum - minimum)
        ).astype(np.uint8)

    def _score_color(self, score):
        normalized = float(np.clip(score / 100.0, 0.0, 1.0))
        return (
            0,
            int(round(255.0 * (1.0 - normalized))),
            int(round(255.0 * normalized)),
        )

    def _load_calibration(self):
        path = config.MODEL_FREE_CALIBRATION_FILE_PATH
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as calibration_file:
                document = json.load(calibration_file)
            section = None
            for key in (
                "dct_block_analysis",
                "dct_block_compression",
                "dct_block",
            ):
                candidate = document.get(key)
                if isinstance(candidate, dict):
                    section = candidate
                    break
            if section is None:
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
            print("DCT/block calibration could not be loaded: " + str(error))
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

    def _handle_region_change(self, face_box):
        current = self._region_tuple(face_box)
        if current is None:
            return
        if self.previous_region is not None:
            overlap = self._intersection_over_union(
                self.previous_region,
                current,
            )
            if overlap < config.EXPERIMENTAL_DCT_REGION_IOU_RESET_THRESHOLD:
                self._reset_temporal_state()
        self.previous_region = current

    def _register_unavailable(self, reason):
        self.invalid_streak += 1
        self._clear_debug_outputs()
        if self.invalid_streak >= config.EXPERIMENTAL_DCT_INVALID_RESET_FRAMES:
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
                    "calibrated" if self.calibration else "experimental-uncalibrated"
                ),
                "encoded_jpeg_bytes_available": False,
            },
            calibrated=self.calibration is not None,
        )

    def _reset_temporal_state(self):
        self.score_history.clear()
        for history in self.component_histories.values():
            history.clear()
        self.invalid_streak = 0

    def _clear_debug_outputs(self):
        self.latest_analysis_crop = None
        self.latest_band_energy_map = None
        self.latest_boundary_visualization = None
        self.latest_blockiness_heatmap = None
        self.latest_coefficient_report = None
        self.latest_block_maps = None
        self.latest_blockiness_maps = None

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
            raise ValueError("DCT score interval must be increasing")
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


# Daha kisa isimle import eden entegrasyonlar icin acik alias.
DCTBlockPreController = DCTBlockAnalysisPreController
DCTBlockCompressionPreController = DCTBlockAnalysisPreController
