"""Module 6: model-free high-pass residual yuz analizi."""

from collections import deque
import json
import math

import cv2
import numpy as np

import config
from model_free_analysis import ModelFreeAnalysisResult


class HighPassResidualPreController:
    """Ince olcekli signed residual yapisini aciklanabilir sekilde olcer.

    Bu modul Noiseprint, PRNU modeli, CNN veya pretrained ag kullanmaz. Guclu
    ya da zayif residual enerji tek basina fraud karari degildir; kamera noise,
    sharpening, denoising ve video compression ayni feature'lari etkileyebilir.
    """

    MODULE_NAME = "Residual"

    def __init__(self):
        history_size = config.EXPERIMENTAL_RESIDUAL_HISTORY_SIZE
        self.score_history = deque(maxlen=history_size)
        self.direction_history = deque(maxlen=history_size)
        self.component_histories = {
            name: deque(maxlen=history_size)
            for name in config.EXPERIMENTAL_RESIDUAL_COMPONENT_WEIGHTS
        }
        self.invalid_streak = 0
        self.previous_region = None
        self.calibration = self._load_calibration()

        self.latest_analysis_crop = None
        self.latest_gaussian_residual = None
        self.latest_laplacian_response = None
        self.latest_gradient_magnitude = None
        self.latest_patch_energy_values = None
        self.latest_debug_images = None
        self.latest_feature_report = None

    def analyze(self, context):
        self._handle_region_change(context.face_bounding_box)
        invalid_reason = self._context_invalid_reason(context)
        if invalid_reason is not None:
            return self._register_unavailable(invalid_reason)

        crop = np.ascontiguousarray(
            context.standardized_aligned_face_crop,
            dtype=np.float32,
        )
        spatial_weights = self._create_spatial_weights(crop.shape)
        quality_features = self._quality_features(
            context,
            crop,
            spatial_weights,
        )
        if quality_features["clipped_pixel_ratio"] >= (
            config.RESIDUAL_UNAVAILABLE_CLIPPING_RATIO
        ):
            return self._register_unavailable(
                "severe intensity clipping makes residual analysis unreliable",
                raw_features=quality_features,
            )

        residuals = self._generate_residuals(crop)
        features = dict(quality_features)
        features.update(
            self._extract_global_features(residuals, spatial_weights)
        )
        local_features, patch_energy_values = self._analyze_local_residual(
            residuals,
            spatial_weights,
            crop.shape,
        )
        features.update(local_features)

        gaussian_score, gaussian_deviations, gaussian_direction = (
            self._score_feature_profiles(
                features,
                config.EXPERIMENTAL_GAUSSIAN_RESIDUAL_FEATURE_PROFILES,
            )
        )
        laplacian_score, laplacian_deviations, laplacian_direction = (
            self._score_feature_profiles(
                features,
                config.EXPERIMENTAL_LAPLACIAN_FEATURE_PROFILES,
            )
        )
        gradient_score, gradient_deviations, gradient_direction = (
            self._score_feature_profiles(
                features,
                config.EXPERIMENTAL_GRADIENT_FEATURE_PROFILES,
            )
        )
        component_scores = {
            "gaussian_residual_score": gaussian_score,
            "laplacian_score": laplacian_score,
            "gradient_score": gradient_score,
            "local_residual_inconsistency_score": float(
                features["local_residual_inconsistency_score"]
            ),
        }
        component_directions = {
            "gaussian_residual_score": gaussian_direction,
            "laplacian_score": laplacian_direction,
            "gradient_score": gradient_direction,
        }

        scoring_mode = config.RESIDUAL_SCORING_MODE.lower()
        calibrated = (
            self.calibration is not None
            and scoring_mode in ("auto", "calibrated")
        )
        calibration_deviations = {}
        calibrated_component_scores = {}
        if calibrated:
            (
                calibrated_component_scores,
                calibrated_component_directions,
                calibration_deviations,
            ) = self._calibration_component_scores(features)
            if not calibrated_component_scores:
                calibrated = False
            else:
                blend = config.EXPERIMENTAL_RESIDUAL_CALIBRATION_BLEND_WEIGHT
                for name, calibrated_score in (
                    calibrated_component_scores.items()
                ):
                    component_scores[name] = (
                        (1.0 - blend) * component_scores[name]
                        + blend * calibrated_score
                    )
                    if name in calibrated_component_directions:
                        component_directions[name] = (
                            calibrated_component_directions[name]
                        )

        features.update(component_scores)
        final_score = float(
            np.clip(
                self._weighted_component_score(component_scores),
                0.0,
                100.0,
            )
        )
        energy_direction = self._combine_energy_directions(
            component_scores,
            component_directions,
        )
        features["calibrated_component_scores"] = (
            calibrated_component_scores
        )
        features["residual_energy_anomaly_direction"] = energy_direction
        features["residual_energy_direction_label"] = (
            self._direction_label(energy_direction)
        )
        features["final_residual_score"] = final_score

        self._update_debug_outputs(
            crop,
            residuals,
            patch_energy_values,
            features,
        )

        if scoring_mode == "calibrated" and not calibrated:
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=[
                    "Compatible residual calibration data is unavailable"
                ],
                debug_data=self._debug_data(
                    context,
                    "calibration-required",
                    self._experimental_deviations(
                        gaussian_deviations,
                        laplacian_deviations,
                        gradient_deviations,
                    ),
                    calibration_deviations,
                ),
            )
        if scoring_mode not in ("auto", "experimental", "calibrated"):
            return ModelFreeAnalysisResult.uncalibrated(
                self.MODULE_NAME,
                raw_features=features,
                raw_score=final_score,
                evidence=["Unsupported residual scoring mode"],
                debug_data={"scoring_mode": scoring_mode},
            )

        return self._stabilize_result(
            context,
            features,
            component_scores,
            final_score,
            energy_direction,
            calibrated,
            self._experimental_deviations(
                gaussian_deviations,
                laplacian_deviations,
                gradient_deviations,
            ),
            calibration_deviations,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None
        self._clear_debug_outputs()

    def get_debug_images(self):
        if self.latest_analysis_crop is None:
            return None
        if self.latest_debug_images is None:
            self.latest_debug_images = {
                "gaussian_residual": self._visualize_signed(
                    self.latest_gaussian_residual
                ),
                "laplacian": self._visualize_signed(
                    self.latest_laplacian_response
                ),
                "gradient_magnitude": self._visualize_magnitude(
                    self.latest_gradient_magnitude
                ),
                "patch_residual_energy_map": self._visualize_patch_map(
                    self.latest_patch_energy_values
                ),
            }
        return {
            name: image.copy()
            for name, image in self.latest_debug_images.items()
        }

    def get_feature_report(self):
        if self.latest_feature_report is None:
            return None
        return dict(self.latest_feature_report)

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
        if min(context.face_dimensions) < config.RESIDUAL_MINIMUM_SOURCE_SIDE:
            return "source face resolution is too low for residual analysis"
        if float(np.std(crop)) <= 1e-6:
            return "aligned face crop has insufficient intensity variation"
        return self._filter_configuration_error()

    def _filter_configuration_error(self):
        kernel_values = (
            config.RESIDUAL_GAUSSIAN_KERNEL_SIZE,
            config.RESIDUAL_LAPLACIAN_KERNEL_SIZE,
            config.RESIDUAL_SOBEL_KERNEL_SIZE,
        )
        if any(value <= 0 or value % 2 == 0 for value in kernel_values):
            return "residual filter kernel sizes must be positive odd numbers"
        if config.RESIDUAL_GAUSSIAN_SIGMA <= 0:
            return "Gaussian residual sigma must be positive"
        if config.RESIDUAL_LOCAL_PATCH_SIZE < 4:
            return "residual local patch size is invalid"
        return None

    def _generate_residuals(self, crop):
        blurred = cv2.GaussianBlur(
            crop,
            (
                config.RESIDUAL_GAUSSIAN_KERNEL_SIZE,
                config.RESIDUAL_GAUSSIAN_KERNEL_SIZE,
            ),
            config.RESIDUAL_GAUSSIAN_SIGMA,
            borderType=cv2.BORDER_REFLECT101,
        )
        gaussian_residual = crop - blurred
        laplacian = cv2.Laplacian(
            crop,
            cv2.CV_32F,
            ksize=config.RESIDUAL_LAPLACIAN_KERNEL_SIZE,
            borderType=cv2.BORDER_REFLECT101,
        )
        sobel_x = cv2.Sobel(
            crop,
            cv2.CV_32F,
            1,
            0,
            ksize=config.RESIDUAL_SOBEL_KERNEL_SIZE,
            borderType=cv2.BORDER_REFLECT101,
        )
        sobel_y = cv2.Sobel(
            crop,
            cv2.CV_32F,
            0,
            1,
            ksize=config.RESIDUAL_SOBEL_KERNEL_SIZE,
            borderType=cv2.BORDER_REFLECT101,
        )
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        return {
            "gaussian": gaussian_residual.astype(np.float32),
            "laplacian": laplacian.astype(np.float32),
            "sobel_x": sobel_x.astype(np.float32),
            "sobel_y": sobel_y.astype(np.float32),
            "gradient": gradient_magnitude.astype(np.float32),
        }

    def _create_spatial_weights(self, shape):
        height, width = shape
        y_coordinates, x_coordinates = np.indices(shape, dtype=np.float32)
        normalized_y = y_coordinates / max(float(height - 1), 1.0)
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius_x = max(
            1.0,
            width * config.RESIDUAL_MASK_HORIZONTAL_RADIUS_RATIO,
        )
        radius_y = max(
            1.0,
            height * config.RESIDUAL_MASK_VERTICAL_RADIUS_RATIO,
        )
        radius_squared = (
            ((x_coordinates - center_x) / radius_x) ** 2
            + ((y_coordinates - center_y) / radius_y) ** 2
        )
        ellipse_weight = np.clip(1.0 - radius_squared, 0.0, 1.0)

        top_end = config.RESIDUAL_MASK_TOP_RAMP_END_RATIO
        top_weight = np.clip(normalized_y / max(top_end, 1e-6), 0.0, 1.0)
        bottom_start = config.RESIDUAL_MASK_BOTTOM_RAMP_START_RATIO
        bottom_weight = np.clip(
            (1.0 - normalized_y) / max(1.0 - bottom_start, 1e-6),
            0.0,
            1.0,
        )
        weights = ellipse_weight * top_weight * bottom_weight

        eye_center = config.RESIDUAL_MASK_EYE_BAND_CENTER_RATIO
        eye_half_height = config.RESIDUAL_MASK_EYE_BAND_HALF_HEIGHT_RATIO
        eye_band = np.abs(normalized_y - eye_center) <= eye_half_height
        weights[eye_band] *= config.RESIDUAL_MASK_EYE_BAND_WEIGHT
        return weights.astype(np.float32)

    def _quality_features(self, context, crop, spatial_weights):
        valid = spatial_weights >= config.RESIDUAL_MINIMUM_MASK_WEIGHT
        values = crop[valid]
        clipped = (
            (values <= config.RESIDUAL_CLIPPED_PIXEL_LOW)
            | (values >= config.RESIDUAL_CLIPPED_PIXEL_HIGH)
        )
        source_side = min(context.face_dimensions)
        return {
            "analysis_width": int(crop.shape[1]),
            "analysis_height": int(crop.shape[0]),
            "source_face_width": int(context.face_dimensions[0]),
            "source_face_height": int(context.face_dimensions[1]),
            "standardization_upscale_ratio": (
                min(crop.shape) / max(float(source_side), 1.0)
            ),
            "clipped_pixel_ratio": float(np.mean(clipped)),
            "residual_mask_effective_ratio": float(np.mean(valid)),
            "residual_mask_mean_weight": float(np.mean(spatial_weights)),
            "input_intensity_mean": float(np.mean(values)),
            "input_intensity_std": float(np.std(values)),
        }

    def _extract_global_features(self, residuals, spatial_weights):
        gaussian = self._signed_statistics(
            residuals["gaussian"],
            spatial_weights,
            "gaussian_residual_",
        )
        laplacian = self._signed_statistics(
            residuals["laplacian"],
            spatial_weights,
            "laplacian_",
        )
        gradient_values, gradient_weights = self._weighted_samples(
            residuals["gradient"],
            spatial_weights,
        )
        sobel_x_values, sobel_weights = self._weighted_samples(
            residuals["sobel_x"],
            spatial_weights,
        )
        sobel_y_values, _ = self._weighted_samples(
            residuals["sobel_y"],
            spatial_weights,
        )
        gradient_energy = self._weighted_mean(
            gradient_values ** 2,
            gradient_weights,
        )
        horizontal_energy = self._weighted_mean(
            sobel_x_values ** 2,
            sobel_weights,
        )
        vertical_energy = self._weighted_mean(
            sobel_y_values ** 2,
            sobel_weights,
        )
        edge_density = self._weighted_mean(
            (
                gradient_values
                >= config.RESIDUAL_EDGE_MAGNITUDE_THRESHOLD
            ).astype(np.float64),
            gradient_weights,
        )
        features = {}
        features.update(gaussian)
        features.update(laplacian)
        # Gaussian high-pass is the baseline residual. Keep explicit generic
        # aliases so downstream reports do not need to infer that mapping.
        features.update(
            {
                "residual_variance": gaussian[
                    "gaussian_residual_variance"
                ],
                "residual_mean_absolute_deviation": gaussian[
                    "gaussian_residual_mean_absolute_deviation"
                ],
                "residual_rms_energy": gaussian[
                    "gaussian_residual_rms_energy"
                ],
                "residual_entropy": gaussian[
                    "gaussian_residual_entropy"
                ],
                "residual_kurtosis": gaussian[
                    "gaussian_residual_kurtosis"
                ],
                "positive_negative_residual_balance": gaussian[
                    "gaussian_residual_positive_negative_balance"
                ],
            }
        )
        features.update(
            {
                "gradient_mean_magnitude": self._weighted_mean(
                    gradient_values,
                    gradient_weights,
                ),
                "gradient_variance": self._weighted_variance(
                    gradient_values,
                    gradient_weights,
                ),
                "gradient_energy": gradient_energy,
                "high_frequency_edge_density": edge_density,
                "sobel_x_energy": horizontal_energy,
                "sobel_y_energy": vertical_energy,
                "gradient_directional_energy_balance": (
                    (horizontal_energy - vertical_energy)
                    / max(horizontal_energy + vertical_energy, 1e-12)
                ),
            }
        )
        return features

    def _signed_statistics(self, array, spatial_weights, prefix):
        values, weights = self._weighted_samples(array, spatial_weights)
        mean = self._weighted_mean(values, weights)
        variance = self._weighted_variance(values, weights, mean)
        mean_absolute_deviation = self._weighted_mean(
            np.abs(values - mean),
            weights,
        )
        rms = math.sqrt(
            max(self._weighted_mean(values ** 2, weights), 0.0)
        )
        centered = values - mean
        fourth_moment = self._weighted_mean(centered ** 4, weights)
        kurtosis = (
            fourth_moment / (variance ** 2)
            if variance > 1e-12
            else 0.0
        )
        positive_weight = float(np.sum(weights[values > 1e-6]))
        negative_weight = float(np.sum(weights[values < -1e-6]))
        signed_weight = positive_weight + negative_weight
        positive_fraction = positive_weight / max(signed_weight, 1e-12)
        negative_fraction = negative_weight / max(signed_weight, 1e-12)
        positive_energy = self._weighted_mean(
            np.where(values > 0, values ** 2, 0.0),
            weights,
        )
        negative_energy = self._weighted_mean(
            np.where(values < 0, values ** 2, 0.0),
            weights,
        )
        return {
            prefix + "mean": mean,
            prefix + "variance": variance,
            prefix + "mean_absolute_deviation": mean_absolute_deviation,
            prefix + "rms_energy": rms,
            prefix + "entropy": self._weighted_histogram_entropy(
                values,
                weights,
            ),
            prefix + "kurtosis": float(kurtosis),
            prefix + "positive_fraction": positive_fraction,
            prefix + "negative_fraction": negative_fraction,
            prefix + "positive_negative_balance": (
                positive_fraction - negative_fraction
            ),
            prefix + "positive_to_negative_energy_ratio": (
                positive_energy / max(negative_energy, 1e-12)
            ),
        }

    def _weighted_samples(self, array, spatial_weights):
        valid = spatial_weights >= config.RESIDUAL_MINIMUM_MASK_WEIGHT
        return (
            array[valid].astype(np.float64),
            spatial_weights[valid].astype(np.float64),
        )

    def _weighted_mean(self, values, weights):
        return float(
            np.sum(values * weights) / max(float(np.sum(weights)), 1e-12)
        )

    def _weighted_variance(self, values, weights, mean=None):
        if mean is None:
            mean = self._weighted_mean(values, weights)
        return self._weighted_mean((values - mean) ** 2, weights)

    def _weighted_histogram_entropy(self, values, weights):
        limit = float(np.percentile(np.abs(values), 99.5))
        if limit <= 1e-12:
            return 0.0
        histogram, _ = np.histogram(
            values,
            bins=64,
            range=(-limit, limit),
            weights=weights,
        )
        total = float(np.sum(histogram))
        if total <= 1e-12:
            return 0.0
        probabilities = histogram / total
        nonzero = probabilities > 0
        entropy = -float(
            np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
        )
        return entropy / math.log2(histogram.size)

    def _analyze_local_residual(
        self,
        residuals,
        spatial_weights,
        face_shape,
    ):
        patch_size = int(config.RESIDUAL_LOCAL_PATCH_SIZE)
        patch_rows = face_shape[0] // patch_size
        patch_columns = face_shape[1] // patch_size
        records = []
        record_grid = {}
        for row in range(patch_rows):
            top = row * patch_size
            for column in range(patch_columns):
                left = column * patch_size
                patch_weights = spatial_weights[
                    top : top + patch_size,
                    left : left + patch_size,
                ].astype(np.float64)
                coverage = float(
                    np.mean(
                        patch_weights
                        >= config.RESIDUAL_MINIMUM_MASK_WEIGHT
                    )
                )
                if coverage < config.RESIDUAL_LOCAL_MINIMUM_MASK_COVERAGE:
                    continue
                gaussian_patch = residuals["gaussian"][
                    top : top + patch_size,
                    left : left + patch_size,
                ].astype(np.float64)
                laplacian_patch = residuals["laplacian"][
                    top : top + patch_size,
                    left : left + patch_size,
                ].astype(np.float64)
                gradient_patch = residuals["gradient"][
                    top : top + patch_size,
                    left : left + patch_size,
                ].astype(np.float64)
                valid = patch_weights >= config.RESIDUAL_MINIMUM_MASK_WEIGHT
                weights = patch_weights[valid]
                gaussian_values = gaussian_patch[valid]
                laplacian_values = laplacian_patch[valid]
                gradient_values = gradient_patch[valid]
                gaussian_energy = self._weighted_mean(
                    gaussian_values ** 2,
                    weights,
                )
                laplacian_energy = self._weighted_mean(
                    laplacian_values ** 2,
                    weights,
                )
                gradient_energy = self._weighted_mean(
                    gradient_values ** 2,
                    weights,
                )
                positive = float(np.sum(weights[gaussian_values > 1e-6]))
                negative = float(np.sum(weights[gaussian_values < -1e-6]))
                balance = (positive - negative) / max(
                    positive + negative,
                    1e-12,
                )
                record = {
                    "row": row,
                    "column": column,
                    "top": top,
                    "left": left,
                    "patch_size": patch_size,
                    "gaussian_energy": gaussian_energy,
                    "descriptor": np.asarray(
                        [
                            math.log1p(gaussian_energy),
                            math.log1p(laplacian_energy),
                            math.log1p(gradient_energy),
                            balance,
                        ],
                        dtype=np.float64,
                    ),
                }
                records.append(record)
                record_grid[(row, column)] = record

        if len(records) < 4:
            return self._empty_local_features(len(records)), np.zeros(
                face_shape,
                dtype=np.float32,
            )

        descriptors = np.stack(
            [record["descriptor"] for record in records],
            axis=0,
        )
        center = np.median(descriptors, axis=0)
        scale = np.maximum(
            np.median(np.abs(descriptors - center), axis=0) * 1.4826,
            np.asarray([0.35, 0.45, 0.45, 0.15]),
        )
        normalized = (descriptors - center) / scale
        distances = np.sqrt(
            np.mean(np.minimum(normalized ** 2, 400.0), axis=1)
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
                    record["descriptor"][0]
                    - neighbor["descriptor"][0]
                ) / scale[0]
                neighbor_differences.append(difference)
                record["neighbor_difference"] = max(
                    record["neighbor_difference"],
                    difference,
                )
                neighbor["neighbor_difference"] = max(
                    neighbor.get("neighbor_difference", 0.0),
                    difference,
                )

        patch_scores = []
        for index, record in enumerate(records):
            distance_score = self._linear_score(
                distances[index],
                config.EXPERIMENTAL_RESIDUAL_LOCAL_DISTANCE_START,
                config.EXPERIMENTAL_RESIDUAL_LOCAL_DISTANCE_FULL,
            )
            neighbor_score = self._linear_score(
                record["neighbor_difference"],
                config.EXPERIMENTAL_RESIDUAL_NEIGHBOR_DIFFERENCE_START,
                config.EXPERIMENTAL_RESIDUAL_NEIGHBOR_DIFFERENCE_FULL,
            )
            score = float(
                np.clip(
                    0.75 * distance_score + 0.25 * neighbor_score,
                    0.0,
                    100.0,
                )
            )
            record["anomaly_score"] = score
            patch_scores.append(score)

        patch_scores = np.asarray(patch_scores, dtype=np.float64)
        outlier_ratio = float(
            np.mean(
                distances
                >= config.EXPERIMENTAL_RESIDUAL_LOCAL_OUTLIER_DISTANCE
            )
        )
        outlier_score = float(
            np.clip(
                100.0
                * outlier_ratio
                / config.EXPERIMENTAL_RESIDUAL_LOCAL_OUTLIER_RATIO_FULL,
                0.0,
                100.0,
            )
        )
        local_score = float(
            np.clip(
                0.65 * np.percentile(patch_scores, 95)
                + 0.35 * outlier_score,
                0.0,
                100.0,
            )
        )
        gaussian_energies = np.asarray(
            [record["gaussian_energy"] for record in records],
            dtype=np.float64,
        )
        neighbor_array = np.asarray(
            neighbor_differences or [0.0],
            dtype=np.float64,
        )
        patch_energy_map = np.zeros(face_shape, dtype=np.float32)
        for record in records:
            top = record["top"]
            left = record["left"]
            patch_energy_map[
                top : top + patch_size,
                left : left + patch_size,
            ] = record["gaussian_energy"]
        patch_energy_map[
            spatial_weights < config.RESIDUAL_MINIMUM_MASK_WEIGHT
        ] = 0.0

        features = {
            "local_residual_patch_count": len(records),
            "patch_residual_energy_mean": float(
                np.mean(gaussian_energies)
            ),
            "patch_residual_energy_std": float(np.std(gaussian_energies)),
            "patch_residual_energy_variation": float(
                np.std(gaussian_energies)
                / max(float(np.mean(gaussian_energies)), 1e-12)
            ),
            "patch_residual_energy_p95": float(
                np.percentile(gaussian_energies, 95)
            ),
            "local_residual_robust_distance_mean": float(
                np.mean(distances)
            ),
            "local_residual_robust_distance_p95": float(
                np.percentile(distances, 95)
            ),
            "local_residual_outlier_patch_ratio": outlier_ratio,
            "neighbor_residual_difference_mean": float(
                np.mean(neighbor_array)
            ),
            "neighbor_residual_difference_p95": float(
                np.percentile(neighbor_array, 95)
            ),
            "local_residual_consistency": float(
                1.0 / (1.0 + np.mean(distances))
            ),
            "local_residual_inconsistency_score": local_score,
        }
        return features, patch_energy_map

    def _empty_local_features(self, patch_count):
        return {
            "local_residual_patch_count": patch_count,
            "patch_residual_energy_mean": 0.0,
            "patch_residual_energy_std": 0.0,
            "patch_residual_energy_variation": 0.0,
            "patch_residual_energy_p95": 0.0,
            "local_residual_robust_distance_mean": 0.0,
            "local_residual_robust_distance_p95": 0.0,
            "local_residual_outlier_patch_ratio": 0.0,
            "neighbor_residual_difference_mean": 0.0,
            "neighbor_residual_difference_p95": 0.0,
            "local_residual_consistency": 0.0,
            "local_residual_inconsistency_score": 0.0,
        }

    def _score_feature_profiles(self, features, profiles):
        weighted_deviation = 0.0
        signed_deviation = 0.0
        total_weight = 0.0
        deviations = {}
        for name, profile in profiles.items():
            if name not in features:
                continue
            value = features[name]
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            deviation, direction = self._range_deviation_with_direction(
                float(value),
                float(profile["minimum"]),
                float(profile["maximum"]),
                float(profile["deviation_scale"]),
            )
            weight = float(profile.get("weight", 1.0))
            deviations[name] = {
                "deviation": deviation,
                "direction": direction,
            }
            weighted_deviation += deviation * weight
            signed_deviation += deviation * direction * weight
            total_weight += weight
        if total_weight <= 0:
            return None, {}, 0.0
        score = float(
            np.clip(100.0 * weighted_deviation / total_weight, 0.0, 100.0)
        )
        direction = signed_deviation / max(weighted_deviation, 1e-12)
        return score, deviations, float(np.clip(direction, -1.0, 1.0))

    def _calibration_component_scores(self, features):
        groups = {
            "gaussian_residual_score": {},
            "laplacian_score": {},
            "gradient_score": {},
            "local_residual_inconsistency_score": {},
        }
        for name, profile in self.calibration["feature_profiles"].items():
            component = self._calibration_component_name(name)
            if component is not None:
                groups[component][name] = profile

        scores = {}
        directions = {}
        deviations = {}
        for component, profiles in groups.items():
            if not profiles:
                continue
            score, group_deviations, direction = (
                self._score_feature_profiles(features, profiles)
            )
            if score is None:
                continue
            scores[component] = score
            directions[component] = direction
            deviations.update(group_deviations)
        return scores, directions, deviations

    def _calibration_component_name(self, feature_name):
        if feature_name.startswith("gaussian_"):
            return "gaussian_residual_score"
        if feature_name.startswith("laplacian_"):
            return "laplacian_score"
        if feature_name.startswith(("gradient_", "sobel_", "high_frequency_")):
            return "gradient_score"
        if feature_name.startswith(("local_", "patch_", "neighbor_")):
            return "local_residual_inconsistency_score"
        return None

    def _weighted_component_score(self, component_scores):
        weighted_sum = 0.0
        total_weight = 0.0
        for name, weight in (
            config.EXPERIMENTAL_RESIDUAL_COMPONENT_WEIGHTS.items()
        ):
            weighted_sum += float(component_scores[name]) * float(weight)
            total_weight += float(weight)
        if total_weight <= 0:
            raise ValueError("residual component weights must be positive")
        return weighted_sum / total_weight

    def _combine_energy_directions(
        self,
        component_scores,
        component_directions,
    ):
        numerator = 0.0
        denominator = 0.0
        for name, direction in component_directions.items():
            score = float(component_scores[name])
            numerator += score * float(direction)
            denominator += score
        if denominator <= 1e-12:
            return 0.0
        return float(np.clip(numerator / denominator, -1.0, 1.0))

    def _direction_label(self, direction):
        if direction >= 0.15:
            return "high"
        if direction <= -0.15:
            return "low"
        return "balanced"

    def _stabilize_result(
        self,
        context,
        features,
        component_scores,
        final_score,
        energy_direction,
        calibrated,
        experimental_deviations,
        calibration_deviations,
    ):
        self.invalid_streak = 0
        self.score_history.append(final_score)
        self.direction_history.append(energy_direction)
        for name, score in component_scores.items():
            self.component_histories[name].append(score)

        stable_score = float(np.median(self.score_history))
        stable_direction = float(np.median(self.direction_history))
        stable_components = {
            name: float(np.median(history))
            for name, history in self.component_histories.items()
        }
        history_length = len(self.score_history)
        history_ready = (
            history_length >= config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY
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
            stable_score >= config.EXPERIMENTAL_RESIDUAL_SUSPICIOUS_SCORE
            and (
                calibrated
                or sum(score >= 50.0 for score in stable_components.values())
                >= 2
            )
        ):
            status = "Suspicious fine-detail evidence"
            warnings.append(
                "Multiple residual signals are elevated; this is supporting evidence only"
            )
        elif stable_components["local_residual_inconsistency_score"] >= (
            config.EXPERIMENTAL_RESIDUAL_LOCAL_STATUS_SCORE
        ):
            status = "Local residual inconsistency"
            warnings.append(
                "Repeated inner-face residual patches differ from neighboring regions"
            )
        elif max(
            stable_components["gaussian_residual_score"],
            stable_components["laplacian_score"],
            stable_components["gradient_score"],
        ) >= config.EXPERIMENTAL_RESIDUAL_ENERGY_STATUS_SCORE:
            if stable_direction <= -0.15:
                status = "Abnormally smooth residual"
                warnings.append(
                    "Fine-scale residual energy is unusually low for the active profile"
                )
            elif stable_direction >= 0.15:
                status = "Excessive high-frequency residual"
                warnings.append(
                    "Fine-scale residual energy is unusually high; noise or sharpening remain possible"
                )
            else:
                status = "Suspicious fine-detail evidence"
        else:
            status = "Normal residual structure"

        evidence = self._create_evidence(
            stable_components,
            stable_direction,
            calibrated,
            experimental_deviations,
            calibration_deviations,
        )
        temporal_confidence = min(
            1.0,
            history_length / config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY,
        )
        confidence_limit = (
            1.0
            if calibrated
            else config.EXPERIMENTAL_RESIDUAL_MAXIMUM_CONFIDENCE
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
                experimental_deviations,
                calibration_deviations,
                history_length,
                stable_components,
                stable_direction,
                uncertainty_reasons,
            ),
            calibrated=calibrated,
        )

    def _quality_uncertainty_reasons(self, context, features):
        reasons = []
        if context.blur_value is not None and context.blur_value < (
            config.RESIDUAL_UNCERTAIN_BLUR_SCORE
        ):
            reasons.append("blur suppresses fine-scale residuals")
        if context.brightness_value is not None and context.brightness_value < (
            config.RESIDUAL_UNCERTAIN_MINIMUM_BRIGHTNESS
        ):
            reasons.append("low light can increase ISO noise")
        if context.brightness_value is not None and context.brightness_value > (
            config.RESIDUAL_UNCERTAIN_MAXIMUM_BRIGHTNESS
        ):
            reasons.append("high exposure affects residual statistics")
        if min(context.face_dimensions) < config.RESIDUAL_UNCERTAIN_SOURCE_SIDE:
            reasons.append("source face resolution is marginal")
        if features["standardization_upscale_ratio"] > (
            config.RESIDUAL_UNCERTAIN_UPSCALE_RATIO
        ):
            reasons.append("standardization requires substantial upscaling")
        if features["clipped_pixel_ratio"] >= (
            config.RESIDUAL_UNCERTAIN_CLIPPING_RATIO
        ):
            reasons.append("intensity clipping affects high-pass responses")
        if (
            context.brightness_value is not None
            and context.brightness_value < 90.0
            and features["gaussian_residual_rms_energy"]
            >= config.RESIDUAL_LOW_LIGHT_NOISE_RMS
        ):
            reasons.append(
                "high residual under low light may be ISO or transmission noise"
            )
        return reasons

    def _create_evidence(
        self,
        stable_components,
        stable_direction,
        calibrated,
        experimental_deviations,
        calibration_deviations,
    ):
        threshold = config.EXPERIMENTAL_RESIDUAL_EVIDENCE_SCORE
        labels = {
            "gaussian_residual_score": "Gaussian residual distribution anomaly",
            "laplacian_score": "Laplacian response anomaly",
            "gradient_score": "Gradient/edge-density anomaly",
            "local_residual_inconsistency_score": (
                "Repeated local residual inconsistency"
            ),
        }
        evidence = [
            labels[name]
            for name, score in stable_components.items()
            if score >= threshold
        ]
        if not evidence:
            evidence.append("No strong fine-detail residual anomaly")
        if abs(stable_direction) >= 0.15:
            evidence.append(
                "Residual anomaly direction: "
                + self._direction_label(stable_direction)
            )
        if experimental_deviations:
            evidence.append("Raw two-sided residual deviations retained")
        if calibrated and calibration_deviations:
            evidence.append("Bona-fide residual calibration profiles applied")
        else:
            evidence.append("Experimental score; calibration unavailable")
        evidence.append(
            "Strong residual energy is not automatically interpreted as fraud"
        )
        return evidence

    def _debug_data(
        self,
        context,
        scoring_mode,
        experimental_deviations,
        calibration_deviations,
        history_length=0,
        stable_components=None,
        stable_direction=0.0,
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
            "stabilized_energy_direction": stable_direction,
            "experimental_feature_deviations": experimental_deviations,
            "calibration_feature_deviations": calibration_deviations,
            "uncertainty_reasons": uncertainty_reasons or [],
            "filters": {
                "gaussian_kernel_size": config.RESIDUAL_GAUSSIAN_KERNEL_SIZE,
                "gaussian_sigma": config.RESIDUAL_GAUSSIAN_SIGMA,
                "laplacian_kernel_size": config.RESIDUAL_LAPLACIAN_KERNEL_SIZE,
                "sobel_kernel_size": config.RESIDUAL_SOBEL_KERNEL_SIZE,
                "edge_magnitude_threshold": (
                    config.RESIDUAL_EDGE_MAGNITUDE_THRESHOLD
                ),
            },
            "input_numeric_scale": (
                "single standardized aligned float32 luminance crop; no repeated normalization"
            ),
            "alignment_applied": context.alignment_applied,
            "pose_alignment_valid": context.pose_alignment_valid,
            "supporting_evidence_only": True,
            "excluded_methods": [
                "Noiseprint",
                "neural PRNU",
                "CNN",
                "F3-Net",
                "pretrained networks",
            ],
            "confounding_factors": [
                "blur",
                "low light and ISO noise",
                "camera sharpening",
                "DroidCam/video compression",
                "denoising",
                "interpolation",
                "sensor noise",
            ],
        }

    def _update_debug_outputs(
        self,
        crop,
        residuals,
        patch_energy_values,
        features,
    ):
        self.latest_analysis_crop = crop.copy()
        self.latest_gaussian_residual = residuals["gaussian"].copy()
        self.latest_laplacian_response = residuals["laplacian"].copy()
        self.latest_gradient_magnitude = residuals["gradient"].copy()
        self.latest_patch_energy_values = patch_energy_values.copy()
        self.latest_debug_images = None
        self.latest_feature_report = dict(features)

    def _visualize_signed(self, values):
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.percentile(np.abs(values), 99.0))
        if scale <= 1e-12:
            return np.full(values.shape, 127, dtype=np.uint8)
        normalized = 127.5 + 127.5 * np.clip(values / scale, -1.0, 1.0)
        return np.rint(normalized).astype(np.uint8)

    def _visualize_magnitude(self, values):
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.percentile(values, 99.0))
        if scale <= 1e-12:
            return np.zeros(values.shape, dtype=np.uint8)
        return np.rint(
            255.0 * np.clip(values / scale, 0.0, 1.0)
        ).astype(np.uint8)

    def _visualize_patch_map(self, values):
        log_energy = np.log1p(
            np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        )
        valid = log_energy > 0.0
        normalized = np.zeros(log_energy.shape, dtype=np.uint8)
        if np.any(valid):
            lower = float(np.percentile(log_energy[valid], 5.0))
            upper = float(np.percentile(log_energy[valid], 95.0))
            if upper > lower + 1e-12:
                scaled = np.clip(
                    (log_energy - lower) / (upper - lower),
                    0.0,
                    1.0,
                )
                normalized[valid] = np.rint(
                    255.0 * scaled[valid]
                ).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        heatmap[~valid] = 0
        return heatmap

    def _load_calibration(self):
        path = config.MODEL_FREE_CALIBRATION_FILE_PATH
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as calibration_file:
                document = json.load(calibration_file)
            section = None
            for key in (
                "high_pass_residual_analysis",
                "high_pass_residual",
                "residual_analysis",
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
            print("Residual calibration could not be loaded: " + str(error))
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
                z_limit = float(profile.get("normal_z_limit", 2.0))
                minimum = mean - z_limit * standard_deviation
                maximum = mean + z_limit * standard_deviation
                scale = max(
                    float(profile.get("deviation_scale", 2.0))
                    * standard_deviation,
                    1e-9,
                )
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

    def _range_deviation_with_direction(
        self,
        value,
        minimum,
        maximum,
        scale,
    ):
        if not math.isfinite(value):
            return 1.0, 0.0
        if scale <= 0:
            raise ValueError("residual deviation scale must be positive")
        if value < minimum:
            return float(np.clip((minimum - value) / scale, 0.0, 1.0)), -1.0
        if value > maximum:
            return float(np.clip((value - maximum) / scale, 0.0, 1.0)), 1.0
        return 0.0, 0.0

    def _experimental_deviations(
        self,
        gaussian_deviations,
        laplacian_deviations,
        gradient_deviations,
    ):
        return {
            "gaussian": gaussian_deviations,
            "laplacian": laplacian_deviations,
            "gradient": gradient_deviations,
        }

    def _register_unavailable(self, reason, raw_features=None):
        self.invalid_streak += 1
        self._clear_debug_outputs()
        if self.invalid_streak >= (
            config.EXPERIMENTAL_RESIDUAL_INVALID_RESET_FRAMES
        ):
            self._reset_temporal_state()
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=False,
            raw_features=dict(raw_features or {}),
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
                "supporting_evidence_only": True,
            },
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
                config.EXPERIMENTAL_RESIDUAL_REGION_IOU_RESET_THRESHOLD
            ):
                self._reset_temporal_state()
        self.previous_region = current

    def _reset_temporal_state(self):
        self.score_history.clear()
        self.direction_history.clear()
        for history in self.component_histories.values():
            history.clear()
        self.invalid_streak = 0

    def _clear_debug_outputs(self):
        self.latest_analysis_crop = None
        self.latest_gaussian_residual = None
        self.latest_laplacian_response = None
        self.latest_gradient_magnitude = None
        self.latest_patch_energy_values = None
        self.latest_debug_images = None
        self.latest_feature_report = None

    def _linear_score(self, value, start, full):
        if full <= start:
            raise ValueError("residual score interval must be increasing")
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


ResidualPreController = HighPassResidualPreController
