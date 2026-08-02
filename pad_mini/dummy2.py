"""Six self-contained mathematical deterministic image methods.

This module intentionally does not import any implementation file from the
original prototype.  It can be moved into the larger project together with its
``contracts`` package.  Runtime dependencies are NumPy and OpenCV only.

Contract notes
--------------
``DeterministicMethodInput`` contains an image but no face box.  Therefore the
complete image is interpreted as the already selected face/analysis ROI.
Three-channel images are interpreted in OpenCV BGR order.  Every public score
is in [0, 1], where a larger value means stronger anomalous/suspicious signal.
The scores are deterministic engineering mappings, not attack probabilities.
"""

from typing import Any
import math

import cv2
import numpy as np

from contracts.deterministic_method import (
    DeterministicMethod,
    DeterministicMethodInput,
    DeterministicMethodResult,
)
from contracts.media import ImageArray


_EPSILON = 1e-12
_ANALYSIS_SIZE = 256
_MINIMUM_SOURCE_SIDE = 96
_MAD_TO_SIGMA = 1.4826


# These are development profiles, not universal biometric reference ranges.
_GLOBAL_FFT_PROFILES = {
    "low_frequency_energy_ratio": (0.05, 0.95, 0.20, 0.20),
    "middle_frequency_energy_ratio": (0.03, 0.80, 0.20, 0.10),
    "high_frequency_energy_ratio": (0.005, 0.65, 0.25, 0.20),
    "spectral_centroid": (0.05, 0.70, 0.20, 0.15),
    "spectral_entropy": (0.10, 0.99, 0.20, 0.15),
    "spectral_slope": (-8.0, 0.25, 3.0, 0.10),
    "high_to_low_energy_ratio": (0.005, 5.0, 3.0, 0.10),
}

_RADIAL_PROFILES = {
    "radial_spectral_slope": (-8.0, 0.50, 3.0, 0.25),
    "slope_fit_error": (0.0, 0.65, 0.50, 0.20),
    "radial_entropy": (0.15, 1.0, 0.20, 0.15),
    "dominant_radial_energy_ratio": (0.0, 0.20, 0.30, 0.20),
    "narrow_band_energy_concentration": (1.0, 8.0, 8.0, 0.20),
}

_ANGULAR_PROFILES = {
    "maximum_angular_energy": (0.0, 0.20, 0.30, 0.25),
    "angular_entropy": (0.25, 1.0, 0.25, 0.20),
    "directional_anisotropy": (0.0, 0.75, 0.25, 0.25),
    "horizontal_concentration": (0.0, 0.65, 0.35, 0.10),
    "vertical_concentration": (0.0, 0.65, 0.35, 0.10),
    "diagonal_concentration": (0.0, 0.75, 0.25, 0.10),
}

_DCT_BAND_PROFILES = {
    "low_frequency_ac_energy_ratio": (0.30, 0.995, 0.30, 0.30),
    "middle_frequency_ac_energy_ratio": (0.003, 0.55, 0.30, 0.25),
    "high_frequency_ac_energy_ratio": (0.0001, 0.30, 0.25, 0.25),
    "ac_to_dc_ratio_mean": (0.00005, 0.75, 0.50, 0.20),
}

_DCT_SPARSITY_PROFILES = {
    "near_zero_ac_coefficient_ratio": (0.0, 0.88, 0.12, 0.55),
    "coefficient_entropy_mean": (0.08, 0.98, 0.20, 0.25),
    "coefficient_kurtosis_global": (1.0, 80.0, 80.0, 0.20),
}

_WAVELET_ENERGY_PROFILES = {
    "level_1_detail_to_approximation_energy_ratio": (
        0.00001,
        0.45,
        0.40,
        0.30,
    ),
    "level_2_detail_to_approximation_energy_ratio": (
        0.00001,
        0.55,
        0.45,
        0.25,
    ),
    "global_detail_entropy_mean": (0.08, 0.99, 0.20, 0.20),
    "global_detail_sparsity_mean": (0.0, 0.92, 0.08, 0.25),
}

_GAUSSIAN_RESIDUAL_PROFILES = {
    "gaussian_residual_variance": (6.25, 625.0, 6.25, 0.24),
    "gaussian_residual_mean_absolute_deviation": (1.50, 20.0, 1.50, 0.20),
    "gaussian_residual_rms_energy": (2.50, 25.0, 2.50, 0.24),
    "gaussian_residual_entropy": (0.10, 0.99, 0.20, 0.12),
    "gaussian_residual_kurtosis": (1.0, 40.0, 40.0, 0.10),
    "gaussian_residual_positive_negative_balance": (
        -0.25,
        0.25,
        0.50,
        0.10,
    ),
}

_LAPLACIAN_PROFILES = {
    "laplacian_variance": (100.0, 8000.0, 100.0, 0.55),
    "laplacian_rms_energy": (10.0, 90.0, 10.0, 0.30),
    "laplacian_kurtosis": (1.0, 50.0, 50.0, 0.15),
}

_GRADIENT_PROFILES = {
    "gradient_energy": (100.0, 25000.0, 100.0, 0.50),
    "gradient_mean_magnitude": (5.0, 100.0, 5.0, 0.25),
    "high_frequency_edge_density": (0.01, 0.85, 0.01, 0.25),
}


def _portable(value: Any) -> Any:
    """Return JSON-friendly diagnostics without NumPy scalar/array objects."""
    if isinstance(value, np.ndarray):
        return [_portable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _portable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _clip_score(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _linear_score(value: float, start: float, full: float) -> float:
    if full <= start:
        raise ValueError("score range must be increasing")
    return _clip_score((float(value) - start) / (full - start))


def _range_deviation(
    value: float,
    minimum: float,
    maximum: float,
    scale: float,
) -> tuple[float, int]:
    """Return distance outside a provisional interval and its direction."""
    if scale <= 0.0:
        raise ValueError("deviation scale must be positive")
    if value < minimum:
        return _clip_score((minimum - value) / scale), -1
    if value > maximum:
        return _clip_score((value - maximum) / scale), 1
    return 0.0, 0


def _profile_score(
    measurements: dict[str, Any],
    profiles: dict[str, tuple[float, float, float, float]],
) -> tuple[float, dict[str, Any]]:
    weighted_sum = 0.0
    total_weight = 0.0
    deviations = {}
    for name, (minimum, maximum, scale, weight) in profiles.items():
        value = float(measurements[name])
        deviation, direction = _range_deviation(
            value,
            minimum,
            maximum,
            scale,
        )
        deviations[name] = {
            "value": value,
            "deviation": deviation,
            "direction": direction,
            "reference_interval": [minimum, maximum],
        }
        weighted_sum += weight * deviation
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError("feature profile weights must be positive")
    return _clip_score(weighted_sum / total_weight), deviations


def _prepare_grayscale(image: ImageArray) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("method_input.media.data must not be empty")
    if array.ndim not in (2, 3):
        raise ValueError("image must have two or three dimensions")

    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError("image contains non-finite pixel values")
        array = array.astype(np.float32, copy=False)
        if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise TypeError("image pixels must be boolean, integer, or float")

    if array.ndim == 2:
        gray = array
        channels = 1
    else:
        channels = int(array.shape[2])
        if channels == 1:
            gray = array[:, :, 0]
        elif channels == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("image must have 1, 3, or 4 channels")

    gray = np.ascontiguousarray(gray, dtype=np.float32)
    source_height, source_width = gray.shape
    brightness = float(np.mean(gray))
    intensity_std = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    clipped_ratio = float(np.mean((gray <= 2.0) | (gray >= 253.0)))

    reasons = []
    if min(source_height, source_width) < _MINIMUM_SOURCE_SIDE:
        reasons.append("source ROI side is below 96 pixels")
    if intensity_std <= 1e-6:
        reasons.append("ROI has insufficient intensity variation")
    if blur < 90.0:
        reasons.append("ROI is too blurred for reliable fine-detail analysis")
    if brightness < 45.0:
        reasons.append("ROI is too dark")
    if brightness > 215.0:
        reasons.append("ROI is overexposed")

    interpolation = (
        cv2.INTER_CUBIC
        if min(source_height, source_width) < _ANALYSIS_SIZE
        else cv2.INTER_AREA
    )
    resized = cv2.resize(
        gray,
        (_ANALYSIS_SIZE, _ANALYSIS_SIZE),
        interpolation=interpolation,
    ).astype(np.float32)
    quality = {
        "supported": not reasons,
        "reasons": reasons,
        "source_width": source_width,
        "source_height": source_height,
        "source_channels": channels,
        "analysis_width": _ANALYSIS_SIZE,
        "analysis_height": _ANALYSIS_SIZE,
        "brightness": brightness,
        "intensity_std": intensity_std,
        "laplacian_variance": blur,
        "clipped_pixel_ratio": clipped_ratio,
        "upscale_ratio": _ANALYSIS_SIZE / max(float(min(gray.shape)), 1.0),
    }
    return resized, quality


def _unavailable_details(
    quality: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "available": False,
        "calibrated": False,
        "status": "unavailable",
        "quality": quality,
        "measurements": {},
        "component_scores": {},
        "parameters": parameters,
        "limitations": [
            "A zero score with available=false means no interpretable result, not bona-fide evidence."
        ],
    }


def _analysis_details(
    quality: dict[str, Any],
    measurements: dict[str, Any],
    components: dict[str, Any],
    parameters: dict[str, Any],
    score: float,
    suspicious_threshold: float,
    limitations: list[str],
) -> dict[str, Any]:
    if score >= suspicious_threshold:
        status = "suspicious_signal"
    elif score >= suspicious_threshold * 0.70:
        status = "uncertain_signal"
    else:
        status = "no_strong_anomaly"
    return {
        "available": True,
        "calibrated": False,
        "status": status,
        "quality": quality,
        "measurements": measurements,
        "component_scores": components,
        "parameters": parameters,
        "limitations": [
            "The score is a deterministic engineering mapping, not an attack probability.",
            "Thresholds require calibration on the target camera and attack domain.",
            *limitations,
        ],
    }


def _fft_bundle(gray: np.ndarray) -> dict[str, np.ndarray]:
    standardized = gray.astype(np.float64) - float(np.mean(gray))
    standard_deviation = float(np.std(standardized))
    if standard_deviation > 1e-6:
        standardized /= standard_deviation
    window = np.outer(
        np.hanning(gray.shape[0]),
        np.hanning(gray.shape[1]),
    )
    windowed = standardized * window
    shifted = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(shifted)
    power = magnitude ** 2
    height, width = gray.shape
    y_coordinates, x_coordinates = np.indices(gray.shape)
    center_x = width // 2
    center_y = height // 2
    radius_scale = float(max(1, min(center_x, center_y)))
    delta_x = x_coordinates - center_x
    delta_y = y_coordinates - center_y
    radius = np.hypot(delta_x, delta_y) / radius_scale
    angle = np.mod(np.arctan2(delta_y, delta_x), math.pi)
    return {
        "power": power,
        "log_power": np.log1p(power),
        "radius": radius,
        "angle": angle,
    }


def _mass_entropy(values: np.ndarray) -> float:
    magnitudes = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    total = float(np.sum(magnitudes))
    if magnitudes.size <= 1 or total <= _EPSILON:
        return 0.0
    probabilities = magnitudes / total
    positive = probabilities[probabilities > 0.0]
    return float(
        -np.sum(positive * np.log(positive))
        / math.log(max(2, probabilities.size))
    )


def _pearson_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered ** 2))
    if variance <= _EPSILON:
        return 0.0
    return float(np.mean(centered ** 4) / (variance ** 2))


def _robust_radial_slope(
    power: np.ndarray,
    radius: np.ndarray,
    inner: float = 0.02,
    outer: float = 0.92,
    bin_count: int = 48,
) -> tuple[float, float, list[float], list[float]]:
    edges = np.linspace(inner, outer, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.asarray(
        [
            float(np.mean(power[(radius >= low) & (radius < high)]))
            for low, high in zip(edges[:-1], edges[1:])
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(means) & (means > 0.0) & (centers > 0.0)
    x_values = np.log(centers[valid])
    y_values = np.log(means[valid])
    if x_values.size < 8:
        raise ValueError("insufficient radial bins for spectral slope")

    pairwise_slopes = []
    for index in range(x_values.size - 1):
        delta_x = x_values[index + 1 :] - x_values[index]
        delta_y = y_values[index + 1 :] - y_values[index]
        usable = np.abs(delta_x) > _EPSILON
        pairwise_slopes.extend((delta_y[usable] / delta_x[usable]).tolist())
    slope = float(np.median(pairwise_slopes))
    intercept = float(np.median(y_values - slope * x_values))
    residuals = y_values - (slope * x_values + intercept)
    fit_mad = float(np.median(np.abs(residuals - np.median(residuals))))
    return slope, fit_mad, centers.tolist(), means.tolist()


def _ordinary_radial_slope(
    centers: np.ndarray,
    means: np.ndarray,
) -> tuple[float, float]:
    valid = np.isfinite(means) & (means > 0.0) & (centers > 0.0)
    x_values = np.log(centers[valid])
    y_values = np.log(means[valid])
    if x_values.size < 8:
        raise ValueError("insufficient radial bins")
    centered_x = x_values - float(np.mean(x_values))
    denominator = float(np.sum(centered_x ** 2))
    slope = float(
        np.sum(centered_x * (y_values - float(np.mean(y_values))))
        / max(denominator, _EPSILON)
    )
    intercept = float(np.mean(y_values) - slope * np.mean(x_values))
    residuals = y_values - (slope * x_values + intercept)
    rmse = math.sqrt(float(np.mean(residuals ** 2)))
    return slope, float(rmse / max(float(np.std(y_values)), _EPSILON))


def _robust_z_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    median = float(np.median(selected))
    scale = max(
        _MAD_TO_SIGMA * float(np.median(np.abs(selected - median))),
        1e-6,
    )
    return (values - median) / scale


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sum(values * weights) / max(float(np.sum(weights)), _EPSILON)
    )


def _histogram_entropy(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    bins: int = 64,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    limit = float(np.percentile(np.abs(values), 99.5))
    if limit <= _EPSILON:
        return 0.0
    histogram, _ = np.histogram(
        values,
        bins=bins,
        range=(-limit, limit),
        weights=weights,
    )
    total = float(np.sum(histogram))
    if total <= _EPSILON:
        return 0.0
    probabilities = histogram / total
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)) / math.log2(bins))


def _robust_local_inconsistency(
    records: list[dict[str, Any]],
    scale_floors: np.ndarray,
    distance_start: float,
    distance_full: float,
    neighbor_start: float,
    neighbor_full: float,
) -> tuple[float, dict[str, Any]]:
    if len(records) < 4:
        return 0.0, {
            "patch_count": len(records),
            "robust_distance_mean": 0.0,
            "robust_distance_p95": 0.0,
            "outlier_patch_ratio": 0.0,
            "neighbor_difference_p95": 0.0,
        }

    descriptors = np.stack([record["descriptor"] for record in records])
    center = np.median(descriptors, axis=0)
    scale = np.maximum(
        _MAD_TO_SIGMA * np.median(np.abs(descriptors - center), axis=0),
        np.asarray(scale_floors, dtype=np.float64),
    )
    normalized = (descriptors - center) / scale
    distances = np.sqrt(np.mean(np.minimum(normalized ** 2, 400.0), axis=1))
    grid = {(record["row"], record["column"]): record for record in records}
    neighbor_maxima = np.zeros(len(records), dtype=np.float64)
    neighbor_values = []
    indices = {id(record): index for index, record in enumerate(records)}
    for record in records:
        first_index = indices[id(record)]
        for row_offset, column_offset in ((0, 1), (1, 0)):
            neighbor = grid.get(
                (record["row"] + row_offset, record["column"] + column_offset)
            )
            if neighbor is None:
                continue
            second_index = indices[id(neighbor)]
            difference = float(
                abs(record["descriptor"][0] - neighbor["descriptor"][0])
                / max(scale[0], _EPSILON)
            )
            neighbor_values.append(difference)
            neighbor_maxima[first_index] = max(
                neighbor_maxima[first_index], difference
            )
            neighbor_maxima[second_index] = max(
                neighbor_maxima[second_index], difference
            )

    patch_scores = np.asarray(
        [
            0.75 * _linear_score(distance, distance_start, distance_full)
            + 0.25
            * _linear_score(neighbor, neighbor_start, neighbor_full)
            for distance, neighbor in zip(distances, neighbor_maxima)
        ],
        dtype=np.float64,
    )
    outlier_ratio = float(np.mean(distances >= 3.0))
    outlier_score = _clip_score(outlier_ratio / 0.35)
    score = _clip_score(
        0.65 * float(np.percentile(patch_scores, 95))
        + 0.35 * outlier_score
    )
    neighbor_array = np.asarray(neighbor_values or [0.0])
    return score, {
        "patch_count": len(records),
        "robust_distance_mean": float(np.mean(distances)),
        "robust_distance_p95": float(np.percentile(distances, 95)),
        "outlier_patch_ratio": outlier_ratio,
        "neighbor_difference_p95": float(np.percentile(neighbor_array, 95)),
    }


class GlobalFFTDeterministicMethod(DeterministicMethod):
    """Global frequency-band distribution and spectral-shape analysis."""

    @property
    def name(self) -> str:
        return "global_fft"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "window": "hann",
            "analysis_radius": [0.02, 0.92],
            "frequency_bands": {
                "low": [0.02, 0.16],
                "middle": [0.16, 0.45],
                "high": [0.45, 0.92],
            },
            "radial_bin_count": 48,
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0

        bundle = _fft_bundle(image)
        power = bundle["power"]
        radius = bundle["radius"]
        analysis_mask = (radius >= 0.02) & (radius <= 0.92)
        values = power[analysis_mask]
        total_energy = float(np.sum(values))
        if total_energy <= _EPSILON:
            self._details = _unavailable_details(
                {**self._quality, "reasons": ["FFT analysis energy is zero"]},
                parameters,
            )
            return 0.0

        band_ratio = {}
        for name, lower, upper in (
            ("low", 0.02, 0.16),
            ("middle", 0.16, 0.45),
            ("high", 0.45, 0.92),
        ):
            mask = (radius >= lower) & (radius < upper)
            band_ratio[name] = float(np.sum(power[mask]) / total_energy)

        selected_radii = radius[analysis_mask]
        centroid = float(np.sum(selected_radii * values) / total_energy)
        entropy = _mass_entropy(values)
        flatness = float(
            np.exp(np.mean(np.log(values + _EPSILON)))
            / (float(np.mean(values)) + _EPSILON)
        )
        order = np.argsort(selected_radii)
        cumulative = np.cumsum(values[order])
        rolloff_index = min(
            int(np.searchsorted(cumulative, 0.85 * total_energy)),
            order.size - 1,
        )
        rolloff = float(selected_radii[order[rolloff_index]])
        slope, fit_mad, radial_centers, radial_means = _robust_radial_slope(
            power,
            radius,
        )

        masked_power = np.where(analysis_mask, power, -1.0)
        peak_y, peak_x = np.unravel_index(np.argmax(masked_power), power.shape)
        counterpart_y = (2 * (power.shape[0] // 2) - peak_y) % power.shape[0]
        counterpart_x = (2 * (power.shape[1] // 2) - peak_x) % power.shape[1]
        peak_power = float(power[peak_y, peak_x])
        counterpart_power = float(power[counterpart_y, counterpart_x])
        pair_symmetry = min(peak_power, counterpart_power) / max(
            peak_power,
            counterpart_power,
            _EPSILON,
        )

        measurements = {
            "low_frequency_energy_ratio": band_ratio["low"],
            "middle_frequency_energy_ratio": band_ratio["middle"],
            "high_frequency_energy_ratio": band_ratio["high"],
            "spectral_centroid": centroid,
            "spectral_entropy": entropy,
            "spectral_flatness": flatness,
            "spectral_rolloff_85_radius": rolloff,
            "spectral_kurtosis_pearson": _pearson_kurtosis(values),
            "spectral_slope": slope,
            "spectral_slope_fit_mad": fit_mad,
            "high_to_low_energy_ratio": band_ratio["high"]
            / max(band_ratio["low"], _EPSILON),
            "dominant_non_dc_peak_ratio": peak_power / total_energy,
            "dominant_symmetric_pair_energy_ratio": (
                peak_power + counterpart_power
            )
            / total_energy,
            "dominant_pair_amplitude_symmetry": pair_symmetry,
            "dominant_peak": {"x": int(peak_x), "y": int(peak_y)},
            "radial_bin_centers": radial_centers,
            "radial_mean_power_profile": radial_means,
        }
        score, deviations = _profile_score(
            measurements,
            _GLOBAL_FFT_PROFILES,
        )
        self._details = _analysis_details(
            self._quality,
            measurements,
            {"feature_profile_score": score, "feature_deviations": deviations},
            parameters,
            score,
            0.60,
            [
                "Natural texture, blur, resizing, sharpening, and compression can alter the spectrum."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


def _select_moire_candidates(
    power: np.ndarray,
    log_power: np.ndarray,
    radius: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    analysis_mask = (radius >= 0.14) & (radius <= 0.82)
    total_energy = float(np.sum(power[analysis_mask]))
    if total_energy <= _EPSILON:
        return [], total_energy

    background = cv2.GaussianBlur(log_power, (0, 0), 4.0)
    residual = log_power - background
    residual_z = _robust_z_map(residual, analysis_mask)
    contrast = log_power - cv2.blur(log_power, (9, 9))
    contrast_z = _robust_z_map(contrast, analysis_mask)
    dilated = cv2.dilate(residual_z, np.ones((5, 5), dtype=np.uint8))
    candidate_mask = (
        analysis_mask
        & (residual_z >= dilated)
        & (residual_z >= 6.5)
        & (contrast_z >= 4.0)
    )

    candidates = []
    height, width = power.shape
    for y_coordinate, x_coordinate in np.argwhere(candidate_mask):
        top = max(0, int(y_coordinate) - 2)
        bottom = min(height, int(y_coordinate) + 3)
        left = max(0, int(x_coordinate) - 2)
        right = min(width, int(x_coordinate) + 3)
        energy_share = float(
            np.sum(power[top:bottom, left:right]) / total_energy
        )
        if energy_share < 0.0002:
            continue
        candidates.append(
            {
                "x": int(x_coordinate),
                "y": int(y_coordinate),
                "z_score": float(residual_z[y_coordinate, x_coordinate]),
                "power": float(power[y_coordinate, x_coordinate]),
                "energy_share": energy_share,
            }
        )
    candidates.sort(
        key=lambda candidate: candidate["z_score"] * candidate["energy_share"],
        reverse=True,
    )
    selected = []
    for candidate in candidates:
        if any(
            math.hypot(
                candidate["x"] - existing["x"],
                candidate["y"] - existing["y"],
            )
            < 5.0
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= 24:
            break
    return selected, total_energy


def _moire_metrics(gray: np.ndarray) -> dict[str, Any]:
    bundle = _fft_bundle(gray)
    power = bundle["power"]
    candidates, _total_energy = _select_moire_candidates(
        power,
        bundle["log_power"],
        bundle["radius"],
    )
    strongest = candidates[:8]
    if strongest:
        mean_z = float(np.mean([candidate["z_score"] for candidate in strongest]))
        prominence = _linear_score(mean_z, 6.5, 11.0)
        count_support = min(1.0, len(candidates) / 8.0)
        energy_share = min(
            1.0,
            float(sum(candidate["energy_share"] for candidate in candidates)),
        )
        energy_score = _linear_score(energy_share, 0.002, 0.04)
        periodic = (
            0.55 * prominence + 0.20 * count_support + 0.25 * energy_score
        ) * min(1.0, len(candidates) / 2.0)
    else:
        mean_z = 0.0
        energy_share = 0.0
        periodic = 0.0

    height, width = power.shape
    center_x = width // 2
    center_y = height // 2
    half_plane = [
        candidate
        for candidate in candidates
        if candidate["y"] < center_y
        or (candidate["y"] == center_y and candidate["x"] < center_x)
    ]
    symmetry_matches = []
    for candidate in half_plane:
        target_x = 2 * center_x - candidate["x"]
        target_y = 2 * center_y - candidate["y"]
        possible = [
            (
                math.hypot(item["x"] - target_x, item["y"] - target_y),
                item,
            )
            for item in candidates
        ]
        if not possible:
            continue
        distance, counterpart = min(possible, key=lambda item: item[0])
        if distance > 3.0:
            continue
        amplitude_ratio = min(candidate["power"], counterpart["power"]) / max(
            candidate["power"], counterpart["power"], _EPSILON
        )
        if amplitude_ratio < 0.35:
            continue
        distance_score = 1.0 - min(1.0, distance / 3.0)
        symmetry_matches.append(amplitude_ratio * (0.70 + 0.30 * distance_score))
    symmetry = (
        float(np.mean(symmetry_matches))
        * len(symmetry_matches)
        / max(len(half_plane), 1)
        if symmetry_matches
        else 0.0
    )

    if half_plane:
        angles = np.asarray(
            [
                math.atan2(
                    candidate["y"] - center_y,
                    candidate["x"] - center_x,
                )
                % math.pi
                for candidate in half_plane
            ]
        )
        weights = np.asarray([candidate["energy_share"] for candidate in half_plane])
        histogram, edges = np.histogram(
            angles,
            bins=12,
            range=(0.0, math.pi),
            weights=weights,
        )
        histogram_total = float(np.sum(histogram))
        dominant_index = int(np.argmax(histogram)) if histogram_total > 0.0 else 0
        dominant_share = (
            float(histogram[dominant_index] / histogram_total)
            if histogram_total > 0.0
            else 0.0
        )
        direction = _linear_score(dominant_share, 0.35, 0.75)
        dominant_angle = float(
            math.degrees((edges[dominant_index] + edges[dominant_index + 1]) / 2.0)
        )
    else:
        dominant_share = 0.0
        direction = 0.0
        dominant_angle = None

    periodic = _clip_score(periodic)
    symmetry = _clip_score(symmetry)
    score = _clip_score(
        0.60 * periodic
        + 0.22 * periodic * symmetry
        + 0.18 * periodic * direction
    )
    return {
        "score": score,
        "periodic_peak_score": periodic,
        "symmetric_peak_score": symmetry,
        "directional_concentration_score": direction,
        "candidate_peak_count": len(candidates),
        "symmetric_pair_count": len(symmetry_matches),
        "mean_peak_z_score": mean_z,
        "peak_energy_share": energy_share,
        "dominant_direction_share": dominant_share,
        "dominant_frequency_angle_degrees": dominant_angle,
        "candidates": candidates,
    }


class MoireDeterministicMethod(DeterministicMethod):
    """Global/local periodic FFT peaks, symmetry, and direction analysis."""

    @property
    def name(self) -> str:
        return "moire_periodic_pattern"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "window": "hann",
            "moire_radius": [0.14, 0.82],
            "peak_z_threshold": 6.5,
            "contrast_z_threshold": 4.0,
            "maximum_peak_count": 24,
            "local_patch_size": 40,
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0

        global_metrics = _moire_metrics(image)
        patch_size = 40
        local_measurements = []
        for row, top in enumerate(range(16, image.shape[0] - patch_size + 1, patch_size)):
            for column, left in enumerate(
                range(16, image.shape[1] - patch_size + 1, patch_size)
            ):
                patch = image[top : top + patch_size, left : left + patch_size]
                if float(np.std(patch)) < 5.0:
                    continue
                metrics = _moire_metrics(patch)
                strong = bool(
                    metrics["score"] >= 0.50
                    and metrics["candidate_peak_count"] >= 2
                    and metrics["symmetric_pair_count"] >= 1
                )
                local_measurements.append(
                    {
                        "row": row,
                        "column": column,
                        "x": left,
                        "y": top,
                        "size": patch_size,
                        "score": metrics["score"],
                        "strong": strong,
                        "candidate_peak_count": metrics["candidate_peak_count"],
                        "symmetric_pair_count": metrics["symmetric_pair_count"],
                        "dominant_frequency_angle_degrees": metrics[
                            "dominant_frequency_angle_degrees"
                        ],
                    }
                )

        strong_patches = [item for item in local_measurements if item["strong"]]
        if len(strong_patches) >= 2:
            top_scores = sorted(
                (item["score"] for item in strong_patches), reverse=True
            )[:3]
            doubled_angles = np.asarray(
                [
                    math.radians(2.0 * item["dominant_frequency_angle_degrees"])
                    for item in strong_patches
                    if item["dominant_frequency_angle_degrees"] is not None
                ]
            )
            orientation_consistency = (
                float(
                    math.hypot(
                        float(np.mean(np.cos(doubled_angles))),
                        float(np.mean(np.sin(doubled_angles))),
                    )
                )
                if doubled_angles.size
                else 0.0
            )
            local_score = _clip_score(
                0.70 * float(np.mean(top_scores))
                + 0.30 * orientation_consistency
            )
        else:
            orientation_consistency = 0.0
            local_score = 0.0

        score = max(float(global_metrics["score"]), local_score)
        measurements = {
            **{key: value for key, value in global_metrics.items() if key != "score"},
            "global_moire_score": global_metrics["score"],
            "local_patch_moire_score": local_score,
            "local_patch_count": len(local_measurements),
            "local_patch_strong_count": len(strong_patches),
            "local_patch_vote_ratio": len(strong_patches)
            / max(len(local_measurements), 1),
            "local_patch_orientation_consistency": orientation_consistency,
            "local_patch_measurements": local_measurements,
        }
        self._details = _analysis_details(
            self._quality,
            measurements,
            {
                "global_moire_score": global_metrics["score"],
                "local_patch_moire_score": local_score,
                "final_moire_score": score,
            },
            parameters,
            score,
            0.72,
            [
                "Hair, clothing, architecture, compression, and resampling can also create periodic peaks."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


class RadialAngularDeterministicMethod(DeterministicMethod):
    """Radial spectral decay and axial angular-concentration analysis."""

    @property
    def name(self) -> str:
        return "radial_angular_spectrum"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "window": "hann",
            "analysis_radius": [0.02, 0.92],
            "radial_bin_count": 48,
            "angular_bin_count": 36,
            "angular_domain_degrees": [0.0, 180.0],
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0

        bundle = _fft_bundle(image)
        power = bundle["power"]
        radius = bundle["radius"]
        angle = bundle["angle"]
        valid_mask = (radius >= 0.02) & (radius <= 0.92)
        total_energy = float(np.sum(power[valid_mask]))
        if total_energy <= _EPSILON:
            self._details = _unavailable_details(
                {**self._quality, "reasons": ["FFT analysis energy is zero"]},
                parameters,
            )
            return 0.0

        radial_edges = np.linspace(0.02, 0.92, 49)
        radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2.0
        radial_means = np.asarray(
            [
                float(np.mean(power[(radius >= lower) & (radius < upper)]))
                for lower, upper in zip(radial_edges[:-1], radial_edges[1:])
            ]
        )
        radial_energy = np.asarray(
            [
                float(np.sum(power[(radius >= lower) & (radius < upper)]))
                for lower, upper in zip(radial_edges[:-1], radial_edges[1:])
            ]
        )
        radial_profile = radial_energy / max(float(np.sum(radial_energy)), _EPSILON)
        radial_slope, slope_error = _ordinary_radial_slope(
            radial_centers,
            radial_means,
        )
        dominant_radial_index = int(np.argmax(radial_profile))
        left = max(0, dominant_radial_index - 2)
        right = min(radial_profile.size, dominant_radial_index + 3)
        expected_uniform = (right - left) / radial_profile.size
        narrow_concentration = float(
            np.sum(radial_profile[left:right]) / max(expected_uniform, _EPSILON)
        )

        angular_edges = np.linspace(0.0, math.pi, 37)
        angular_centers = (angular_edges[:-1] + angular_edges[1:]) / 2.0
        angular_energy = np.asarray(
            [
                float(
                    np.sum(
                        power[
                            valid_mask & (angle >= lower) & (angle < upper)
                        ]
                    )
                )
                for lower, upper in zip(angular_edges[:-1], angular_edges[1:])
            ]
        )
        angular_profile = angular_energy / max(
            float(np.sum(angular_energy)), _EPSILON
        )
        angular_degrees = np.degrees(angular_centers)
        dominant_angular_index = int(np.argmax(angular_profile))
        positive_angular = angular_profile[angular_profile > 0.0]
        angular_entropy = float(
            -np.sum(positive_angular * np.log(positive_angular))
            / math.log(angular_profile.size)
        )
        uniform = np.full(angular_profile.shape, 1.0 / angular_profile.size)
        anisotropy = float(
            0.5
            * np.sum(np.abs(angular_profile - uniform))
            / (1.0 - 1.0 / angular_profile.size)
        )

        def sector_energy(targets: tuple[float, ...]) -> float:
            selected = np.zeros(angular_profile.shape, dtype=bool)
            for target in targets:
                distance = np.abs(
                    ((angular_degrees - target + 90.0) % 180.0) - 90.0
                )
                selected |= distance <= 12.5
            return float(np.sum(angular_profile[selected]))

        measurements = {
            "radial_spectral_slope": radial_slope,
            "slope_fit_error": slope_error,
            "radial_entropy": _mass_entropy(radial_profile),
            "dominant_radial_frequency": float(
                radial_centers[dominant_radial_index]
            ),
            "dominant_radial_energy_ratio": float(
                radial_profile[dominant_radial_index]
            ),
            "narrow_band_energy_concentration": narrow_concentration,
            "maximum_angular_energy": float(
                angular_profile[dominant_angular_index]
            ),
            "dominant_frequency_angle_degrees": float(
                angular_degrees[dominant_angular_index]
            ),
            "dominant_image_line_angle_degrees": float(
                (angular_degrees[dominant_angular_index] + 90.0) % 180.0
            ),
            "angular_entropy": angular_entropy,
            "directional_anisotropy": anisotropy,
            "horizontal_concentration": sector_energy((0.0,)),
            "vertical_concentration": sector_energy((90.0,)),
            "diagonal_concentration": sector_energy((45.0, 135.0)),
            "radial_bin_centers": radial_centers,
            "radial_normalized_energy_profile": radial_profile,
            "angular_bin_centers_degrees": angular_degrees,
            "angular_energy_profile": angular_profile,
        }
        radial_score, radial_deviations = _profile_score(
            measurements,
            _RADIAL_PROFILES,
        )
        angular_score, angular_deviations = _profile_score(
            measurements,
            _ANGULAR_PROFILES,
        )
        score = _clip_score(0.55 * radial_score + 0.45 * angular_score)
        self._details = _analysis_details(
            self._quality,
            measurements,
            {
                "radial_score": radial_score,
                "angular_score": angular_score,
                "combined_score": score,
                "radial_feature_deviations": radial_deviations,
                "angular_feature_deviations": angular_deviations,
            },
            parameters,
            score,
            0.60,
            [
                "Pose, hair, illumination gradients, and crop geometry can create directional concentration."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


def _coefficient_entropy_rows(ac_values: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(ac_values)
    total = np.sum(magnitudes, axis=2, keepdims=True)
    probabilities = np.divide(
        magnitudes,
        total,
        out=np.zeros_like(magnitudes),
        where=total > _EPSILON,
    )
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
    return -np.sum(terms, axis=2) / math.log2(ac_values.shape[2])


class DCTBlockDeterministicMethod(DeterministicMethod):
    """8x8 DCT bands, decoded block boundaries, and local consistency."""

    @property
    def name(self) -> str:
        return "dct_block_analysis"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "block_size": 8,
            "near_zero_coefficient_threshold": 1.0,
            "dct_bands_by_index_sum": {
                "low": [1, 3],
                "middle": [4, 7],
                "high": [8, 14],
            },
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0

        block_size = 8
        block_rows = image.shape[0] // block_size
        block_columns = image.shape[1] // block_size
        coefficients = np.empty(
            (block_rows, block_columns, block_size, block_size),
            dtype=np.float32,
        )
        for row in range(block_rows):
            for column in range(block_columns):
                block = image[
                    row * block_size : (row + 1) * block_size,
                    column * block_size : (column + 1) * block_size,
                ]
                coefficients[row, column] = cv2.dct(
                    np.ascontiguousarray(block, dtype=np.float32)
                )

        index_sum = np.indices((8, 8)).sum(axis=0)
        ac_mask = index_sum > 0
        low_mask = (index_sum >= 1) & (index_sum <= 3)
        middle_mask = (index_sum >= 4) & (index_sum <= 7)
        high_mask = index_sum >= 8
        dc = coefficients[:, :, 0, 0].astype(np.float64)
        ac = coefficients[:, :, ac_mask].astype(np.float64)
        low_energy = np.sum(coefficients[:, :, low_mask] ** 2, axis=2)
        middle_energy = np.sum(coefficients[:, :, middle_mask] ** 2, axis=2)
        high_energy = np.sum(coefficients[:, :, high_mask] ** 2, axis=2)
        total_ac = low_energy + middle_energy + high_energy
        global_ac = float(np.sum(total_ac))
        sparsity = np.mean(np.abs(ac) <= 1.0, axis=2)
        entropy_map = _coefficient_entropy_rows(ac)
        centered_ac = ac - np.mean(ac, axis=2, keepdims=True)
        variance_map = np.mean(centered_ac ** 2, axis=2)
        kurtosis_map = np.divide(
            np.mean(centered_ac ** 4, axis=2),
            variance_map ** 2,
            out=np.zeros_like(variance_map),
            where=variance_map > _EPSILON,
        )

        vertical_boundary = float(
            np.mean(np.abs(image[:, 8::8] - image[:, 7:-1:8]))
        )
        horizontal_boundary = float(
            np.mean(np.abs(image[8::8, :] - image[7:-1:8, :]))
        )
        vertical_reference = float(
            np.mean(np.abs(image[:, 4::8] - image[:, 3:-1:8]))
        )
        horizontal_reference = float(
            np.mean(np.abs(image[4::8, :] - image[3:-1:8, :]))
        )
        vertical_ratio = vertical_boundary / max(vertical_reference, _EPSILON)
        horizontal_ratio = horizontal_boundary / max(
            horizontal_reference, _EPSILON
        )
        blockiness_ratio = 0.5 * (vertical_ratio + horizontal_ratio)
        blockiness_score = _linear_score(blockiness_ratio, 1.08, 2.20)

        descriptors = np.stack(
            [
                low_energy / (total_ac + _EPSILON),
                middle_energy / (total_ac + _EPSILON),
                high_energy / (total_ac + _EPSILON),
                sparsity,
                entropy_map,
                np.log1p(total_ac) / 10.0,
            ],
            axis=2,
        )
        local_records = []
        patch_blocks = 4
        for row in range(0, block_rows, patch_blocks):
            for column in range(0, block_columns, patch_blocks):
                patch = descriptors[
                    row : row + patch_blocks,
                    column : column + patch_blocks,
                ]
                if patch.shape[:2] != (patch_blocks, patch_blocks):
                    continue
                local_records.append(
                    {
                        "row": row // patch_blocks,
                        "column": column // patch_blocks,
                        "descriptor": np.mean(patch, axis=(0, 1)),
                    }
                )
        local_score, local_metrics = _robust_local_inconsistency(
            local_records,
            np.asarray([0.04, 0.04, 0.02, 0.04, 0.04, 0.10]),
            2.0,
            7.0,
            1.5,
            5.0,
        )

        measurements = {
            "low_frequency_ac_energy_ratio": float(np.sum(low_energy))
            / max(global_ac, _EPSILON),
            "middle_frequency_ac_energy_ratio": float(np.sum(middle_energy))
            / max(global_ac, _EPSILON),
            "high_frequency_ac_energy_ratio": float(np.sum(high_energy))
            / max(global_ac, _EPSILON),
            "ac_to_dc_ratio_mean": float(
                np.mean(total_ac / (dc ** 2 + _EPSILON))
            ),
            "near_zero_ac_coefficient_ratio": float(np.mean(np.abs(ac) <= 1.0)),
            "coefficient_entropy_global": _mass_entropy(ac),
            "coefficient_entropy_mean": float(np.mean(entropy_map)),
            "coefficient_kurtosis_global": _pearson_kurtosis(ac.ravel()),
            "coefficient_kurtosis_mean": float(np.mean(kurtosis_map)),
            "vertical_boundary_difference": vertical_boundary,
            "horizontal_boundary_difference": horizontal_boundary,
            "vertical_blockiness_ratio": vertical_ratio,
            "horizontal_blockiness_ratio": horizontal_ratio,
            "combined_8x8_periodicity_ratio": blockiness_ratio,
            **{f"local_{key}": value for key, value in local_metrics.items()},
        }
        band_score, band_deviations = _profile_score(
            measurements,
            _DCT_BAND_PROFILES,
        )
        sparsity_score, sparsity_deviations = _profile_score(
            measurements,
            _DCT_SPARSITY_PROFILES,
        )
        score = _clip_score(
            0.30 * band_score
            + 0.20 * sparsity_score
            + 0.30 * blockiness_score
            + 0.20 * local_score
        )
        self._details = _analysis_details(
            self._quality,
            measurements,
            {
                "dct_band_anomaly_score": band_score,
                "coefficient_sparsity_score": sparsity_score,
                "blockiness_score": blockiness_score,
                "local_dct_inconsistency_score": local_score,
                "final_dct_score": score,
                "band_feature_deviations": band_deviations,
                "sparsity_feature_deviations": sparsity_deviations,
            },
            parameters,
            score,
            0.68,
            [
                "Decoded pixels do not expose JPEG quantization tables; this method does not claim double-JPEG detection.",
                "Resizing can weaken or create apparent 8x8 alignment."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


# Analysis filters from the orthonormal Daubechies-2 wavelet.
_DB2_LOW = np.asarray(
    [-0.12940952255126034, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416],
    dtype=np.float64,
)
_DB2_HIGH = np.asarray(
    [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126034],
    dtype=np.float64,
)


def _periodic_filter_downsample(
    values: np.ndarray,
    coefficients: np.ndarray,
    axis: int,
) -> np.ndarray:
    length = values.shape[axis]
    output_length = length // 2
    base_indices = 2 * np.arange(output_length)
    output_shape = list(values.shape)
    output_shape[axis] = output_length
    output = np.zeros(output_shape, dtype=np.float64)
    for offset, coefficient in enumerate(coefficients):
        indices = (base_indices + offset - 1) % length
        output += coefficient * np.take(values, indices, axis=axis)
    return output


def _db2_level(values: np.ndarray) -> dict[str, np.ndarray]:
    low_rows = _periodic_filter_downsample(values, _DB2_LOW, axis=1)
    high_rows = _periodic_filter_downsample(values, _DB2_HIGH, axis=1)
    return {
        "LL": _periodic_filter_downsample(low_rows, _DB2_LOW, axis=0),
        "LH": _periodic_filter_downsample(low_rows, _DB2_HIGH, axis=0),
        "HL": _periodic_filter_downsample(high_rows, _DB2_LOW, axis=0),
        "HH": _periodic_filter_downsample(high_rows, _DB2_HIGH, axis=0),
    }


class WaveletDeterministicMethod(DeterministicMethod):
    """Two-level db2 energy, direction, and local-texture analysis."""

    @property
    def name(self) -> str:
        return "wavelet_analysis"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "wavelet": "db2",
            "decomposition_levels": 2,
            "boundary_mode": "periodic",
            "detail_near_zero_threshold": 1.0,
            "local_patch_face_size": 32,
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0
        if self._quality["clipped_pixel_ratio"] >= 0.25:
            quality = {
                **self._quality,
                "supported": False,
                "reasons": ["severe intensity clipping makes wavelet analysis unreliable"],
            }
            self._details = _unavailable_details(quality, parameters)
            return 0.0

        level_1 = _db2_level(image.astype(np.float64))
        level_2 = _db2_level(level_1["LL"])
        levels = {1: level_1, 2: level_2}
        measurements = {}
        detail_entropies = []
        detail_sparsities = []
        dominance_values = []
        anisotropy_values = []
        for level_number, level in levels.items():
            approximation_energy = float(np.mean(level["LL"] ** 2))
            detail_energies = {
                band: float(np.mean(level[band] ** 2))
                for band in ("LH", "HL", "HH")
            }
            total_detail = float(sum(detail_energies.values()))
            measurements[
                f"level_{level_number}_detail_to_approximation_energy_ratio"
            ] = total_detail / max(approximation_energy, _EPSILON)
            measurements[f"level_{level_number}_detail_energies"] = detail_energies
            energy_values = np.asarray(list(detail_energies.values()))
            dominance = float(np.max(energy_values) / max(total_detail, _EPSILON))
            anisotropy = float(
                (np.max(energy_values) - np.min(energy_values))
                / max(total_detail, _EPSILON)
            )
            dominance_values.append(dominance)
            anisotropy_values.append(anisotropy)
            for band in ("LH", "HL", "HH"):
                detail_entropies.append(_mass_entropy(level[band]))
                detail_sparsities.append(float(np.mean(np.abs(level[band]) <= 1.0)))

        measurements["global_detail_entropy_mean"] = float(
            np.mean(detail_entropies)
        )
        measurements["global_detail_sparsity_mean"] = float(
            np.mean(detail_sparsities)
        )
        measurements["directional_energy_dominance"] = float(
            np.max(dominance_values)
        )
        measurements["directional_energy_anisotropy"] = float(
            np.max(anisotropy_values)
        )

        records = []
        coefficient_patch = 16  # 16 level-1 coefficients represent 32 face pixels.
        for row in range(0, level_1["LH"].shape[0], coefficient_patch):
            for column in range(0, level_1["LH"].shape[1], coefficient_patch):
                energies = []
                for band in ("LH", "HL", "HH"):
                    patch = level_1[band][
                        row : row + coefficient_patch,
                        column : column + coefficient_patch,
                    ]
                    if patch.shape != (coefficient_patch, coefficient_patch):
                        energies = []
                        break
                    energies.append(float(np.mean(patch ** 2)))
                if not energies:
                    continue
                total = sum(energies)
                records.append(
                    {
                        "row": row // coefficient_patch,
                        "column": column // coefficient_patch,
                        "descriptor": np.asarray(
                            [
                                math.log1p(total),
                                energies[0] / max(total, _EPSILON),
                                energies[1] / max(total, _EPSILON),
                                energies[2] / max(total, _EPSILON),
                            ]
                        ),
                    }
                )
        local_score, local_metrics = _robust_local_inconsistency(
            records,
            np.asarray([0.35, 0.08, 0.08, 0.08]),
            2.0,
            6.0,
            1.5,
            5.0,
        )
        measurements.update(
            {f"local_{key}": value for key, value in local_metrics.items()}
        )

        energy_score, energy_deviations = _profile_score(
            measurements,
            _WAVELET_ENERGY_PROFILES,
        )
        dominance_score = _linear_score(
            measurements["directional_energy_dominance"], 0.55, 0.90
        )
        anisotropy_score = _linear_score(
            measurements["directional_energy_anisotropy"], 0.35, 0.80
        )
        directional_score = 0.55 * dominance_score + 0.45 * anisotropy_score
        score = _clip_score(
            0.35 * energy_score + 0.25 * directional_score + 0.40 * local_score
        )
        self._details = _analysis_details(
            self._quality,
            measurements,
            {
                "wavelet_energy_score": energy_score,
                "directional_wavelet_score": directional_score,
                "local_wavelet_inconsistency_score": local_score,
                "final_wavelet_score": score,
                "energy_feature_deviations": energy_deviations,
            },
            parameters,
            score,
            0.68,
            [
                "Blur, resampling, compression, and sharpening affect wavelet subbands.",
                "The db2 basis is an experimental default and must be validated against alternatives."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


def _residual_spatial_weights(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y_coordinates, x_coordinates = np.indices(shape, dtype=np.float64)
    normalized_y = y_coordinates / max(float(height - 1), 1.0)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    ellipse = np.clip(
        1.0
        - ((x_coordinates - center_x) / (0.40 * width)) ** 2
        - ((y_coordinates - center_y) / (0.45 * height)) ** 2,
        0.0,
        1.0,
    )
    top_weight = np.clip(normalized_y / 0.25, 0.0, 1.0)
    bottom_weight = np.clip((1.0 - normalized_y) / (1.0 - 0.82), 0.0, 1.0)
    weights = ellipse * top_weight * bottom_weight
    eye_band = np.abs(normalized_y - 0.38) <= 0.075
    weights[eye_band] *= 0.45
    return weights


def _signed_statistics(
    values: np.ndarray,
    weights: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    mean = _weighted_mean(values, weights)
    centered = values - mean
    variance = _weighted_mean(centered ** 2, weights)
    positive_weight = float(np.sum(weights[values > 1e-6]))
    negative_weight = float(np.sum(weights[values < -1e-6]))
    signed_weight = positive_weight + negative_weight
    return {
        prefix + "mean": mean,
        prefix + "variance": variance,
        prefix + "mean_absolute_deviation": _weighted_mean(
            np.abs(centered), weights
        ),
        prefix + "rms_energy": math.sqrt(
            max(_weighted_mean(values ** 2, weights), 0.0)
        ),
        prefix + "entropy": _histogram_entropy(values, weights),
        prefix + "kurtosis": _weighted_mean(centered ** 4, weights)
        / max(variance ** 2, _EPSILON),
        prefix + "positive_negative_balance": (
            positive_weight - negative_weight
        )
        / max(signed_weight, _EPSILON),
    }


class HighPassResidualDeterministicMethod(DeterministicMethod):
    """Gaussian, Laplacian, gradient, and local residual analysis."""

    @property
    def name(self) -> str:
        return "high_pass_residual_analysis"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data
        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)
        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(self, image: ImageArray) -> ImageArray:
        processed, self._quality = _prepare_grayscale(image)
        return processed

    def _analyze(self, image: ImageArray) -> float:
        parameters = {
            "analysis_size": _ANALYSIS_SIZE,
            "gaussian_kernel_size": 5,
            "gaussian_sigma": 1.2,
            "laplacian_kernel_size": 3,
            "sobel_kernel_size": 3,
            "edge_magnitude_threshold": 24.0,
            "local_patch_size": 32,
        }
        if not self._quality["supported"]:
            self._details = _unavailable_details(self._quality, parameters)
            return 0.0
        if self._quality["clipped_pixel_ratio"] >= 0.25:
            quality = {
                **self._quality,
                "supported": False,
                "reasons": ["severe intensity clipping makes residual analysis unreliable"],
            }
            self._details = _unavailable_details(quality, parameters)
            return 0.0

        gaussian_blur = cv2.GaussianBlur(
            image,
            (5, 5),
            1.2,
            borderType=cv2.BORDER_REFLECT101,
        )
        gaussian = image - gaussian_blur
        laplacian = cv2.Laplacian(
            image,
            cv2.CV_32F,
            ksize=3,
            borderType=cv2.BORDER_REFLECT101,
        )
        sobel_x = cv2.Sobel(
            image,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
            borderType=cv2.BORDER_REFLECT101,
        )
        sobel_y = cv2.Sobel(
            image,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
            borderType=cv2.BORDER_REFLECT101,
        )
        gradient = cv2.magnitude(sobel_x, sobel_y)
        spatial_weights = _residual_spatial_weights(image.shape)
        valid = spatial_weights >= 0.08
        weights = spatial_weights[valid]
        gaussian_values = gaussian[valid].astype(np.float64)
        laplacian_values = laplacian[valid].astype(np.float64)
        gradient_values = gradient[valid].astype(np.float64)

        measurements = {}
        measurements.update(
            _signed_statistics(
                gaussian_values,
                weights,
                "gaussian_residual_",
            )
        )
        measurements.update(
            _signed_statistics(laplacian_values, weights, "laplacian_")
        )
        measurements.update(
            {
                "gradient_mean_magnitude": _weighted_mean(
                    gradient_values, weights
                ),
                "gradient_energy": _weighted_mean(
                    gradient_values ** 2, weights
                ),
                "high_frequency_edge_density": _weighted_mean(
                    (gradient_values >= 24.0).astype(np.float64), weights
                ),
                "sobel_x_energy": _weighted_mean(
                    sobel_x[valid].astype(np.float64) ** 2, weights
                ),
                "sobel_y_energy": _weighted_mean(
                    sobel_y[valid].astype(np.float64) ** 2, weights
                ),
                "residual_mask_effective_ratio": float(np.mean(valid)),
            }
        )

        records = []
        patch_size = 32
        for row, top in enumerate(range(0, image.shape[0], patch_size)):
            for column, left in enumerate(range(0, image.shape[1], patch_size)):
                patch_valid = valid[
                    top : top + patch_size,
                    left : left + patch_size,
                ]
                if patch_valid.shape != (patch_size, patch_size):
                    continue
                if float(np.mean(patch_valid)) < 0.30:
                    continue
                patch_weights = spatial_weights[
                    top : top + patch_size,
                    left : left + patch_size,
                ][patch_valid]
                gaussian_patch = gaussian[
                    top : top + patch_size,
                    left : left + patch_size,
                ][patch_valid]
                laplacian_patch = laplacian[
                    top : top + patch_size,
                    left : left + patch_size,
                ][patch_valid]
                gradient_patch = gradient[
                    top : top + patch_size,
                    left : left + patch_size,
                ][patch_valid]
                gaussian_energy = _weighted_mean(
                    gaussian_patch.astype(np.float64) ** 2,
                    patch_weights,
                )
                laplacian_energy = _weighted_mean(
                    laplacian_patch.astype(np.float64) ** 2,
                    patch_weights,
                )
                gradient_energy = _weighted_mean(
                    gradient_patch.astype(np.float64) ** 2,
                    patch_weights,
                )
                positive = float(np.sum(patch_weights[gaussian_patch > 1e-6]))
                negative = float(np.sum(patch_weights[gaussian_patch < -1e-6]))
                records.append(
                    {
                        "row": row,
                        "column": column,
                        "descriptor": np.asarray(
                            [
                                math.log1p(gaussian_energy),
                                math.log1p(laplacian_energy),
                                math.log1p(gradient_energy),
                                (positive - negative)
                                / max(positive + negative, _EPSILON),
                            ]
                        ),
                    }
                )
        local_score, local_metrics = _robust_local_inconsistency(
            records,
            np.asarray([0.35, 0.45, 0.45, 0.15]),
            2.0,
            6.0,
            1.5,
            5.0,
        )
        measurements.update(
            {f"local_{key}": value for key, value in local_metrics.items()}
        )

        gaussian_score, gaussian_deviations = _profile_score(
            measurements,
            _GAUSSIAN_RESIDUAL_PROFILES,
        )
        laplacian_score, laplacian_deviations = _profile_score(
            measurements,
            _LAPLACIAN_PROFILES,
        )
        gradient_score, gradient_deviations = _profile_score(
            measurements,
            _GRADIENT_PROFILES,
        )
        score = _clip_score(
            0.30 * gaussian_score
            + 0.20 * laplacian_score
            + 0.20 * gradient_score
            + 0.30 * local_score
        )
        self._details = _analysis_details(
            self._quality,
            measurements,
            {
                "gaussian_residual_score": gaussian_score,
                "laplacian_score": laplacian_score,
                "gradient_score": gradient_score,
                "local_residual_inconsistency_score": local_score,
                "final_residual_score": score,
                "gaussian_feature_deviations": gaussian_deviations,
                "laplacian_feature_deviations": laplacian_deviations,
                "gradient_feature_deviations": gradient_deviations,
            },
            parameters,
            score,
            0.68,
            [
                "Noise, denoising, sharpening, compression, and resizing all affect residual energy.",
                "This method does not identify a sensor and does not claim PRNU extraction."
            ],
        )
        return score

    def _normalize_score(self, raw_value: float) -> float:
        return _clip_score(raw_value)

    def _build_details(self, raw_value: float) -> dict[str, Any]:
        return _portable({"raw_value": raw_value, **self._details})


__all__ = [
    "GlobalFFTDeterministicMethod",
    "MoireDeterministicMethod",
    "RadialAngularDeterministicMethod",
    "DCTBlockDeterministicMethod",
    "WaveletDeterministicMethod",
    "HighPassResidualDeterministicMethod",
]
