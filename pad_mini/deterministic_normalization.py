"""Auditable, training-free mappings from raw measurements to [0, 1]."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


ROBUST_GAUSSIAN_MAD_SCALE = 1.4826


@dataclass(frozen=True)
class FeatureCalibration:
    """Human-readable calibration parameters for one scalar feature.

    ``direction`` is one of ``high``, ``low`` or ``outside``. For high/low
    features, robust z thresholds are used. For outside features, lower and
    upper valid limits are required and ``transition_width`` controls the
    monotonic risk ramp outside that interval.
    """

    direction: str
    median_reference: Optional[float] = None
    mad_reference: Optional[float] = None
    z_start: float = 2.0
    z_full: float = 5.0
    valid_interval: Optional[Tuple[float, float]] = None
    transition_width: Optional[float] = None
    winsor_limits: Optional[Tuple[float, float]] = None
    minimum_mad: float = 1e-6


def robust_z(value, median_reference, mad_reference, minimum_mad=1e-6):
    """Return a signed robust standardized value.

    A near-zero MAD is not silently accepted: the configured floor is used,
    and callers should document how that floor was obtained for their feature.
    """
    scale = max(
        abs(float(mad_reference)) * ROBUST_GAUSSIAN_MAD_SCALE,
        float(minimum_mad),
    )
    return (float(value) - float(median_reference)) / scale


def smoothstep(value, start, full):
    """Monotonic cubic mapping with zero slope at both configured bounds."""
    start = float(start)
    full = float(full)
    if full <= start:
        raise ValueError("smoothstep requires full > start")
    t = float(np.clip((float(value) - start) / (full - start), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def normalize_feature(value, calibration):
    """Normalize one finite raw feature according to explicit suspicion direction."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("feature value must be finite")
    if calibration.winsor_limits is not None:
        low, high = calibration.winsor_limits
        if high <= low:
            raise ValueError("winsor high limit must exceed low limit")
        numeric = float(np.clip(numeric, low, high))

    direction = calibration.direction.lower()
    if direction in ("high", "low"):
        if (
            calibration.median_reference is None
            or calibration.mad_reference is None
        ):
            raise ValueError("high/low calibration requires median and MAD")
        z_value = robust_z(
            numeric,
            calibration.median_reference,
            calibration.mad_reference,
            calibration.minimum_mad,
        )
        suspicious_z = z_value if direction == "high" else -z_value
        return smoothstep(
            suspicious_z,
            calibration.z_start,
            calibration.z_full,
        )

    if direction == "outside":
        if calibration.valid_interval is None:
            raise ValueError("outside calibration requires a valid interval")
        lower, upper = calibration.valid_interval
        if upper <= lower:
            raise ValueError("valid interval upper bound must exceed lower")
        width = calibration.transition_width
        if width is None or width <= 0.0:
            raise ValueError("outside calibration requires transition_width > 0")
        distance = max(float(lower) - numeric, numeric - float(upper), 0.0)
        return smoothstep(distance, 0.0, float(width))

    raise ValueError("unsupported feature suspicion direction: " + direction)

