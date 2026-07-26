"""Timestamp quality assessment shared by future deterministic video methods."""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalSamplingAssessment:
    supported: bool
    reliability: float
    frame_count: int
    duration_s: float
    estimated_fps: Optional[float]
    interval_median_s: Optional[float]
    interval_mad_s: Optional[float]
    jitter_ratio: Optional[float]
    large_gap_ratio: Optional[float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "supported": self.supported,
            "reliability": self.reliability,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "estimated_fps": self.estimated_fps,
            "interval_median_s": self.interval_median_s,
            "interval_mad_s": self.interval_mad_s,
            "jitter_ratio": self.jitter_ratio,
            "large_gap_ratio": self.large_gap_ratio,
            "warnings": list(self.warnings),
        }


def assess_temporal_sampling(
    timestamps_s: Sequence[float],
    minimum_frames=32,
    minimum_duration_s=1.0,
    minimum_fps=12.0,
):
    """Assess whether timestamps can support PSD/rPPG/motion calculations."""
    values = np.asarray(timestamps_s, dtype=np.float64)
    warnings = []
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return _unsupported(values.size, "timestamps are missing or non-finite")
    intervals = np.diff(values)
    if np.any(intervals <= 0.0):
        return _unsupported(values.size, "timestamps are not strictly increasing")

    duration = float(values[-1] - values[0])
    median_interval = float(np.median(intervals))
    mad_interval = float(np.median(np.abs(intervals - median_interval)))
    estimated_fps = 1.0 / median_interval
    jitter_ratio = 1.4826 * mad_interval / max(median_interval, 1e-9)
    large_gap_ratio = float(np.mean(intervals > 1.8 * median_interval))

    frame_factor = min(1.0, values.size / max(1.0, float(minimum_frames)))
    duration_factor = min(1.0, duration / max(float(minimum_duration_s), 1e-9))
    fps_factor = min(1.0, estimated_fps / max(float(minimum_fps), 1e-9))
    jitter_factor = float(np.clip(1.0 - jitter_ratio / 0.25, 0.0, 1.0))
    gap_factor = float(np.clip(1.0 - large_gap_ratio / 0.20, 0.0, 1.0))
    reliability = float(
        np.clip(
            frame_factor
            * duration_factor
            * fps_factor
            * jitter_factor
            * gap_factor,
            0.0,
            1.0,
        )
    )
    if values.size < minimum_frames:
        warnings.append("too few frames")
    if duration < minimum_duration_s:
        warnings.append("temporal window is too short")
    if estimated_fps < minimum_fps:
        warnings.append("estimated frame rate is too low")
    if jitter_ratio > 0.10:
        warnings.append("timestamp cadence is irregular")
    if large_gap_ratio > 0.05:
        warnings.append("dropped-frame gaps are present")
    supported = bool(
        values.size >= minimum_frames
        and duration >= minimum_duration_s
        and estimated_fps >= minimum_fps
        and reliability >= 0.35
    )
    return TemporalSamplingAssessment(
        supported=supported,
        reliability=reliability,
        frame_count=int(values.size),
        duration_s=duration,
        estimated_fps=estimated_fps,
        interval_median_s=median_interval,
        interval_mad_s=mad_interval,
        jitter_ratio=jitter_ratio,
        large_gap_ratio=large_gap_ratio,
        warnings=warnings,
    )


def _unsupported(frame_count, warning):
    return TemporalSamplingAssessment(
        supported=False,
        reliability=0.0,
        frame_count=int(frame_count),
        duration_s=0.0,
        estimated_fps=None,
        interval_median_s=None,
        interval_mad_s=None,
        jitter_ratio=None,
        large_gap_ratio=None,
        warnings=[warning],
    )

