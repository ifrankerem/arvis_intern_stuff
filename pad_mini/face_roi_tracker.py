"""Yuz ROI tespitini karar mantigindan ayiran kararlı takip katmani."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_models import FaceBox


@dataclass(frozen=True)
class FaceROITrackingResult:
    """Bir karede PreControl'a verilecek yuz ROI'sinin durumu."""

    box: Optional[FaceBox]
    status: str
    detection_count: int
    fresh_detection: bool
    reliability: float
    reason: str

    @property
    def supported(self):
        return self.box is not None and self.status in ("DETECTED", "HELD")


class StableFaceROITracker:
    """Landmark kutusunu yumusatir ve kisa tespit bosluklarini kapatir.

    Bu sinif yalnızca ROI altyapisidir. Urettigi durum veya guven degeri
    presentation-attack/liveness skoruna kanit olarak verilmez.
    """

    def __init__(
        self,
        detector,
        smoothing_alpha=0.32,
        hold_seconds=0.80,
        horizontal_expansion_ratio=0.10,
        vertical_expansion_ratio=0.12,
        jump_iou_threshold=0.12,
        minimum_frame_margin_ratio=0.01,
    ):
        self.detector = detector
        self.smoothing_alpha = float(
            np.clip(smoothing_alpha, 0.01, 1.0)
        )
        self.hold_seconds = max(0.0, float(hold_seconds))
        self.horizontal_expansion_ratio = max(
            0.0,
            float(horizontal_expansion_ratio),
        )
        self.vertical_expansion_ratio = max(
            0.0,
            float(vertical_expansion_ratio),
        )
        self.jump_iou_threshold = float(
            np.clip(jump_iou_threshold, 0.0, 1.0)
        )
        self.minimum_frame_margin_ratio = float(
            np.clip(minimum_frame_margin_ratio, 0.0, 0.20)
        )
        self.smoothed_box = None
        self.last_detection_timestamp_s = None
        self.last_detector_timestamp_ms = None

    def update(self, frame, timestamp_s):
        if self.detector is None:
            return FaceROITrackingResult(
                None,
                "DETECTOR_UNAVAILABLE",
                0,
                False,
                0.0,
                "face detector is unavailable",
            )

        timestamp_s = float(timestamp_s)
        timestamp_ms = int(round(timestamp_s * 1000.0))
        if self.last_detector_timestamp_ms is not None:
            timestamp_ms = max(
                timestamp_ms,
                self.last_detector_timestamp_ms + 1,
            )
        self.last_detector_timestamp_ms = timestamp_ms

        try:
            detected_faces = self.detector.detect_faces(frame, timestamp_ms)
        except Exception as error:
            return self._held_or_missing(
                timestamp_s,
                "face detector error: " + str(error),
                status_when_missing="DETECTOR_ERROR",
            )

        detection_count = len(detected_faces)
        if detection_count > 1:
            self.smoothed_box = None
            self.last_detection_timestamp_s = None
            return FaceROITrackingResult(
                None,
                "MULTIPLE_FACES",
                detection_count,
                False,
                0.0,
                "multiple faces detected",
            )
        if detection_count == 0:
            return self._held_or_missing(
                timestamp_s,
                "face temporarily not detected",
            )

        frame_height, frame_width = frame.shape[:2]
        detected_box = self._expanded_box(
            detected_faces[0].box,
            frame_width,
            frame_height,
        )
        if detected_box.width <= 0 or detected_box.height <= 0:
            return self._held_or_missing(
                timestamp_s,
                "detected face ROI is empty",
            )

        if (
            self.smoothed_box is None
            or self._intersection_over_union(
                self.smoothed_box,
                detected_box,
            )
            < self.jump_iou_threshold
        ):
            self.smoothed_box = detected_box
        else:
            self.smoothed_box = self._smooth_box(
                self.smoothed_box,
                detected_box,
                frame_width,
                frame_height,
            )
        self.last_detection_timestamp_s = timestamp_s
        return FaceROITrackingResult(
            self.smoothed_box,
            "DETECTED",
            1,
            True,
            1.0,
            "face detected and tracked",
        )

    def reset(self):
        self.smoothed_box = None
        self.last_detection_timestamp_s = None
        self.last_detector_timestamp_ms = None

    def _held_or_missing(
        self,
        timestamp_s,
        reason,
        status_when_missing="NO_FACE",
    ):
        if (
            self.smoothed_box is not None
            and self.last_detection_timestamp_s is not None
            and self.hold_seconds > 0.0
        ):
            elapsed = max(
                0.0,
                timestamp_s - self.last_detection_timestamp_s,
            )
            if elapsed <= self.hold_seconds:
                reliability = max(0.0, 1.0 - elapsed / self.hold_seconds)
                return FaceROITrackingResult(
                    self.smoothed_box,
                    "HELD",
                    0,
                    False,
                    reliability,
                    reason,
                )

        self.smoothed_box = None
        self.last_detection_timestamp_s = None
        return FaceROITrackingResult(
            None,
            status_when_missing,
            0,
            False,
            0.0,
            reason,
        )

    def _expanded_box(self, box, frame_width, frame_height):
        expand_x = int(round(box.width * self.horizontal_expansion_ratio))
        expand_y = int(round(box.height * self.vertical_expansion_ratio))
        expanded = FaceBox(
            box.x - expand_x,
            box.y - expand_y,
            box.width + 2 * expand_x,
            box.height + 2 * expand_y,
        ).clamp_to_frame(frame_width, frame_height)

        margin_x = int(frame_width * self.minimum_frame_margin_ratio)
        margin_y = int(frame_height * self.minimum_frame_margin_ratio)
        base = box.clamp_to_frame(frame_width, frame_height)
        base_touches_edge = (
            base.x <= margin_x
            or base.y <= margin_y
            or base.x + base.width >= frame_width - margin_x
            or base.y + base.height >= frame_height - margin_y
        )
        if base_touches_edge:
            # Gercek landmark kutusu kenardaysa kalite kapisinin parcali yuz
            # uyarisini koru. Yalnizca ek genisletme kenara tasiyorsa asagida
            # ROI'yi guvenli paya kadar daralt.
            return expanded

        left = max(margin_x + 1, expanded.x)
        top = max(margin_y + 1, expanded.y)
        right = min(
            frame_width - margin_x - 1,
            expanded.x + expanded.width,
        )
        bottom = min(
            frame_height - margin_y - 1,
            expanded.y + expanded.height,
        )
        return FaceBox(
            left,
            top,
            max(0, right - left),
            max(0, bottom - top),
        )

    def _smooth_box(
        self,
        previous,
        current,
        frame_width,
        frame_height,
    ):
        alpha = self.smoothing_alpha

        def blended(old, new):
            return int(round((1.0 - alpha) * old + alpha * new))

        return FaceBox(
            blended(previous.x, current.x),
            blended(previous.y, current.y),
            max(1, blended(previous.width, current.width)),
            max(1, blended(previous.height, current.height)),
        ).clamp_to_frame(frame_width, frame_height)

    @staticmethod
    def _intersection_over_union(first, second):
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = first.get_area() + second.get_area() - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)
