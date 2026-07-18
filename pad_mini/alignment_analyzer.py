from __future__ import annotations

import math
from dataclasses import dataclass

from data_models import FaceAlignment, FaceLandmarkDetection


@dataclass(frozen=True)
class AlignmentThresholds:
    """Kafa ve ağız hizası için ayarlanabilir eşikler."""

    maximum_yaw_degrees: float = 15.0
    maximum_pitch_degrees: float = 12.0
    maximum_roll_degrees: float = 10.0
    maximum_mouth_angle_degrees: float = 10.0
    maximum_mouth_lateral_difference: float = 0.25
    maximum_jaw_open_score: float = 0.35


class AlignmentAnalyzer:
    """MediaPipe çıktısından kafa ve ağız hizasını hesaplar."""

    def __init__(
        self,
        thresholds: AlignmentThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or AlignmentThresholds()

    def analyze(self, face: FaceLandmarkDetection) -> FaceAlignment:
        screen_left_corner = min(
            face.left_mouth_corner,
            face.right_mouth_corner,
            key=lambda point: point.x,
        )
        screen_right_corner = max(
            face.left_mouth_corner,
            face.right_mouth_corner,
            key=lambda point: point.x,
        )
        mouth_delta_x = (
            screen_right_corner.x - screen_left_corner.x
        )
        mouth_delta_y = (
            screen_right_corner.y - screen_left_corner.y
        )
        mouth_angle_degrees = math.degrees(
            math.atan2(mouth_delta_y, mouth_delta_x)
        )
        mouth_lateral_difference = abs(
            face.mouth_left_score - face.mouth_right_score
        )

        head_aligned = (
            abs(face.yaw_degrees)
            <= self.thresholds.maximum_yaw_degrees
            and abs(face.pitch_degrees)
            <= self.thresholds.maximum_pitch_degrees
            and abs(face.roll_degrees)
            <= self.thresholds.maximum_roll_degrees
        )
        mouth_closed = (
            face.jaw_open_score
            <= self.thresholds.maximum_jaw_open_score
        )
        mouth_centered = (
            mouth_lateral_difference
            <= self.thresholds.maximum_mouth_lateral_difference
        )
        mouth_aligned = (
            mouth_closed
            and mouth_centered
            and abs(mouth_angle_degrees)
            <= self.thresholds.maximum_mouth_angle_degrees
        )

        return FaceAlignment(
            yaw_degrees=face.yaw_degrees,
            pitch_degrees=face.pitch_degrees,
            roll_degrees=face.roll_degrees,
            mouth_angle_degrees=mouth_angle_degrees,
            mouth_open_score=face.jaw_open_score,
            mouth_lateral_difference=mouth_lateral_difference,
            head_aligned=head_aligned,
            mouth_closed=mouth_closed,
            mouth_centered=mouth_centered,
            mouth_aligned=mouth_aligned,
        )

    def get_message(self, alignment: FaceAlignment | None) -> str:
        if alignment is None:
            return "HIZALAMA YOK"

        if not alignment.mouth_closed:
            return "AGZINI KAPAT"

        if not alignment.mouth_centered:
            return "AGZINI DUZ TUT"

        if abs(alignment.yaw_degrees) > self.thresholds.maximum_yaw_degrees:
            return "KAMERAYA DUZ BAK"

        if abs(alignment.pitch_degrees) > self.thresholds.maximum_pitch_degrees:
            return "BASINI DUZ TUT"

        if abs(alignment.roll_degrees) > self.thresholds.maximum_roll_degrees:
            return "BASINI YANA EGME"

        if not alignment.mouth_aligned:
            return "AGIZ HIZASI UYGUN DEGIL"

        return "HIZALAMA UYGUN"
