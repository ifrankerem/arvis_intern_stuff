import math

import config
from data_models import Color, DisplayMessage, FaceAlignment


class AlignmentAnalyzer:
    """Kafa ve agiz hizalamasini kontrol eder."""

    def analyze(self, face):
        left_corner = face.left_mouth_corner
        right_corner = face.right_mouth_corner

        # Ekranda solda kalan noktayi acik bir sekilde belirliyoruz.
        if left_corner.x <= right_corner.x:
            screen_left_corner = left_corner
            screen_right_corner = right_corner
        else:
            screen_left_corner = right_corner
            screen_right_corner = left_corner

        mouth_delta_x = screen_right_corner.x - screen_left_corner.x
        mouth_delta_y = screen_right_corner.y - screen_left_corner.y

        mouth_angle_radians = math.atan2(
            mouth_delta_y,
            mouth_delta_x,
        )
        mouth_angle_degrees = math.degrees(mouth_angle_radians)

        mouth_lateral_difference = abs(
            face.mouth_left_score - face.mouth_right_score
        )

        head_aligned = self.is_head_aligned(face)
        mouth_closed = (
            face.jaw_open_score <= config.MAXIMUM_JAW_OPEN_SCORE
        )
        mouth_centered = (
            mouth_lateral_difference
            <= config.MAXIMUM_MOUTH_LATERAL_DIFFERENCE
        )
        mouth_angle_ok = (
            abs(mouth_angle_degrees)
            <= config.MAXIMUM_MOUTH_ANGLE_DEGREES
        )
        mouth_aligned = (
            mouth_closed
            and mouth_centered
            and mouth_angle_ok
        )

        return FaceAlignment(
            face.yaw_degrees,
            face.pitch_degrees,
            face.roll_degrees,
            mouth_angle_degrees,
            face.jaw_open_score,
            mouth_lateral_difference,
            head_aligned,
            mouth_closed,
            mouth_centered,
            mouth_aligned,
        )

    def is_head_aligned(self, face):
        yaw_ok = abs(face.yaw_degrees) <= config.MAXIMUM_YAW_DEGREES
        pitch_ok = (
            abs(face.pitch_degrees) <= config.MAXIMUM_PITCH_DEGREES
        )
        roll_ok = (
            abs(face.roll_degrees) <= config.MAXIMUM_ROLL_DEGREES
        )

        return yaw_ok and pitch_ok and roll_ok

    def get_message(self, alignment):
        warning_color = Color(0, 165, 255)
        success_color = Color(0, 255, 0)

        if alignment is None:
            return DisplayMessage("HIZALAMA YOK", warning_color)

        if not alignment.mouth_closed:
            return DisplayMessage("AGZINI KAPAT", warning_color)

        if not alignment.mouth_centered:
            return DisplayMessage("AGZINI DUZ TUT", warning_color)

        if abs(alignment.yaw_degrees) > config.MAXIMUM_YAW_DEGREES:
            return DisplayMessage("KAMERAYA DUZ BAK", warning_color)

        if abs(alignment.pitch_degrees) > config.MAXIMUM_PITCH_DEGREES:
            return DisplayMessage("BASINI DUZ TUT", warning_color)

        if abs(alignment.roll_degrees) > config.MAXIMUM_ROLL_DEGREES:
            return DisplayMessage("BASINI YANA EGME", warning_color)

        if not alignment.mouth_aligned:
            return DisplayMessage("AGIZ HIZASI UYGUN DEGIL", warning_color)

        return DisplayMessage("HIZALAMA UYGUN", success_color)
