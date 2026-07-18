import cv2

import config
from data_models import Color, DisplayMessage, FaceQuality


class QualityAnalyzer:
    """Yuz bolgesinin goruntu kalitesini kontrol eder."""

    def analyze(self, frame, face_box):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        safe_box = face_box.clamp_to_frame(
            frame_width,
            frame_height,
        )

        if safe_box.get_area() == 0:
            raise ValueError("Yuz bolgesi bos olarak kirpildi.")

        frame_area = frame_height * frame_width
        face_area = safe_box.get_area()
        face_area_ratio = face_area / frame_area

        face_region = frame[
            safe_box.y : safe_box.y + safe_box.height,
            safe_box.x : safe_box.x + safe_box.width,
        ]

        face_gray = cv2.cvtColor(
            face_region,
            cv2.COLOR_BGR2GRAY,
        )

        laplacian_image = cv2.Laplacian(
            face_gray,
            cv2.CV_64F,
        )
        blur_score = float(laplacian_image.var())
        brightness = float(face_gray.mean())

        face_large_enough = (
            face_area_ratio >= config.MINIMUM_FACE_AREA_RATIO
        )
        sharp_enough = blur_score >= config.MINIMUM_BLUR_SCORE
        brightness_ok = (
            config.MINIMUM_BRIGHTNESS
            <= brightness
            <= config.MAXIMUM_BRIGHTNESS
        )

        return FaceQuality(
            face_area_ratio,
            blur_score,
            brightness,
            face_large_enough,
            sharp_enough,
            brightness_ok,
        )

    def get_message(self, number_of_faces, quality):
        error_color = Color(0, 0, 255)
        warning_color = Color(0, 165, 255)
        success_color = Color(0, 255, 0)

        if number_of_faces == 0:
            return DisplayMessage("YUZ BULUNAMADI", error_color)

        if number_of_faces > 1:
            return DisplayMessage("BIRDEN FAZLA YUZ", error_color)

        if quality is None:
            return DisplayMessage("ANALIZ HATASI", error_color)

        if not quality.face_large_enough:
            return DisplayMessage("KAMERAYA YAKLAS", warning_color)

        if not quality.sharp_enough:
            return DisplayMessage("GORUNTU BULANIK", warning_color)

        if quality.brightness < config.MINIMUM_BRIGHTNESS:
            return DisplayMessage("ORTAM COK KARANLIK", warning_color)

        if quality.brightness > config.MAXIMUM_BRIGHTNESS:
            return DisplayMessage("ORTAM COK PARLAK", warning_color)

        return DisplayMessage("ANALIZE UYGUN", success_color)
