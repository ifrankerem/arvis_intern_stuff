from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from data_models import FaceBox, FaceQuality


@dataclass(frozen=True)
class QualityThresholds:
    """Kameraya ve ortama göre ayarlanabilecek kalite eşikleri."""

    min_face_area_ratio: float = 0.08
    min_blur_score: float = 80.0
    min_brightness: float = 60.0
    max_brightness: float = 200.0


class QualityAnalyzer:
    """Tespit edilen yüz bölgesinin görüntü kalitesini ölçer."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def analyze(self, frame: np.ndarray, face_box: FaceBox) -> FaceQuality:
        frame_height, frame_width = frame.shape[:2]
        safe_box = face_box.clamp_to_frame(frame_width, frame_height)

        if safe_box.area == 0:
            raise ValueError("Yüz bölgesi boş olarak kırpıldı.")

        frame_area = frame_height * frame_width
        face_area_ratio = safe_box.area / frame_area

        face_region = frame[
            safe_box.y : safe_box.y + safe_box.height,
            safe_box.x : safe_box.x + safe_box.width,
        ]
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        blur_score = float(
            cv2.Laplacian(face_gray, cv2.CV_64F).var()
        )
        brightness = float(face_gray.mean())

        return FaceQuality(
            face_area_ratio=face_area_ratio,
            blur_score=blur_score,
            brightness=brightness,
            face_large_enough=(
                face_area_ratio >= self.thresholds.min_face_area_ratio
            ),
            sharp_enough=(
                blur_score >= self.thresholds.min_blur_score
            ),
            brightness_ok=(
                self.thresholds.min_brightness
                <= brightness
                <= self.thresholds.max_brightness
            ),
        )

    def get_message(
        self,
        number_of_faces: int,
        quality: FaceQuality | None,
    ) -> tuple[str, tuple[int, int, int]]:
        """Ekranda gösterilecek mesajı ve BGR rengini döndürür."""

        if number_of_faces == 0:
            return "YUZ BULUNAMADI", (0, 0, 255)

        if number_of_faces > 1:
            return "BIRDEN FAZLA YUZ", (0, 0, 255)

        if quality is None:
            return "ANALIZ HATASI", (0, 0, 255)

        if not quality.face_large_enough:
            return "KAMERAYA YAKLAS", (0, 165, 255)

        if not quality.sharp_enough:
            return "GORUNTU BULANIK", (0, 165, 255)

        if quality.brightness < self.thresholds.min_brightness:
            return "ORTAM COK KARANLIK", (0, 165, 255)

        if quality.brightness > self.thresholds.max_brightness:
            return "ORTAM COK PARLAK", (0, 165, 255)

        return "ANALIZE UYGUN", (0, 255, 0)
