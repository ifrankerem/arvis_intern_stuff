from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from data_models import FaceBox, FaceDetection, Point


class YuNetDetector:
    """YuNet modelini yükler ve görüntüdeki yüzleri tespit eder."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(f"Yüz modeli bulunamadı: {model_path}")

        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Bir kamera karesindeki bütün yüzleri anlamlandırarak döndürür."""

        frame_height, frame_width = frame.shape[:2]
        self._detector.setInputSize((frame_width, frame_height))

        _, raw_detections = self._detector.detect(frame)
        detected_faces: list[FaceDetection] = []

        if raw_detections is None:
            return detected_faces

        for raw_detection in raw_detections:
            detected_face = self._parse_detection(raw_detection)
            detected_faces.append(detected_face)

        return detected_faces

    @staticmethod
    def _parse_detection(raw_detection: np.ndarray) -> FaceDetection:
        """YuNet'in 15 sayılık ham çıktısını nesnelere dönüştürür."""

        (
            face_x,
            face_y,
            face_width,
            face_height,
            right_eye_x,
            right_eye_y,
            left_eye_x,
            left_eye_y,
            nose_tip_x,
            nose_tip_y,
            right_mouth_corner_x,
            right_mouth_corner_y,
            left_mouth_corner_x,
            left_mouth_corner_y,
            confidence_score,
        ) = raw_detection

        return FaceDetection(
            box=FaceBox(
                x=int(face_x),
                y=int(face_y),
                width=int(face_width),
                height=int(face_height),
            ),
            right_eye=Point(x=int(right_eye_x), y=int(right_eye_y)),
            left_eye=Point(x=int(left_eye_x), y=int(left_eye_y)),
            nose_tip=Point(x=int(nose_tip_x), y=int(nose_tip_y)),
            right_mouth_corner=Point(
                x=int(right_mouth_corner_x),
                y=int(right_mouth_corner_y),
            ),
            left_mouth_corner=Point(
                x=int(left_mouth_corner_x),
                y=int(left_mouth_corner_y),
            ),
            confidence_score=float(confidence_score),
        )
