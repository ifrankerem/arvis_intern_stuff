from __future__ import annotations

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from data_models import FaceBox, FaceLandmarkDetection, Point


LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
)
NOSE_TIP_INDEX = 1
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
UPPER_LIP_CENTER_INDEX = 13
LOWER_LIP_CENTER_INDEX = 14


class MediaPipeFaceLandmarker:
    """Yüz, yoğun landmark, blendshape ve kafa pozu tespiti yapar."""

    def __init__(
        self,
        model_path: Path,
        maximum_faces: int = 2,
        mirrored_input: bool = False,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(f"Yüz landmark modeli bulunamadı: {model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=maximum_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )
        self._mirrored_input = mirrored_input

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> list[FaceLandmarkDetection]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_pipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        raw_result = self._landmarker.detect_for_video(
            media_pipe_image,
            timestamp_ms,
        )

        detected_faces: list[FaceLandmarkDetection] = []

        for face_index, normalized_landmarks in enumerate(
            raw_result.face_landmarks
        ):
            detected_face = self._parse_face(
                normalized_landmarks=normalized_landmarks,
                blendshape_categories=raw_result.face_blendshapes[face_index],
                transformation_matrix=(
                    raw_result.facial_transformation_matrixes[face_index]
                ),
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                mirrored_input=self._mirrored_input,
            )
            detected_faces.append(detected_face)

        return detected_faces

    def close(self) -> None:
        self._landmarker.close()

    @staticmethod
    def _parse_face(
        normalized_landmarks: list[object],
        blendshape_categories: list[object],
        transformation_matrix: np.ndarray,
        frame_width: int,
        frame_height: int,
        mirrored_input: bool,
    ) -> FaceLandmarkDetection:
        landmarks: list[Point] = []

        for normalized_landmark in normalized_landmarks:
            point = Point(
                x=int(normalized_landmark.x * frame_width),
                y=int(normalized_landmark.y * frame_height),
            )
            landmarks.append(point)

        left = min(point.x for point in landmarks)
        top = min(point.y for point in landmarks)
        right = max(point.x for point in landmarks)
        bottom = max(point.y for point in landmarks)

        face_box = FaceBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        ).clamp_to_frame(frame_width, frame_height)

        blendshape_scores = {
            category.category_name: float(category.score)
            for category in blendshape_categories
        }
        pitch, yaw, roll = (
            MediaPipeFaceLandmarker._rotation_matrix_to_euler_angles(
                transformation_matrix
            )
        )

        # Ayna görüntüsünde kullanıcının sağı ve solu model açısından ters döner.
        user_yaw = -yaw if mirrored_input else yaw

        return FaceLandmarkDetection(
            box=face_box,
            landmarks=tuple(landmarks),
            left_eye_landmarks=tuple(
                landmarks[index] for index in LEFT_EYE_INDICES
            ),
            right_eye_landmarks=tuple(
                landmarks[index] for index in RIGHT_EYE_INDICES
            ),
            mouth_landmarks=tuple(
                landmarks[index] for index in MOUTH_INDICES
            ),
            nose_tip=landmarks[NOSE_TIP_INDEX],
            left_mouth_corner=landmarks[LEFT_MOUTH_CORNER_INDEX],
            right_mouth_corner=landmarks[RIGHT_MOUTH_CORNER_INDEX],
            upper_lip_center=landmarks[UPPER_LIP_CENTER_INDEX],
            lower_lip_center=landmarks[LOWER_LIP_CENTER_INDEX],
            left_eye_blink_score=blendshape_scores.get("eyeBlinkLeft", 0.0),
            right_eye_blink_score=blendshape_scores.get("eyeBlinkRight", 0.0),
            jaw_open_score=blendshape_scores.get("jawOpen", 0.0),
            mouth_left_score=blendshape_scores.get("mouthLeft", 0.0),
            mouth_right_score=blendshape_scores.get("mouthRight", 0.0),
            yaw_degrees=user_yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
        )

    @staticmethod
    def _rotation_matrix_to_euler_angles(
        transformation_matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        rotation = transformation_matrix[:3, :3]
        horizontal_length = math.sqrt(
            rotation[0, 0] ** 2 + rotation[1, 0] ** 2
        )

        pitch = math.degrees(
            math.atan2(rotation[2, 1], rotation[2, 2])
        )
        yaw = math.degrees(
            math.atan2(-rotation[2, 0], horizontal_length)
        )
        roll = math.degrees(
            math.atan2(rotation[1, 0], rotation[0, 0])
        )

        return pitch, yaw, roll
