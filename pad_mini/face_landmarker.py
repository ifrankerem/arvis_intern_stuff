import math

import cv2
import mediapipe as mp

from data_models import (
    FaceBox,
    FaceLandmarkDetection,
    HeadRotation,
    Point,
)


LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
)

NOSE_TIP_INDEX = 1
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
UPPER_LIP_CENTER_INDEX = 13
LOWER_LIP_CENTER_INDEX = 14


class FaceROIDetection:
    """PreControl ROI takibi icin gereken en kucuk landmark sonucu."""

    def __init__(self, box, landmarks):
        self.box = box
        self.landmarks = landmarks


class FaceLandmarker:
    """MediaPipe modelini calistirir ve yuz verilerini donusturur."""

    def __init__(
        self,
        model_path,
        maximum_faces,
        mirrored_input,
        minimum_detection_confidence=0.5,
        minimum_presence_confidence=0.5,
        minimum_tracking_confidence=0.5,
        roi_only=False,
    ):
        if not model_path.exists():
            message = "Yuz landmark modeli bulunamadi: " + str(model_path)
            raise RuntimeError(message)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        )

        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=maximum_faces,
            min_face_detection_confidence=minimum_detection_confidence,
            min_face_presence_confidence=minimum_presence_confidence,
            min_tracking_confidence=minimum_tracking_confidence,
            output_face_blendshapes=not roi_only,
            output_facial_transformation_matrixes=not roi_only,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                landmarker_options
            )
        )
        self.mirrored_input = mirrored_input
        self.roi_only = bool(roi_only)

    def detect_faces(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        raw_result = self.landmarker.detect_for_video(
            mediapipe_image,
            timestamp_ms,
        )

        detected_faces = []

        for face_index in range(len(raw_result.face_landmarks)):
            normalized_landmarks = raw_result.face_landmarks[face_index]
            if self.roi_only:
                landmarks = self.convert_landmarks_to_pixels(
                    normalized_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                )
                detected_faces.append(
                    FaceROIDetection(
                        self.create_face_box(
                            landmarks,
                            frame.shape[1],
                            frame.shape[0],
                        ),
                        landmarks,
                    )
                )
                continue
            blendshapes = raw_result.face_blendshapes[face_index]
            transformation_matrix = (
                raw_result.facial_transformation_matrixes[face_index]
            )

            detected_face = self.convert_raw_face(
                normalized_landmarks,
                blendshapes,
                transformation_matrix,
                frame.shape[1],
                frame.shape[0],
            )

            detected_faces.append(detected_face)

        return detected_faces

    def convert_raw_face(
        self,
        normalized_landmarks,
        blendshapes,
        transformation_matrix,
        frame_width,
        frame_height,
    ):
        landmarks = self.convert_landmarks_to_pixels(
            normalized_landmarks,
            frame_width,
            frame_height,
        )

        face_box = self.create_face_box(
            landmarks,
            frame_width,
            frame_height,
        )

        blendshape_scores = self.create_blendshape_map(blendshapes)
        head_rotation = self.calculate_head_rotation(
            transformation_matrix
        )

        if self.mirrored_input:
            head_rotation.yaw_degrees = -head_rotation.yaw_degrees

        left_eye_landmarks = self.select_landmarks(
            landmarks,
            LEFT_EYE_INDICES,
        )
        right_eye_landmarks = self.select_landmarks(
            landmarks,
            RIGHT_EYE_INDICES,
        )
        mouth_landmarks = self.select_landmarks(
            landmarks,
            MOUTH_INDICES,
        )

        return FaceLandmarkDetection(
            face_box,
            landmarks,
            left_eye_landmarks,
            right_eye_landmarks,
            mouth_landmarks,
            landmarks[NOSE_TIP_INDEX],
            landmarks[LEFT_MOUTH_CORNER_INDEX],
            landmarks[RIGHT_MOUTH_CORNER_INDEX],
            landmarks[UPPER_LIP_CENTER_INDEX],
            landmarks[LOWER_LIP_CENTER_INDEX],
            self.get_score(blendshape_scores, "eyeBlinkLeft"),
            self.get_score(blendshape_scores, "eyeBlinkRight"),
            self.get_score(blendshape_scores, "jawOpen"),
            self.get_score(blendshape_scores, "mouthLeft"),
            self.get_score(blendshape_scores, "mouthRight"),
            head_rotation.yaw_degrees,
            head_rotation.pitch_degrees,
            head_rotation.roll_degrees,
        )

    def convert_landmarks_to_pixels(
        self,
        normalized_landmarks,
        frame_width,
        frame_height,
    ):
        pixel_landmarks = []

        for normalized_landmark in normalized_landmarks:
            pixel_x = int(normalized_landmark.x * frame_width)
            pixel_y = int(normalized_landmark.y * frame_height)
            pixel_landmarks.append(Point(pixel_x, pixel_y))

        return pixel_landmarks

    def create_face_box(self, landmarks, frame_width, frame_height):
        first_landmark = landmarks[0]
        left = first_landmark.x
        top = first_landmark.y
        right = first_landmark.x
        bottom = first_landmark.y

        for landmark in landmarks:
            if landmark.x < left:
                left = landmark.x
            if landmark.y < top:
                top = landmark.y
            if landmark.x > right:
                right = landmark.x
            if landmark.y > bottom:
                bottom = landmark.y

        face_box = FaceBox(
            left,
            top,
            right - left,
            bottom - top,
        )

        return face_box.clamp_to_frame(frame_width, frame_height)

    def select_landmarks(self, all_landmarks, selected_indices):
        selected_landmarks = []

        for landmark_index in selected_indices:
            selected_landmarks.append(all_landmarks[landmark_index])

        return selected_landmarks

    def create_blendshape_map(self, blendshapes):
        scores = {}

        for blendshape in blendshapes:
            name = blendshape.category_name
            score = float(blendshape.score)
            scores[name] = score

        return scores

    def get_score(self, scores, score_name):
        if score_name in scores:
            return scores[score_name]

        return 0.0

    def calculate_head_rotation(self, transformation_matrix):
        rotation = transformation_matrix[:3, :3]

        horizontal_length = math.sqrt(
            rotation[0, 0] ** 2
            + rotation[1, 0] ** 2
        )

        pitch_radians = math.atan2(
            rotation[2, 1],
            rotation[2, 2],
        )
        yaw_radians = math.atan2(
            -rotation[2, 0],
            horizontal_length,
        )
        roll_radians = math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )

        return HeadRotation(
            math.degrees(pitch_radians),
            math.degrees(yaw_radians),
            math.degrees(roll_radians),
        )

    def close(self):
        self.landmarker.close()
