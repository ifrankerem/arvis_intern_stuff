import time
from datetime import datetime

import cv2

import config
from alignment_analyzer import AlignmentAnalyzer
from data_models import (
    FaceAnalysisResult,
    FaceFrameAnalysis,
    FrameProcessingResult,
)
from face_landmarker import FaceLandmarker
from json_result_writer import JsonResultWriter
from quality_analyzer import QualityAnalyzer


class FaceQualityApplication:
    """Kamera, analiz, ekran ve kayit akislarini yonetir."""

    def __init__(self):
        self.face_landmarker = FaceLandmarker(
            config.MODEL_PATH,
            config.MAXIMUM_FACE_COUNT,
            config.MIRROR_CAMERA_IMAGE,
        )
        self.quality_analyzer = QualityAnalyzer()
        self.alignment_analyzer = AlignmentAnalyzer()
        self.json_writer = JsonResultWriter(config.OUTPUT_DIRECTORY)

        self.started_at = time.perf_counter()
        self.latest_face_image = None
        self.latest_analysis_result = None

    def run(self):
        camera = cv2.VideoCapture(config.CAMERA_INDEX)

        if not camera.isOpened():
            self.face_landmarker.close()
            raise RuntimeError("Kamera acilamadi.")

        self.print_instructions()

        try:
            should_exit = False

            while not should_exit:
                frame_was_read, camera_frame = camera.read()

                if not frame_was_read or camera_frame is None:
                    print("Kameradan goruntu alinamadi.")
                    break

                frame_result = self.process_frame(camera_frame)

                self.latest_face_image = frame_result.face_image
                self.latest_analysis_result = frame_result.analysis_result

                cv2.imshow(
                    "Face Quality + Alignment",
                    frame_result.display_frame,
                )

                pressed_key = cv2.waitKey(1) & 0xFF
                should_exit = self.handle_key(pressed_key)
        finally:
            camera.release()
            self.face_landmarker.close()
            cv2.destroyAllWindows()

        self.save_final_json_result()

    def process_frame(self, camera_frame):
        analysis_frame = self.prepare_frame(camera_frame)
        timestamp_ms = time.monotonic_ns() // 1_000_000

        detected_faces = self.face_landmarker.detect_faces(
            analysis_frame,
            timestamp_ms,
        )

        face_analysis = self.analyze_single_face(
            analysis_frame,
            detected_faces,
        )

        analysis_result = self.create_analysis_result(
            len(detected_faces),
            face_analysis,
        )

        display_frame = analysis_frame.copy()
        self.draw_detected_faces(display_frame, detected_faces)
        self.draw_analysis_information(
            display_frame,
            len(detected_faces),
            face_analysis,
        )

        return FrameProcessingResult(
            display_frame,
            face_analysis.face_image,
            analysis_result,
        )

    def prepare_frame(self, camera_frame):
        if config.MIRROR_CAMERA_IMAGE:
            return cv2.flip(camera_frame, 1)

        return camera_frame

    def analyze_single_face(self, frame, detected_faces):
        if len(detected_faces) != 1:
            return FaceFrameAnalysis(None, None, None, None)

        detected_face = detected_faces[0]
        alignment = self.alignment_analyzer.analyze(detected_face)
        quality = None
        face_image = None

        try:
            quality = self.quality_analyzer.analyze(
                frame,
                detected_face.box,
            )
            face_image = self.crop_face(frame, detected_face.box)
        except (ValueError, cv2.error) as error:
            print("Kalite analizi yapilamadi: " + str(error))

        return FaceFrameAnalysis(
            detected_face,
            quality,
            alignment,
            face_image,
        )

    def crop_face(self, frame, face_box):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        safe_box = face_box.clamp_to_frame(
            frame_width,
            frame_height,
        )

        face_image = frame[
            safe_box.y : safe_box.y + safe_box.height,
            safe_box.x : safe_box.x + safe_box.width,
        ]

        return face_image.copy()

    def create_analysis_result(self, number_of_faces, face_analysis):
        if number_of_faces == 0:
            quality_status = "NO_FACE"
            alignment_status = "NO_FACE"
        elif number_of_faces > 1:
            quality_status = "MULTIPLE_FACES"
            alignment_status = "MULTIPLE_FACES"
        else:
            quality_status = self.get_quality_status(face_analysis.quality)
            alignment_status = self.get_alignment_status(
                face_analysis.alignment
            )

        processing_time_ms = int(
            (time.perf_counter() - self.started_at) * 1000
        )

        return FaceAnalysisResult(
            number_of_faces > 0,
            quality_status,
            alignment_status,
            face_analysis.alignment,
            processing_time_ms,
        )

    def get_quality_status(self, quality):
        if quality is not None and quality.is_acceptable():
            return "ACCEPTABLE"

        return "UNACCEPTABLE"

    def get_alignment_status(self, alignment):
        if alignment is not None and alignment.is_acceptable():
            return "ACCEPTABLE"

        return "UNACCEPTABLE"

    def draw_detected_faces(self, frame, detected_faces):
        for detected_face in detected_faces:
            self.draw_face_box(frame, detected_face)
            self.draw_eye_landmarks(frame, detected_face)
            self.draw_mouth_landmarks(frame, detected_face)
            self.draw_nose_tip(frame, detected_face)

    def draw_face_box(self, frame, detected_face):
        top_left = detected_face.box.get_top_left().to_tuple()
        bottom_right = detected_face.box.get_bottom_right().to_tuple()

        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            (255, 255, 0),
            2,
        )

    def draw_eye_landmarks(self, frame, detected_face):
        for landmark in detected_face.left_eye_landmarks:
            self.draw_point(frame, landmark, (0, 255, 255), 2)

        for landmark in detected_face.right_eye_landmarks:
            self.draw_point(frame, landmark, (0, 255, 255), 2)

    def draw_mouth_landmarks(self, frame, detected_face):
        for landmark in detected_face.mouth_landmarks:
            self.draw_point(frame, landmark, (0, 255, 0), 2)

    def draw_nose_tip(self, frame, detected_face):
        self.draw_point(
            frame,
            detected_face.nose_tip,
            (255, 0, 255),
            3,
        )

    def draw_point(self, frame, point, color, radius):
        cv2.circle(
            frame,
            point.to_tuple(),
            radius,
            color,
            -1,
        )

    def draw_analysis_information(
        self,
        frame,
        number_of_faces,
        face_analysis,
    ):
        quality_message = self.quality_analyzer.get_message(
            number_of_faces,
            face_analysis.quality,
        )
        alignment_message = self.alignment_analyzer.get_message(
            face_analysis.alignment
        )

        self.draw_text(
            frame,
            quality_message.text,
            35,
            quality_message.color.to_tuple(),
            0.75,
            2,
        )
        self.draw_text(
            frame,
            alignment_message.text,
            70,
            alignment_message.color.to_tuple(),
            0.70,
            2,
        )

        self.draw_metrics(frame, face_analysis)

        self.draw_text(
            frame,
            "q: cikis | s: yuz | j: JSON",
            frame.shape[0] - 20,
            (255, 255, 255),
            0.55,
            1,
        )

    def draw_metrics(self, frame, face_analysis):
        quality = face_analysis.quality
        alignment = face_analysis.alignment

        if quality is None or alignment is None:
            return

        head_text = (
            "Yaw: %.1f | Pitch: %.1f | Roll: %.1f"
            % (
                alignment.yaw_degrees,
                alignment.pitch_degrees,
                alignment.roll_degrees,
            )
        )
        mouth_text = (
            "Agiz acikligi: %.2f | Agiz acisi: %.1f"
            % (
                alignment.mouth_open_score,
                alignment.mouth_angle_degrees,
            )
        )
        quality_text = (
            "Blur: %.1f | Isik: %.1f | Yuz: %.1f%%"
            % (
                quality.blur_score,
                quality.brightness,
                quality.face_area_ratio * 100.0,
            )
        )

        self.draw_text(frame, head_text, 100, (255, 255, 255), 0.50, 1)
        self.draw_text(frame, mouth_text, 125, (255, 255, 255), 0.50, 1)
        self.draw_text(frame, quality_text, 150, (255, 255, 255), 0.50, 1)

    def draw_text(self, frame, text, y, color, font_scale, thickness):
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )

    def handle_key(self, pressed_key):
        if pressed_key == ord("q"):
            return True

        if pressed_key == ord("j"):
            self.save_json_result()

        if pressed_key == ord("s"):
            self.save_face_image()

        return False

    def save_json_result(self):
        if self.latest_analysis_result is None:
            print("Kaydedilecek analiz sonucu yok.")
            return

        output_path = self.json_writer.write(
            self.latest_analysis_result
        )
        print("JSON sonucu kaydedildi: " + str(output_path))

    def save_face_image(self):
        if self.latest_face_image is None:
            print("Kaydedilecek tek bir yuz bulunamadi.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = "face_" + timestamp + ".jpg"
        output_path = config.OUTPUT_DIRECTORY / file_name

        image_was_saved = cv2.imwrite(
            str(output_path),
            self.latest_face_image,
        )

        if image_was_saved:
            print("Yuz kaydedildi: " + str(output_path))
        else:
            print("Yuz dosyasi kaydedilemedi.")

    def save_final_json_result(self):
        if self.latest_analysis_result is None:
            return

        output_path = self.json_writer.write(
            self.latest_analysis_result
        )
        print("Son JSON sonucu kaydedildi: " + str(output_path))

    def print_instructions(self):
        print("Program basladi.")
        print("Cikis: q")
        print("Yuz kirpimini kaydet: s")
        print("JSON sonucunu kaydet: j")
