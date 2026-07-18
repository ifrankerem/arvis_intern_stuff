from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from alignment_analyzer import AlignmentAnalyzer
from data_models import (
    FaceAlignment,
    FaceAnalysisResult,
    FaceLandmarkDetection,
    FaceQuality,
)
from mediapipe_face_landmarker import MediaPipeFaceLandmarker
from quality_analyzer import QualityAnalyzer
from result_writer import JsonResultWriter


CAMERA_INDEX = 0
PROJECT_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIRECTORY / "models" / "face_landmarker.task"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"


def draw_face_detection(
    frame: np.ndarray,
    detected_face: FaceLandmarkDetection,
) -> None:
    """Yüz kutusunu, gözleri, burnu ve ağız landmark'larını çizer."""

    cv2.rectangle(
        frame,
        detected_face.box.top_left.as_tuple(),
        detected_face.box.bottom_right.as_tuple(),
        (255, 255, 0),
        2,
    )

    eye_landmarks = (
        detected_face.left_eye_landmarks
        + detected_face.right_eye_landmarks
    )

    for eye_landmark in eye_landmarks:
        cv2.circle(
            frame,
            eye_landmark.as_tuple(),
            2,
            (0, 255, 255),
            -1,
        )

    for mouth_landmark in detected_face.mouth_landmarks:
        cv2.circle(
            frame,
            mouth_landmark.as_tuple(),
            2,
            (0, 255, 0),
            -1,
        )

    cv2.circle(
        frame,
        detected_face.nose_tip.as_tuple(),
        3,
        (255, 0, 255),
        -1,
    )


def build_analysis_result(
    face_detected: bool,
    quality: FaceQuality | None,
    alignment: FaceAlignment | None,
    started_at: float,
) -> FaceAnalysisResult:
    if not face_detected:
        quality_status = "NO_FACE"
        alignment_status = "NO_FACE"
    else:
        quality_status = (
            "ACCEPTABLE"
            if quality is not None and quality.is_acceptable
            else "UNACCEPTABLE"
        )
        alignment_status = (
            "ACCEPTABLE"
            if alignment is not None and alignment.is_acceptable
            else "UNACCEPTABLE"
        )

    processing_time_ms = int((time.perf_counter() - started_at) * 1000)

    return FaceAnalysisResult(
        face_detected=face_detected,
        quality_status=quality_status,
        alignment_status=alignment_status,
        alignment=alignment,
        processing_time_ms=processing_time_ms,
    )


def draw_analysis_text(
    frame: np.ndarray,
    quality: FaceQuality | None,
    alignment: FaceAlignment | None,
    quality_message: str,
    quality_color: tuple[int, int, int],
    alignment_message: str,
) -> None:
    cv2.putText(
        frame,
        quality_message,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        quality_color,
        2,
    )

    alignment_color = (
        (0, 255, 0)
        if alignment is not None and alignment.is_acceptable
        else (0, 165, 255)
    )
    cv2.putText(
        frame,
        alignment_message,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        alignment_color,
        2,
    )

    if quality is not None and alignment is not None:
        head_metrics = (
            f"Yaw: {alignment.yaw_degrees:.1f} | "
            f"Pitch: {alignment.pitch_degrees:.1f} | "
            f"Roll: {alignment.roll_degrees:.1f}"
        )
        mouth_metrics = (
            f"Agiz acikligi: {alignment.mouth_open_score:.2f} | "
            f"Agiz acisi: {alignment.mouth_angle_degrees:.1f}"
        )
        quality_metrics = (
            f"Blur: {quality.blur_score:.1f} | "
            f"Isik: {quality.brightness:.1f} | "
            f"Yuz: {quality.face_area_ratio * 100:.1f}%"
        )

        cv2.putText(
            frame,
            head_metrics,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            mouth_metrics,
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            quality_metrics,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        "q: cikis | s: yuz | j: JSON",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def main() -> None:
    face_landmarker = MediaPipeFaceLandmarker(
        model_path=MODEL_PATH,
        mirrored_input=True,
    )
    quality_analyzer = QualityAnalyzer()
    alignment_analyzer = AlignmentAnalyzer()
    result_writer = JsonResultWriter(output_directory=OUTPUT_DIRECTORY)
    started_at = time.perf_counter()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        face_landmarker.close()
        raise RuntimeError(
            "Kamera açılamadı. CAMERA_INDEX değerini 1 veya 2 yapmayı dene."
        )

    print("Program başladı.")
    print("Çıkış: q")
    print("Yüz kırpımını kaydet: s")
    print("JSON sonucunu kaydet: j")

    latest_face_image: np.ndarray | None = None
    latest_result: FaceAnalysisResult | None = None

    try:
        while True:
            frame_was_read, camera_frame = camera.read()

            if not frame_was_read or camera_frame is None:
                print("Kameradan görüntü alınamadı.")
                break

            analysis_frame = cv2.flip(camera_frame, 1)
            timestamp_ms = time.monotonic_ns() // 1_000_000
            detected_faces = face_landmarker.detect(
                frame=analysis_frame,
                timestamp_ms=timestamp_ms,
            )

            quality: FaceQuality | None = None
            alignment: FaceAlignment | None = None
            latest_face_image = None

            if len(detected_faces) == 1:
                only_face = detected_faces[0]
                alignment = alignment_analyzer.analyze(only_face)

                try:
                    quality = quality_analyzer.analyze(
                        analysis_frame,
                        only_face.box,
                    )
                    safe_face_box = only_face.box.clamp_to_frame(
                        frame_width=analysis_frame.shape[1],
                        frame_height=analysis_frame.shape[0],
                    )
                    latest_face_image = analysis_frame[
                        safe_face_box.y
                        : safe_face_box.y + safe_face_box.height,
                        safe_face_box.x
                        : safe_face_box.x + safe_face_box.width,
                    ].copy()
                except (ValueError, cv2.error) as error:
                    print(f"Kalite analizi yapılamadı: {error}")
                    quality = None

            face_detected = len(detected_faces) == 1
            latest_result = build_analysis_result(
                face_detected=face_detected,
                quality=quality,
                alignment=alignment,
                started_at=started_at,
            )
            quality_message, quality_color = quality_analyzer.get_message(
                number_of_faces=len(detected_faces),
                quality=quality,
            )
            alignment_message = alignment_analyzer.get_message(alignment)

            display_frame = analysis_frame.copy()

            for detected_face in detected_faces:
                draw_face_detection(display_frame, detected_face)

            draw_analysis_text(
                frame=display_frame,
                quality=quality,
                alignment=alignment,
                quality_message=quality_message,
                quality_color=quality_color,
                alignment_message=alignment_message,
            )

            cv2.imshow("Face Quality + Alignment", display_frame)
            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key == ord("q"):
                break

            if pressed_key == ord("j"):
                output_path = result_writer.write(latest_result)
                print(f"JSON sonucu kaydedildi: {output_path}")

            if pressed_key == ord("s"):
                if latest_face_image is None:
                    print("Kaydedilecek tek bir yüz bulunamadı.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = OUTPUT_DIRECTORY / f"face_{timestamp}.jpg"
                    image_was_saved = cv2.imwrite(
                        str(output_path),
                        latest_face_image,
                    )

                    if image_was_saved:
                        print(f"Yüz kaydedildi: {output_path}")
                    else:
                        print("Yüz dosyası kaydedilemedi.")
    finally:
        camera.release()
        face_landmarker.close()
        cv2.destroyAllWindows()

    if latest_result is not None:
        output_path = result_writer.write(latest_result)
        print(f"Son JSON sonucu kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
