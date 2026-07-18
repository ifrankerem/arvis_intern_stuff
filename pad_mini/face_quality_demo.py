from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from data_models import FaceQuality
from quality_analyzer import QualityAnalyzer
from yunet_detector import YuNetDetector


CAMERA_INDEX = 0
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def main() -> None:
    face_detector = YuNetDetector(model_path=MODEL_PATH)
    quality_analyzer = QualityAnalyzer()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Kamera açılamadı. CAMERA_INDEX değerini 1 veya 2 yapmayı dene."
        )

    print("Program başladı.")
    print("Çıkış: q")
    print("Yüz kırpımını kaydet: s")

    latest_face: np.ndarray | None = None

    try:
        while True:
            frame_was_read, frame = camera.read()

            if not frame_was_read or frame is None:
                print("Kameradan görüntü alınamadı.")
                break

            # Aynadaki gibi görünmesi için görüntüyü yatay çeviriyoruz.
            frame = cv2.flip(frame, 1)

            detected_faces = face_detector.detect(frame)

            quality: FaceQuality | None = None
            latest_face = None

            for detected_face in detected_faces:
                face_box = detected_face.box

                cv2.rectangle(
                    frame,
                    face_box.top_left.as_tuple(),
                    face_box.bottom_right.as_tuple(),
                    (255, 255, 0),
                    2,
                )

                for landmark_point in detected_face.landmark_points:
                    cv2.circle(
                        frame,
                        landmark_point.as_tuple(),
                        3,
                        (0, 255, 255),
                        -1,
                    )

            if len(detected_faces) == 1:
                only_face = detected_faces[0]
                frame_height, frame_width = frame.shape[:2]
                safe_face_box = only_face.box.clamp_to_frame(
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                try:
                    quality = quality_analyzer.analyze(
                        frame,
                        only_face.box,
                    )

                    latest_face = frame[
                        safe_face_box.y
                        : safe_face_box.y + safe_face_box.height,
                        safe_face_box.x
                        : safe_face_box.x + safe_face_box.width,
                    ].copy()

                    metrics_text = (
                        f"Blur: {quality.blur_score:.1f} | "
                        f"Isik: {quality.brightness:.1f} | "
                        f"Yuz orani: "
                        f"{quality.face_area_ratio * 100:.1f}%"
                    )

                    cv2.putText(
                        frame,
                        metrics_text,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (255, 255, 255),
                        2,
                    )

                except (ValueError, cv2.error) as error:
                    print(f"Kalite analizi yapılamadı: {error}")
                    quality = None

            message, color = quality_analyzer.get_message(
                number_of_faces=len(detected_faces),
                quality=quality,
            )

            cv2.putText(
                frame,
                message,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
            )

            cv2.putText(
                frame,
                "q: cikis | s: yuz kirpimini kaydet",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Face Detection + Quality Gate", frame)

            pressed_key = cv2.waitKey(1) & 0xFF

            if pressed_key == ord("q"):
                break

            if pressed_key == ord("s"):
                if latest_face is None:
                    print("Kaydedilecek tek bir yüz bulunamadı.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(f"face_{timestamp}.jpg")

                    image_was_saved = cv2.imwrite(
                        str(output_path),
                        latest_face,
                    )

                    if image_was_saved:
                        print(f"Yüz kaydedildi: {output_path.resolve()}")
                    else:
                        print("Yüz dosyası kaydedilemedi.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
