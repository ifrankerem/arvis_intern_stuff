"""Herhangi bir yuz tespit modeli yuklemeden calisan pre-control modu."""

from datetime import datetime

import cv2
import numpy as np

import config
from data_models import FaceBox, FrameProcessingResult
from moire_pre_control import MoirePeriodicPatternPreController
from pre_control import GlobalFFTPreController


class ModelFreePreControlApplication:
    """Sabit bir ekran kilavuzunu ROI kabul ederek FFT analizi yapar."""

    def __init__(self):
        self.fft_pre_controller = GlobalFFTPreController()
        self.moire_pre_controller = MoirePeriodicPatternPreController()
        self.latest_pre_control_results = {}
        self.latest_face_image = None
        self.latest_fft_visualization = None
        self.latest_analysis_result = None
        self.is_closed = False

    def process_frame(self, camera_frame):
        analysis_frame = self.prepare_frame(camera_frame)
        guide_box = self.create_guide_box(analysis_frame)
        self.latest_face_image = None
        self.latest_fft_visualization = None

        fft_result = self.fft_pre_controller.analyze_face_box(
            analysis_frame,
            guide_box,
        )
        self.latest_pre_control_results["fft"] = fft_result

        # Yeni FFT hesaplama: Moire denetimi ayni kare icin mevcut FFT
        # denetleyicisinin urettigi kaydirilmis sayisal spektrumlari kullanir.
        moire_result = self.moire_pre_controller.analyze(
            self.fft_pre_controller.latest_power_spectrum,
            self.fft_pre_controller.latest_log_spectrum,
            guide_box,
            fft_result.quality_status,
        )
        self.latest_pre_control_results["moire"] = moire_result

        face_crop = self.fft_pre_controller.latest_face_crop
        log_spectrum = self.fft_pre_controller.latest_log_spectrum
        if face_crop is not None and log_spectrum is not None:
            self.latest_face_image = face_crop.copy()
            self.latest_fft_visualization = (
                self.create_fft_visualization(log_spectrum)
            )

        display_frame = analysis_frame.copy()
        self.draw_guide(
            display_frame,
            guide_box,
            fft_result,
            moire_result,
        )

        return FrameProcessingResult(
            display_frame,
            self.latest_face_image,
            None,
        )

    def create_fft_visualization(self, log_spectrum):
        """Mevcut log spektrumunun analizden bagimsiz ekran kopyasi."""
        normalized = cv2.normalize(
            log_spectrum.copy(),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
        spectrum_uint8 = normalized.astype(np.uint8)
        visualization = cv2.cvtColor(
            spectrum_uint8,
            cv2.COLOR_GRAY2BGR,
        )
        self.draw_fft_annotations(visualization)
        return visualization

    def draw_fft_annotations(self, visualization):
        height, width = visualization.shape[:2]
        center_x = width // 2
        center_y = height // 2

        cv2.drawMarker(
            visualization,
            (center_x, center_y),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )
        self.draw_label(
            visualization,
            "Low Frequency",
            (center_x + 8, max(14, center_y - 8)),
        )
        self.draw_label(
            visualization,
            "High Frequency",
            (7, height - 9),
        )

    def draw_label(self, image, text, position):
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )

    def save_fft_sample(self):
        face_crop = self.latest_face_image
        fft_visualization = self.latest_fft_visualization
        if face_crop is None or fft_visualization is None:
            print("Gecerli yuz crop'u veya FFT goruntusu yok; kaydedilmedi.")
            return False

        config.FFT_SAMPLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_path = (
            config.FFT_SAMPLE_DIRECTORY / ("face_" + timestamp + ".png")
        ).resolve()
        fft_path = (
            config.FFT_SAMPLE_DIRECTORY / ("fft_" + timestamp + ".png")
        ).resolve()

        face_was_saved = cv2.imwrite(str(face_path), face_crop)
        fft_was_saved = cv2.imwrite(str(fft_path), fft_visualization)
        if not face_was_saved or not fft_was_saved:
            print("FFT ornek dosyalari kaydedilemedi.")
            return False

        print("Yuz crop'u kaydedildi: " + str(face_path))
        print("FFT goruntusu kaydedildi: " + str(fft_path))
        return True

    def prepare_frame(self, camera_frame):
        if config.MIRROR_CAMERA_IMAGE:
            return cv2.flip(camera_frame, 1)
        return camera_frame

    def create_guide_box(self, frame):
        frame_height, frame_width = frame.shape[:2]
        side = int(min(frame_width, frame_height) * 0.58)
        center_x = frame_width // 2
        center_y = int(frame_height * 0.48)

        return FaceBox(
            center_x - side // 2,
            center_y - side // 2,
            side,
            side,
        ).clamp_to_frame(frame_width, frame_height)

    def draw_guide(
        self,
        frame,
        guide_box,
        fft_result,
        moire_result,
    ):
        center = (
            guide_box.x + guide_box.width // 2,
            guide_box.y + guide_box.height // 2,
        )
        axes = (
            guide_box.width // 2,
            guide_box.height // 2,
        )
        guide_color = (0, 255, 0) if fft_result.passed else (0, 165, 255)
        if fft_result.warning:
            guide_color = (0, 0, 255)

        cv2.ellipse(
            frame,
            center,
            axes,
            0,
            0,
            360,
            guide_color,
            3,
        )
        self.draw_text(
            frame,
            "MODEL-FREE PRE-CONTROL",
            35,
            (255, 255, 255),
            0.75,
            2,
        )
        self.draw_text(
            frame,
            "Yuzunu kilavuzun icine yerlestir",
            68,
            guide_color,
            0.65,
            2,
        )

        score_text = "FFT skoru: hazirlaniyor"
        if fft_result.score is not None:
            score_text = "FFT skoru: %d/100" % round(fft_result.score)
        self.draw_text(
            frame,
            score_text,
            101,
            (255, 255, 255),
            0.60,
            2,
        )
        self.draw_text(
            frame,
            "Durum: " + fft_result.status,
            132,
            guide_color,
            0.55,
            2,
        )

        moire_color = (0, 255, 0)
        if moire_result.status in ("Analysis Uncertain", "Unavailable"):
            moire_color = (0, 165, 255)
        elif moire_result.status in (
            "Suspicious",
            "Possible Screen Replay",
        ):
            moire_color = (0, 0, 255)

        moire_text = "Moire: " + moire_result.status
        if moire_result.score is not None:
            moire_text += " | %d/100" % round(moire_result.score)
        self.draw_text(
            frame,
            moire_text,
            163,
            moire_color,
            0.55,
            2,
        )

        if moire_result.warning:
            self.draw_moire_warning(frame)
        elif fft_result.warning:
            self.draw_warning(frame, fft_result.attack_type)

    def draw_moire_warning(self, frame):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        top = max(195, frame_height - 165)
        cv2.rectangle(
            overlay,
            (0, top),
            (frame_width, frame_height),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.draw_text(
            frame,
            "WARNING: Possible screen replay / periodic display pattern",
            top + 48,
            (255, 255, 255),
            0.55,
            2,
        )
        self.draw_text(
            frame,
            "Suspicion signal only - not a definitive fake decision",
            top + 88,
            (255, 255, 255),
            0.48,
            1,
        )

    def draw_warning(self, frame, attack_type):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        top = max(165, frame_height - 150)
        cv2.rectangle(
            overlay,
            (0, top),
            (frame_width, frame_height),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.draw_text(
            frame,
            "UYARI: SUPHELI FREKANS YAPISI",
            top + 48,
            (255, 255, 255),
            0.75,
            2,
        )
        self.draw_text(
            frame,
            "Olasi saldiri: " + attack_type,
            top + 88,
            (255, 255, 255),
            0.58,
            1,
        )

    def draw_text(self, frame, text, y, color, scale, thickness):
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

    def shutdown(self):
        if self.is_closed:
            return
        self.fft_pre_controller.reset()
        self.moire_pre_controller.reset()
        self.is_closed = True
