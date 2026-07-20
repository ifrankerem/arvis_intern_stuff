"""Yuz ROI'si icin Global FFT tabanli sunum-saldirisi on kontrolu.

Bu dosya bilerek uygulamanin geri kalanindan bagimsiz tutulur. Calistirildiginda
mevcut ``FaceQualityApplication`` sinifini genisletir; dolayisiyla var olan yuz
tespiti, kalite/hizalama analizi ve kayit davranislari degismeden kalir.
"""

from collections import deque
import time

import cv2
import numpy as np

from data_models import FrameProcessingResult
from face_quality_application import FaceQualityApplication


class FFTAnalysisResult:
    """Bir kare icin FFT on kontrolunun ekranda kullanilan sonucu."""

    def __init__(
        self,
        score,
        status,
        attack_type,
        quality_status,
        warning,
        passed,
        metrics=None,
    ):
        self.score = score
        self.status = status
        self.attack_type = attack_type
        self.quality_status = quality_status
        self.warning = warning
        self.passed = passed
        self.metrics = metrics or {}


class GlobalFFTPreController:
    """Sadece tespit edilmis yuz bolgesinin 2B FFT spektrumunu inceler.

    Esikler ampirik baslangic degerleridir. Kamera, mesafe ve isik kosullarina
    gore gercek canli/saldiri ornekleriyle kalibre edilmelidir. Bu sinifin
    sonucu tek basina kesin bir canlilik karari degildir.
    """

    STANDARD_FACE_SIZE = 256
    HISTORY_SIZE = 12
    MINIMUM_VALID_FRAMES = 6
    ATTACK_SCORE_THRESHOLD = 65.0
    RELEASE_SCORE_THRESHOLD = 52.0
    REQUIRED_HIGH_FRAMES = 4

    MINIMUM_FACE_SIDE = 96
    MINIMUM_FACE_AREA_RATIO = 0.035
    MINIMUM_BLUR_SCORE = 55.0
    MINIMUM_BRIGHTNESS = 45.0
    MAXIMUM_BRIGHTNESS = 215.0
    FRAME_EDGE_MARGIN_RATIO = 0.01

    def __init__(self):
        self.score_history = deque(maxlen=self.HISTORY_SIZE)
        self.attack_history = deque(maxlen=self.HISTORY_SIZE)
        self.high_score_streak = 0
        self.low_score_streak = 0
        self.invalid_frame_streak = 0
        self.warning_is_active = False

        coordinates = np.linspace(-1.0, 1.0, self.STANDARD_FACE_SIZE)
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        self.radius_map = np.sqrt(grid_x * grid_x + grid_y * grid_y)

        hann_1d = np.hanning(self.STANDARD_FACE_SIZE)
        self.hann_window = np.outer(hann_1d, hann_1d).astype(np.float32)

    def ft_pre_control(self, frame, detected_faces):
        """FFT on kontrolunu calistirir ve stabilize edilmis sonucu dondurur.

        Yeniden kullanim icin gereken minimal girisler kamera karesi ve mevcut
        landmarker'in urettigi yuz listesidir. FFT hicbir zaman tam kareye
        uygulanmaz.
        """
        if len(detected_faces) != 1:
            self._register_invalid_frame()
            reason = "one detected face required"
            if len(detected_faces) > 1:
                reason = "multiple faces detected"
            return self._unavailable_result(reason)

        face_box = detected_faces[0].box
        quality_reason, face_region = self._get_quality_checked_face(
            frame,
            face_box,
        )

        if quality_reason is not None:
            self._register_invalid_frame()
            return self._unavailable_result(quality_reason)

        standardized_face = self._standardize_face(face_region)
        score, metrics = self._calculate_fft_score(standardized_face)
        attack_type = self._estimate_attack_type(metrics)
        return self._stabilize_result(score, attack_type, metrics)

    def reset(self):
        self.score_history.clear()
        self.attack_history.clear()
        self.high_score_streak = 0
        self.low_score_streak = 0
        self.invalid_frame_streak = 0
        self.warning_is_active = False

    def _get_quality_checked_face(self, frame, face_box):
        frame_height, frame_width = frame.shape[:2]
        safe_box = face_box.clamp_to_frame(frame_width, frame_height)

        if safe_box.width < self.MINIMUM_FACE_SIDE:
            return "face too small", None
        if safe_box.height < self.MINIMUM_FACE_SIDE:
            return "face too small", None

        frame_area = float(frame_width * frame_height)
        if safe_box.get_area() / frame_area < self.MINIMUM_FACE_AREA_RATIO:
            return "face too small", None

        edge_x = int(frame_width * self.FRAME_EDGE_MARGIN_RATIO)
        edge_y = int(frame_height * self.FRAME_EDGE_MARGIN_RATIO)
        touches_frame_edge = (
            safe_box.x <= edge_x
            or safe_box.y <= edge_y
            or safe_box.x + safe_box.width >= frame_width - edge_x
            or safe_box.y + safe_box.height >= frame_height - edge_y
        )
        if touches_frame_edge:
            return "face partially outside frame", None

        face_region = frame[
            safe_box.y : safe_box.y + safe_box.height,
            safe_box.x : safe_box.x + safe_box.width,
        ]
        if face_region.size == 0:
            return "empty face crop", None

        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        brightness = float(face_gray.mean())
        blur_score = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

        if blur_score < self.MINIMUM_BLUR_SCORE:
            return "face is blurred", None
        if brightness < self.MINIMUM_BRIGHTNESS:
            return "face is too dark", None
        if brightness > self.MAXIMUM_BRIGHTNESS:
            return "face is overexposed", None

        return None, face_region

    def _standardize_face(self, face_region):
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        interpolation = cv2.INTER_AREA
        if min(face_gray.shape[:2]) < self.STANDARD_FACE_SIZE:
            interpolation = cv2.INTER_CUBIC

        resized_face = cv2.resize(
            face_gray,
            (self.STANDARD_FACE_SIZE, self.STANDARD_FACE_SIZE),
            interpolation=interpolation,
        ).astype(np.float32)

        resized_face -= float(resized_face.mean())
        standard_deviation = float(resized_face.std())
        if standard_deviation > 1e-6:
            resized_face /= standard_deviation

        # Pencereleme, dikdortgen kirpimin sinirinda FFT'nin uretecegi yapay
        # yuksek frekanslari bastirir.
        return resized_face * self.hann_window

    def _calculate_fft_score(self, standardized_face):
        fft_result = np.fft.fftshift(np.fft.fft2(standardized_face))
        power_spectrum = np.abs(fft_result) ** 2
        log_spectrum = np.log1p(power_spectrum).astype(np.float32)

        analysis_mask = (
            (self.radius_map >= 0.08)
            & (self.radius_map <= 0.92)
        )
        mid_mask = (
            (self.radius_map >= 0.16)
            & (self.radius_map < 0.45)
        )
        high_mask = (
            (self.radius_map >= 0.45)
            & (self.radius_map <= 0.92)
        )

        total_energy = float(power_spectrum[analysis_mask].sum()) + 1e-9
        mid_energy_ratio = float(power_spectrum[mid_mask].sum()) / total_energy
        high_energy_ratio = float(power_spectrum[high_mask].sum()) / total_energy

        # Genis ve dogal 1/f spektrum arka planini cikartip ince, periyodik
        # zirveleri belirginlestirir. Gercek bir goruntunun FFT'sinde karsilikli
        # zirveler dogal olarak simetrik oldugundan tek yari-duzlem sayilir.
        spectral_background = cv2.GaussianBlur(
            log_spectrum,
            (0, 0),
            5.0,
        )
        spectral_residual = log_spectrum - spectral_background
        residual_values = spectral_residual[analysis_mask]
        residual_median = float(np.median(residual_values))
        median_deviation = float(
            np.median(np.abs(residual_values - residual_median))
        )
        robust_scale = max(1.4826 * median_deviation, 1e-6)
        residual_z = (spectral_residual - residual_median) / robust_scale

        local_maximum = residual_z >= cv2.dilate(
            residual_z,
            np.ones((3, 3), dtype=np.uint8),
        )
        half_plane = np.zeros_like(analysis_mask)
        center = self.STANDARD_FACE_SIZE // 2
        half_plane[:center, :] = True
        half_plane[center, :center] = True
        peak_mask = (
            analysis_mask
            & half_plane
            & local_maximum
            & (residual_z >= 5.5)
        )
        peak_values = residual_z[peak_mask]
        peak_count = int(peak_values.size)
        peak_strength = 0.0
        if peak_count > 0:
            strongest_peaks = np.sort(peak_values)[-12:]
            peak_strength = float(strongest_peaks.mean())

        peak_score = np.clip((peak_strength - 5.5) / 3.5, 0.0, 1.0)
        peak_score *= min(1.0, peak_count / 2.0)
        peak_count_score = np.clip((peak_count - 3.0) / 25.0, 0.0, 1.0)
        mid_energy_score = np.clip(
            (mid_energy_ratio - 0.22) / 0.30,
            0.0,
            1.0,
        )
        high_energy_score = np.clip(
            (high_energy_ratio - 0.08) / 0.22,
            0.0,
            1.0,
        )

        anomaly_score = 100.0 * (
            0.48 * peak_score
            + 0.18 * peak_count_score
            + 0.20 * mid_energy_score
            + 0.14 * high_energy_score
        )

        metrics = {
            "mid_energy_ratio": mid_energy_ratio,
            "high_energy_ratio": high_energy_ratio,
            "peak_count": peak_count,
            "peak_strength": peak_strength,
            "peak_score": float(peak_score),
        }
        return float(np.clip(anomaly_score, 0.0, 100.0)), metrics

    def _estimate_attack_type(self, metrics):
        peak_score = metrics["peak_score"]
        mid_energy = metrics["mid_energy_ratio"]
        high_energy = metrics["high_energy_ratio"]

        if high_energy >= 0.16 or metrics["peak_count"] >= 5:
            return "Printed photo"
        if peak_score >= 0.55 and mid_energy >= high_energy:
            return "Screen replay"
        return "Replay or printed photo"

    def _stabilize_result(self, score, attack_type, metrics):
        self.invalid_frame_streak = 0
        self.score_history.append(score)
        if score >= self.ATTACK_SCORE_THRESHOLD:
            self.attack_history.append(attack_type)
        else:
            self.attack_history.append("None")

        if score >= self.ATTACK_SCORE_THRESHOLD:
            self.high_score_streak += 1
            self.low_score_streak = 0
        elif score <= self.RELEASE_SCORE_THRESHOLD:
            self.low_score_streak += 1
            self.high_score_streak = 0
        else:
            self.high_score_streak = 0
            self.low_score_streak = 0

        history_is_ready = (
            len(self.score_history) >= self.MINIMUM_VALID_FRAMES
        )
        rolling_average = float(np.mean(self.score_history))
        high_frame_count = sum(
            value >= self.ATTACK_SCORE_THRESHOLD
            for value in self.score_history
        )

        should_activate_warning = (
            history_is_ready
            and self.high_score_streak >= self.REQUIRED_HIGH_FRAMES
            and high_frame_count >= self.REQUIRED_HIGH_FRAMES
            and rolling_average >= 60.0
        )
        if should_activate_warning:
            self.warning_is_active = True

        if self.warning_is_active and self.low_score_streak >= 4:
            recent_average = float(
                np.mean(list(self.score_history)[-4:])
            )
            if recent_average <= self.RELEASE_SCORE_THRESHOLD:
                self.warning_is_active = False

        if self.warning_is_active:
            stable_attack_type = self._majority_attack_type()
            status = "Suspicious frequency pattern"
            if stable_attack_type == "Screen replay":
                status = "Possible replay attack"
            elif stable_attack_type == "Printed photo":
                status = "Possible printed-photo attack"

            return FFTAnalysisResult(
                rolling_average,
                status,
                stable_attack_type,
                "Sufficient",
                "WARNING: Possible replay or printed-photo attack detected.",
                False,
                metrics,
            )

        if not history_is_ready or score >= self.ATTACK_SCORE_THRESHOLD:
            status = "Analysis uncertain"
        else:
            status = "Normal frequency structure"

        displayed_attack_type = "Unknown"
        if status == "Normal frequency structure":
            displayed_attack_type = "None"
        elif score >= self.ATTACK_SCORE_THRESHOLD:
            displayed_attack_type = attack_type

        return FFTAnalysisResult(
            rolling_average,
            status,
            displayed_attack_type,
            "Sufficient",
            "",
            status == "Normal frequency structure",
            metrics,
        )

    def _majority_attack_type(self):
        candidates = list(self.attack_history)
        screen_count = candidates.count("Screen replay")
        print_count = candidates.count("Printed photo")

        if screen_count > print_count and screen_count >= 3:
            return "Screen replay"
        if print_count > screen_count and print_count >= 3:
            return "Printed photo"
        return "Replay or printed photo"

    def _register_invalid_frame(self):
        self.invalid_frame_streak += 1
        if self.invalid_frame_streak >= 3:
            self.reset()

    def _unavailable_result(self, reason):
        if self.warning_is_active:
            return FFTAnalysisResult(
                None,
                "Suspicious frequency pattern",
                self._majority_attack_type(),
                "FFT analysis unavailable: insufficient image quality. "
                + "(" + reason + ")",
                "WARNING: Possible replay or printed-photo attack detected.",
                False,
            )

        return FFTAnalysisResult(
            None,
            "Analysis uncertain",
            "Unknown",
            "FFT analysis unavailable: insufficient image quality. "
            + "(" + reason + ")",
            "",
            False,
        )


class FFTPreControlApplication(FaceQualityApplication):
    """Mevcut uygulamaya FFT on kontrolunu minimal olarak ekler."""

    def __init__(self):
        super().__init__()
        self.fft_pre_controller = GlobalFFTPreController()
        self.latest_fft_result = None

    def ft_pre_control(self, frame, detected_faces):
        """Diger pre-controller'larin da cagirabilecegi ince giris noktasi."""
        return self.fft_pre_controller.ft_pre_control(
            frame,
            detected_faces,
        )

    def process_frame(self, camera_frame):
        analysis_frame = self.prepare_frame(camera_frame)
        timestamp_ms = time.monotonic_ns() // 1_000_000

        detected_faces = self.face_landmarker.detect_faces(
            analysis_frame,
            timestamp_ms,
        )

        # Yuz konumu belli olur olmaz, mevcut kalite/hizalama ve sonuc uretme
        # adimlarindan once FFT on kontrolu calisir.
        fft_result = self.ft_pre_control(
            analysis_frame,
            detected_faces,
        )
        self.latest_fft_result = fft_result

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
        self.draw_fft_information(display_frame, fft_result)

        return FrameProcessingResult(
            display_frame,
            face_analysis.face_image,
            analysis_result,
        )

    def draw_fft_information(self, frame, fft_result):
        score_text = "FFT anomaly score: unavailable"
        if fft_result.score is not None:
            score_text = "FFT anomaly score: %d/100" % round(
                fft_result.score
            )

        quality_color = (0, 255, 0)
        if fft_result.quality_status != "Sufficient":
            quality_color = (0, 165, 255)

        self.draw_text(frame, score_text, 185, (255, 255, 255), 0.55, 1)
        self.draw_text(
            frame,
            "Frequency status: " + fft_result.status,
            210,
            (0, 255, 0) if fft_result.passed else (0, 165, 255),
            0.55,
            2,
        )
        self.draw_text(
            frame,
            "Possible attack: " + fft_result.attack_type,
            235,
            (255, 255, 255),
            0.55,
            1,
        )
        if fft_result.quality_status == "Sufficient":
            self.draw_text(
                frame,
                "FFT quality: Sufficient",
                260,
                quality_color,
                0.48,
                1,
            )
        else:
            quality_reason = fft_result.quality_status.replace(
                "FFT analysis unavailable: insufficient image quality. ",
                "",
            )
            self.draw_text(
                frame,
                "FFT analysis unavailable:",
                260,
                quality_color,
                0.48,
                1,
            )
            self.draw_text(
                frame,
                "insufficient image quality. " + quality_reason,
                282,
                quality_color,
                0.48,
                1,
            )

        if fft_result.warning:
            self._draw_large_warning(frame, fft_result.warning)

    def _draw_large_warning(self, frame, _warning_text):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        banner_top = max(300, frame_height // 2 - 75)
        banner_bottom = min(frame_height - 45, banner_top + 150)
        cv2.rectangle(
            overlay,
            (0, banner_top),
            (frame_width, banner_bottom),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        lines = [
            "WARNING: POSSIBLE ATTACK",
            "Replay or printed-photo pattern detected",
            "FFT is an evidence signal, not mathematical certainty",
        ]
        font_scales = (1.05, 0.72, 0.55)
        thicknesses = (3, 2, 1)

        for index, line in enumerate(lines):
            scale = font_scales[index]
            thickness = thicknesses[index]
            text_size = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            )[0]
            x = max(10, (frame_width - text_size[0]) // 2)
            y = banner_top + 42 + index * 40
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thickness,
            )


def main():
    application = FFTPreControlApplication()
    application.run()


if __name__ == "__main__":
    main()
