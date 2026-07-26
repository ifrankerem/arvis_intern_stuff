"""Model-free PreControl modulleri icin ortak kare ve sonuc yapilari."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from data_models import FaceBox


@dataclass(frozen=True)
class FusionScoreSummary:
    """Canonical fusion/temporal scores consumed by every output layer."""

    current_frame_score: Optional[float] = None
    rolling_median: Optional[float] = None
    temporal_percentile: Optional[float] = None
    temporal_decision_score: Optional[float] = None
    display_score: Optional[float] = None

    @classmethod
    def from_mapping(cls, values):
        values = values if isinstance(values, dict) else {}
        return cls(
            current_frame_score=cls._finite_or_none(
                values.get("current_frame_score")
            ),
            rolling_median=cls._finite_or_none(
                values.get("rolling_median")
            ),
            temporal_percentile=cls._finite_or_none(
                values.get("temporal_percentile")
            ),
            temporal_decision_score=cls._finite_or_none(
                values.get("temporal_decision_score")
            ),
            display_score=cls._finite_or_none(
                values.get("display_score")
            ),
        )

    @staticmethod
    def _finite_or_none(value):
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        return numeric_value if np.isfinite(numeric_value) else None

    def to_dict(self):
        return {
            "current_frame_score": self.current_frame_score,
            "rolling_median": self.rolling_median,
            "temporal_percentile": self.temporal_percentile,
            "temporal_decision_score": self.temporal_decision_score,
            "display_score": self.display_score,
        }


@dataclass
class ModelFreePreControlContext:
    """Bir kamera karesindeki ortak, yeniden kullanilabilir analiz verileri."""

    frame_timestamp: float
    original_frame: np.ndarray
    analysis_frame: np.ndarray
    original_high_resolution_face_crop: Optional[np.ndarray]
    aligned_face_crop: Optional[np.ndarray]
    standardized_analysis_crop: Optional[np.ndarray]
    grayscale_crop: Optional[np.ndarray]
    frame_dimensions: Tuple[int, int]
    face_dimensions: Optional[Tuple[int, int]]
    analysis_dimensions: Optional[Tuple[int, int]]
    face_bounding_box: FaceBox
    face_quality_valid: bool
    quality_reason: Optional[str]
    blur_value: Optional[float]
    brightness_value: Optional[float]
    exposure_valid: Optional[bool]
    pose_alignment_valid: Optional[bool]
    fft_result: Optional[np.ndarray]
    shifted_fft_result: Optional[np.ndarray]
    magnitude_spectrum: Optional[np.ndarray]
    power_spectrum: Optional[np.ndarray]
    log_power_spectrum: Optional[np.ndarray]
    log_magnitude_visualization: Optional[np.ndarray]
    fft_window_type: str
    alignment_applied: bool = False
    standardized_aligned_face_crop: Optional[np.ndarray] = None

    @property
    def has_valid_fft(self):
        return (
            self.face_quality_valid
            and self.fft_result is not None
            and self.shifted_fft_result is not None
            and self.magnitude_spectrum is not None
            and self.power_spectrum is not None
        )


@dataclass
class ModelFreeAnalysisResult:
    """Butun matematiksel PreControl modullerinin ortak sonucu."""

    module_name: str
    available: bool
    raw_features: Dict[str, Any] = field(default_factory=dict)
    raw_score: Optional[float] = None
    stabilized_score: Optional[float] = None
    confidence: Optional[float] = None
    status: str = "Unavailable"
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug_data: Dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False

    @property
    def display_name(self):
        return self.module_name

    @property
    def score(self):
        if not self.available:
            return None
        if self.stabilized_score is not None:
            return self.stabilized_score
        return self.raw_score

    @property
    def score_summary(self):
        """Return the canonical fusion score fields without recomputation."""
        return FusionScoreSummary.from_mapping(
            self.raw_features.get("score_summary")
        ).to_dict()

    @property
    def warning(self):
        return "\n".join(self.warnings)

    @property
    def metrics(self):
        return self.raw_features

    @property
    def attack_type(self):
        return str(self.debug_data.get("possible_attack", "none"))

    @property
    def quality_status(self):
        return str(self.debug_data.get("quality_status", "Unknown"))

    @property
    def passed(self):
        return self.available and self.status in (
            "Normal",
            "Normal frequency structure",
            "Normal spectral distribution",
            "No strong block anomaly",
            "Normal multi-scale texture",
            "Normal residual structure",
            "Normal mathematical evidence",
        )

    @classmethod
    def unavailable(
        cls,
        module_name,
        reason,
        debug_data=None,
        calibrated=False,
    ):
        details = dict(debug_data or {})
        details.setdefault("quality_status", reason)
        return cls(
            module_name=module_name,
            available=False,
            raw_score=None,
            stabilized_score=None,
            confidence=0.0,
            status="Unavailable",
            evidence=[reason],
            debug_data=details,
            calibrated=calibrated,
        )

    @classmethod
    def uncertain(
        cls,
        module_name,
        raw_features=None,
        raw_score=None,
        stabilized_score=None,
        confidence=0.0,
        evidence=None,
        debug_data=None,
        calibrated=False,
    ):
        return cls(
            module_name=module_name,
            available=True,
            raw_features=dict(raw_features or {}),
            raw_score=raw_score,
            stabilized_score=stabilized_score,
            confidence=confidence,
            status="Analysis Uncertain",
            evidence=list(evidence or []),
            debug_data=dict(debug_data or {}),
            calibrated=calibrated,
        )

    @classmethod
    def uncalibrated(
        cls,
        module_name,
        raw_features=None,
        raw_score=None,
        evidence=None,
        debug_data=None,
    ):
        return cls(
            module_name=module_name,
            available=True,
            raw_features=dict(raw_features or {}),
            raw_score=raw_score,
            stabilized_score=None,
            confidence=0.0,
            status="Uncalibrated",
            evidence=list(evidence or ["Calibration data is unavailable"]),
            debug_data=dict(debug_data or {}),
            calibrated=False,
        )


class ModelFreePreControlContextBuilder:
    """Kalite kapisini ve kare basina tek ortak FFT'yi olusturur."""

    def __init__(self):
        analysis_size = config.MODEL_FREE_ANALYSIS_IMAGE_SIZE
        self.analysis_size = analysis_size
        self.fft_window = self._create_fft_window(analysis_size)

    def build(
        self,
        frame_timestamp,
        original_frame,
        analysis_frame,
        face_box,
        aligned_face_crop=None,
        pose_alignment_valid=None,
    ):
        frame_height, frame_width = analysis_frame.shape[:2]
        safe_box = face_box.clamp_to_frame(frame_width, frame_height)
        quality = self._measure_quality(
            analysis_frame,
            safe_box,
            frame_width,
            frame_height,
        )

        face_crop = quality["face_crop"]
        grayscale_crop = quality["grayscale_crop"]
        alignment_applied = aligned_face_crop is not None
        if aligned_face_crop is None:
            # Model-free guide ROI'sinde geometrik hizalama yoktur. Alan yine
            # ortak API'de tutulur fakat pose gecerliligi bilinmiyor kalir.
            aligned_face_crop = face_crop

        context = ModelFreePreControlContext(
            frame_timestamp=frame_timestamp,
            original_frame=original_frame,
            analysis_frame=analysis_frame,
            original_high_resolution_face_crop=face_crop,
            aligned_face_crop=aligned_face_crop,
            standardized_aligned_face_crop=None,
            standardized_analysis_crop=None,
            grayscale_crop=grayscale_crop,
            frame_dimensions=(frame_width, frame_height),
            face_dimensions=self._image_dimensions(face_crop),
            analysis_dimensions=None,
            face_bounding_box=safe_box,
            face_quality_valid=quality["valid"],
            quality_reason=quality["reason"],
            blur_value=quality["blur"],
            brightness_value=quality["brightness"],
            exposure_valid=quality["exposure_valid"],
            pose_alignment_valid=pose_alignment_valid,
            fft_result=None,
            shifted_fft_result=None,
            magnitude_spectrum=None,
            power_spectrum=None,
            log_power_spectrum=None,
            log_magnitude_visualization=None,
            fft_window_type=config.MODEL_FREE_FFT_WINDOW_TYPE,
            alignment_applied=alignment_applied,
        )

        if not context.face_quality_valid:
            return context

        analysis_source = context.aligned_face_crop
        standardized_aligned_crop = self._standardize_aligned_crop(
            analysis_source
        )
        standardized_crop = self._prepare_fft_crop(
            standardized_aligned_crop
        )

        # Tek FFT hesaplama noktasi. Diger tum frekans modulleri bu context'teki
        # kompleks, magnitude veya power temsillerinden uygun olani kullanir.
        fft_result = np.fft.fft2(standardized_crop)
        shifted_fft_result = np.fft.fftshift(fft_result)
        magnitude_spectrum = np.abs(shifted_fft_result)
        power_spectrum = magnitude_spectrum ** 2
        log_power_spectrum = np.log1p(power_spectrum).astype(np.float32)
        log_magnitude = np.log1p(magnitude_spectrum).astype(np.float32)
        log_magnitude_visualization = cv2.normalize(
            log_magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        context.standardized_aligned_face_crop = standardized_aligned_crop
        context.standardized_analysis_crop = standardized_crop
        context.analysis_dimensions = self._image_dimensions(
            standardized_crop
        )
        context.fft_result = fft_result
        context.shifted_fft_result = shifted_fft_result
        context.magnitude_spectrum = magnitude_spectrum
        context.power_spectrum = power_spectrum
        context.log_power_spectrum = log_power_spectrum
        context.log_magnitude_visualization = log_magnitude_visualization
        return context

    def _measure_quality(self, frame, safe_box, frame_width, frame_height):
        face_crop = frame[
            safe_box.y : safe_box.y + safe_box.height,
            safe_box.x : safe_box.x + safe_box.width,
        ].copy()
        if face_crop.size == 0:
            return self._quality_values(False, "empty face crop")

        grayscale_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(grayscale_crop.mean())
        blur = float(cv2.Laplacian(grayscale_crop, cv2.CV_64F).var())
        exposure_valid = (
            config.EXPERIMENTAL_MODEL_FREE_MINIMUM_BRIGHTNESS
            <= brightness
            <= config.EXPERIMENTAL_MODEL_FREE_MAXIMUM_BRIGHTNESS
        )

        reason = self._quality_failure_reason(
            safe_box,
            frame_width,
            frame_height,
            blur,
            brightness,
        )
        return self._quality_values(
            reason is None,
            reason,
            face_crop,
            grayscale_crop,
            blur,
            brightness,
            exposure_valid,
        )

    def _quality_failure_reason(
        self,
        safe_box,
        frame_width,
        frame_height,
        blur,
        brightness,
    ):
        minimum_side = config.EXPERIMENTAL_MODEL_FREE_MINIMUM_FACE_SIDE
        if safe_box.width < minimum_side or safe_box.height < minimum_side:
            return "face too small"

        frame_area = float(frame_width * frame_height)
        if safe_box.get_area() / frame_area < (
            config.EXPERIMENTAL_MODEL_FREE_MINIMUM_FACE_AREA_RATIO
        ):
            return "face too small"

        edge_margin = config.EXPERIMENTAL_MODEL_FREE_FRAME_EDGE_MARGIN_RATIO
        edge_x = int(frame_width * edge_margin)
        edge_y = int(frame_height * edge_margin)
        if (
            safe_box.x <= edge_x
            or safe_box.y <= edge_y
            or safe_box.x + safe_box.width >= frame_width - edge_x
            or safe_box.y + safe_box.height >= frame_height - edge_y
        ):
            return "face partially outside frame"

        if blur < config.EXPERIMENTAL_MODEL_FREE_MINIMUM_BLUR_SCORE:
            return "face is blurred"
        if brightness < config.EXPERIMENTAL_MODEL_FREE_MINIMUM_BRIGHTNESS:
            return "face is too dark"
        if brightness > config.EXPERIMENTAL_MODEL_FREE_MAXIMUM_BRIGHTNESS:
            return "face is overexposed"
        return None

    def _standardize_aligned_crop(self, face_crop):
        """Hizalanmis crop'u ortak boyutta, penceresiz float griye cevirir.

        DCT/blok analizi bu temsili kullanir. FFT'ye ozel ortalama, varyans ve
        Hann islemleri ayrica uygulanir; boylece pencere blok istatistiklerine
        yapay kenar izi eklemez.
        """
        if face_crop.ndim == 3:
            grayscale = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = face_crop

        interpolation = cv2.INTER_AREA
        if min(grayscale.shape[:2]) < self.analysis_size:
            interpolation = cv2.INTER_CUBIC
        resized = cv2.resize(
            grayscale,
            (self.analysis_size, self.analysis_size),
            interpolation=interpolation,
        ).astype(np.float32)
        return resized

    def _prepare_fft_crop(self, standardized_aligned_crop):
        fft_crop = standardized_aligned_crop.copy()
        fft_crop -= float(fft_crop.mean())
        standard_deviation = float(fft_crop.std())
        if standard_deviation > 1e-6:
            fft_crop /= standard_deviation
        return fft_crop * self.fft_window

    def _create_fft_window(self, analysis_size):
        window_type = config.MODEL_FREE_FFT_WINDOW_TYPE.lower()
        if window_type == "hann":
            window_1d = np.hanning(analysis_size)
            return np.outer(window_1d, window_1d).astype(np.float32)
        if window_type == "none":
            return np.ones((analysis_size, analysis_size), dtype=np.float32)
        raise ValueError("Unsupported FFT window type: " + window_type)

    def _quality_values(
        self,
        valid,
        reason,
        face_crop=None,
        grayscale_crop=None,
        blur=None,
        brightness=None,
        exposure_valid=None,
    ):
        return {
            "valid": valid,
            "reason": reason,
            "face_crop": face_crop,
            "grayscale_crop": grayscale_crop,
            "blur": blur,
            "brightness": brightness,
            "exposure_valid": exposure_valid,
        }

    def _image_dimensions(self, image):
        if image is None:
            return None
        height, width = image.shape[:2]
        return (width, height)
