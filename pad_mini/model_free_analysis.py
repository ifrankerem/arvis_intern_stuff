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
class ModelFreeROI:
    """One named ROI and the provenance required to interpret it safely."""

    name: str
    image: Optional[np.ndarray]
    frame_box: Optional[Tuple[int, int, int, int]]
    mask: Optional[np.ndarray] = None
    coordinate_space: str = "analysis_frame"
    transform_history: List[str] = field(default_factory=list)
    semantic_basis: str = "guide_relative"
    valid: bool = True
    reason: Optional[str] = None

    def provenance(self):
        return {
            "name": self.name,
            "frame_box": self.frame_box,
            "coordinate_space": self.coordinate_space,
            "transform_history": list(self.transform_history),
            "semantic_basis": self.semantic_basis,
            "valid": bool(self.valid),
            "reason": self.reason,
            "dimensions": (
                None
                if self.image is None
                else (int(self.image.shape[1]), int(self.image.shape[0]))
            ),
            "has_mask": self.mask is not None,
        }


@dataclass(frozen=True)
class MethodResult:
    """Canonical deterministic method contract used by new integrations.

    Legacy analyzers continue to return ``ModelFreeAnalysisResult``. Its
    ``to_method_result`` adapter exposes this contract without changing the
    established 0-100 score fields consumed by the GUI and fusion code.
    """

    method_name: str
    evidence_family: str
    attack_targets: List[str]
    supported: bool
    raw_metrics: Dict[str, Any]
    normalized_score: float
    reliability: float
    triggered: bool
    reason_codes: List[str]
    human_explanation: str
    visualization_paths: Dict[str, str]
    runtime_ms: float
    warnings: List[str]

    def to_dict(self):
        return {
            "method_name": self.method_name,
            "evidence_family": self.evidence_family,
            "attack_targets": list(self.attack_targets),
            "supported": bool(self.supported),
            "raw_metrics": dict(self.raw_metrics),
            "normalized_score": float(
                np.clip(self.normalized_score, 0.0, 1.0)
            ),
            "reliability": float(np.clip(self.reliability, 0.0, 1.0)),
            "triggered": bool(self.triggered),
            "reason_codes": list(self.reason_codes),
            "human_explanation": self.human_explanation,
            "visualization_paths": dict(self.visualization_paths),
            "runtime_ms": max(0.0, float(self.runtime_ms)),
            "warnings": list(self.warnings),
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
    rois: Dict[str, ModelFreeROI] = field(default_factory=dict)
    capture_metadata: Dict[str, Any] = field(default_factory=dict)

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
    evidence_family: str = "unassigned"
    attack_targets: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    human_explanation: str = ""
    visualization_paths: Dict[str, str] = field(default_factory=dict)
    runtime_ms: float = 0.0
    triggered: bool = False

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
    def supported(self):
        return bool(self.available)

    @property
    def normalized_score(self):
        score = self.score
        if score is None:
            return 0.0
        try:
            return float(np.clip(float(score) / 100.0, 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def reliability(self):
        try:
            return float(np.clip(float(self.confidence or 0.0), 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0

    def to_method_result(self):
        explanation = self.human_explanation
        if not explanation:
            explanation = "; ".join(self.evidence) or self.status
        return MethodResult(
            method_name=self.module_name,
            evidence_family=self.evidence_family,
            attack_targets=list(self.attack_targets),
            supported=self.supported,
            raw_metrics=dict(self.raw_features),
            normalized_score=self.normalized_score,
            reliability=self.reliability,
            triggered=bool(self.triggered),
            reason_codes=list(self.reason_codes),
            human_explanation=explanation,
            visualization_paths=dict(self.visualization_paths),
            runtime_ms=max(0.0, float(self.runtime_ms)),
            warnings=list(self.warnings),
        )

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
        capture_metadata=None,
        quality_override_reason=None,
        roi_semantic_basis="fixed_guide",
        allow_frame_edge_contact=False,
    ):
        frame_height, frame_width = analysis_frame.shape[:2]
        safe_box = face_box.clamp_to_frame(frame_width, frame_height)
        quality = self._measure_quality(
            analysis_frame,
            safe_box,
            frame_width,
            frame_height,
            allow_frame_edge_contact=allow_frame_edge_contact,
        )
        if quality_override_reason is not None:
            quality["valid"] = False
            quality["reason"] = str(quality_override_reason)

        face_crop = quality["face_crop"]
        grayscale_crop = quality["grayscale_crop"]
        alignment_applied = aligned_face_crop is not None
        if aligned_face_crop is None:
            # ROI'de geometrik hizalama yoktur. Alan ortak API'de kimlik
            # alias'i olarak tutulur fakat pose gecerliligi bilinmiyor kalir.
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
            rois=self._build_roi_set(
                analysis_frame,
                safe_box,
                face_crop,
                aligned_face_crop,
                alignment_applied,
                roi_semantic_basis,
            ),
            capture_metadata=dict(capture_metadata or {}),
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

    def _measure_quality(
        self,
        frame,
        safe_box,
        frame_width,
        frame_height,
        allow_frame_edge_contact=False,
    ):
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
            allow_frame_edge_contact=allow_frame_edge_contact,
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
        allow_frame_edge_contact=False,
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
        edge_contact = (
            safe_box.x <= edge_x
            or safe_box.y <= edge_y
            or safe_box.x + safe_box.width >= frame_width - edge_x
            or safe_box.y + safe_box.height >= frame_height - edge_y
        )
        if edge_contact and not allow_frame_edge_contact:
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

    def _build_roi_set(
        self,
        frame,
        face_box,
        raw_face_crop,
        aligned_face_crop,
        alignment_applied,
        roi_semantic_basis="fixed_guide",
    ):
        """Build relative ROIs and preserve how the parent ROI was obtained."""
        frame_height, frame_width = frame.shape[:2]
        rois = {}

        def add_frame_crop(name, box, semantic_basis=None):
            x, y, width, height = box
            safe = FaceBox(x, y, width, height).clamp_to_frame(
                frame_width,
                frame_height,
            )
            image = frame[
                safe.y : safe.y + safe.height,
                safe.x : safe.x + safe.width,
            ].copy()
            valid = image.size > 0 and min(image.shape[:2]) >= 8
            rois[name] = ModelFreeROI(
                name=name,
                image=image if valid else None,
                frame_box=(safe.x, safe.y, safe.width, safe.height),
                transform_history=["decoded_bgr", "mirrored_analysis_frame", "crop"],
                semantic_basis=(
                    semantic_basis
                    if semantic_basis is not None
                    else roi_semantic_basis + "_relative"
                ),
                valid=valid,
                reason=None if valid else "ROI is empty or too small",
            )
            return rois[name]

        raw_valid = raw_face_crop is not None and raw_face_crop.size > 0
        raw_history = ["decoded_bgr", "mirrored_analysis_frame", "crop"]
        rois["raw_face"] = ModelFreeROI(
            name="raw_face",
            image=raw_face_crop.copy() if raw_valid else None,
            frame_box=(face_box.x, face_box.y, face_box.width, face_box.height),
            transform_history=raw_history,
            semantic_basis=roi_semantic_basis,
            valid=raw_valid,
            reason=None if raw_valid else "Face/guide ROI is empty",
        )
        aligned_valid = aligned_face_crop is not None and aligned_face_crop.size > 0
        rois["aligned_face"] = ModelFreeROI(
            name="aligned_face",
            image=aligned_face_crop.copy() if aligned_valid else None,
            frame_box=(face_box.x, face_box.y, face_box.width, face_box.height),
            coordinate_space=(
                "aligned_crop" if alignment_applied else "analysis_frame_crop"
            ),
            transform_history=(
                raw_history + ["external_geometric_alignment"]
                if alignment_applied
                else raw_history + ["identity_alias_not_aligned"]
            ),
            semantic_basis=(
                "external_alignment"
                if alignment_applied
                else roi_semantic_basis
            ),
            valid=aligned_valid,
            reason=None if aligned_valid else "Aligned ROI is empty",
        )

        expand_x = int(round(face_box.width * 0.18))
        expand_y = int(round(face_box.height * 0.18))
        expanded = add_frame_crop(
            "expanded_face",
            (
                face_box.x - expand_x,
                face_box.y - expand_y,
                face_box.width + 2 * expand_x,
                face_box.height + 2 * expand_y,
            ),
            semantic_basis=roi_semantic_basis + "_with_boundary_context",
        )
        if expanded.valid and expanded.frame_box is not None:
            ex, ey, ew, eh = expanded.frame_box
            ring_mask = np.full((eh, ew), 255, dtype=np.uint8)
            inner_left = max(0, face_box.x - ex)
            inner_top = max(0, face_box.y - ey)
            inner_right = min(ew, face_box.x + face_box.width - ex)
            inner_bottom = min(eh, face_box.y + face_box.height - ey)
            ring_mask[inner_top:inner_bottom, inner_left:inner_right] = 0
            rois["background_ring"] = ModelFreeROI(
                name="background_ring",
                image=expanded.image.copy(),
                mask=ring_mask,
                frame_box=expanded.frame_box,
                transform_history=list(expanded.transform_history),
                semantic_basis="outside_" + roi_semantic_basis + "_ring",
                valid=bool(np.count_nonzero(ring_mask)),
                reason=None,
            )
        else:
            rois["background_ring"] = ModelFreeROI(
                name="background_ring",
                image=None,
                frame_box=None,
                semantic_basis="outside_" + roi_semantic_basis + "_ring",
                valid=False,
                reason="Expanded ROI is unavailable",
            )

        relative_boxes = {
            "forehead": (0.22, 0.10, 0.56, 0.23),
            "left_cheek": (0.10, 0.47, 0.34, 0.27),
            "right_cheek": (0.56, 0.47, 0.34, 0.27),
            "nose": (0.37, 0.34, 0.26, 0.40),
            "eyes": (0.12, 0.24, 0.76, 0.24),
        }
        for name, (rx, ry, rw, rh) in relative_boxes.items():
            add_frame_crop(
                name,
                (
                    face_box.x + int(round(rx * face_box.width)),
                    face_box.y + int(round(ry * face_box.height)),
                    max(1, int(round(rw * face_box.width))),
                    max(1, int(round(rh * face_box.height))),
                ),
            )

        rois["full_frame"] = ModelFreeROI(
            name="full_frame",
            image=frame,
            frame_box=(0, 0, frame_width, frame_height),
            transform_history=["decoded_bgr", "mirrored_analysis_frame"],
            semantic_basis="full_frame",
            valid=frame.size > 0,
        )
        return rois

    def _create_fft_window(self, analysis_size):
        window_type = config.MODEL_FREE_FFT_WINDOW_TYPE.lower()
        if window_type == "hann":
            window_1d = np.hanning(analysis_size)
        elif window_type == "hamming":
            window_1d = np.hamming(analysis_size)
        elif window_type == "tukey":
            window_1d = self._tukey_window(
                analysis_size,
                config.MODEL_FREE_FFT_TUKEY_ALPHA,
            )
        elif window_type == "none":
            window_1d = np.ones(analysis_size, dtype=np.float64)
        else:
            raise ValueError("Unsupported FFT window type: " + window_type)
        return np.outer(window_1d, window_1d).astype(np.float32)

    @staticmethod
    def _tukey_window(length, alpha):
        if length <= 1:
            return np.ones(max(1, length), dtype=np.float64)
        alpha = float(alpha)
        if alpha <= 0.0:
            return np.ones(length, dtype=np.float64)
        if alpha >= 1.0:
            return np.hanning(length)
        x = np.linspace(0.0, 1.0, length)
        window = np.ones(length, dtype=np.float64)
        first = x < alpha / 2.0
        last = x >= 1.0 - alpha / 2.0
        window[first] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * x[first] / alpha - 1.0))
        )
        window[last] = 0.5 * (
            1.0
            + np.cos(
                np.pi * (2.0 * x[last] / alpha - 2.0 / alpha + 1.0)
            )
        )
        return window

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
