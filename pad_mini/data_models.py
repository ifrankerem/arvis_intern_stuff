from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point:
    """Görüntü üzerindeki bir piksel koordinatı."""

    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        """OpenCV fonksiyonlarının beklediği (x, y) biçimini döndürür."""

        return self.x, self.y


@dataclass(frozen=True)
class FaceBox:
    """Tespit edilen yüzü çevreleyen dikdörtgen."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(
            x=self.x + self.width,
            y=self.y + self.height,
        )

    def clamp_to_frame(self, frame_width: int, frame_height: int) -> FaceBox:
        """Kutuyu görüntü sınırlarının dışına taşmayacak şekilde daraltır."""

        left = max(0, self.x)
        top = max(0, self.y)
        right = min(frame_width, self.x + self.width)
        bottom = min(frame_height, self.y + self.height)

        return FaceBox(
            x=left,
            y=top,
            width=max(0, right - left),
            height=max(0, bottom - top),
        )


@dataclass(frozen=True)
class FaceQuality:
    """Tek bir yüz için hesaplanan görüntü kalitesi değerleri."""

    face_area_ratio: float
    blur_score: float
    brightness: float
    face_large_enough: bool
    sharp_enough: bool
    brightness_ok: bool

    @property
    def is_acceptable(self) -> bool:
        return (
            self.face_large_enough
            and self.sharp_enough
            and self.brightness_ok
        )


@dataclass(frozen=True)
class FaceLandmarkDetection:
    """MediaPipe'ın yoğun landmark ve yüz hareketi çıktısı."""

    box: FaceBox
    landmarks: tuple[Point, ...]
    left_eye_landmarks: tuple[Point, ...]
    right_eye_landmarks: tuple[Point, ...]
    mouth_landmarks: tuple[Point, ...]
    nose_tip: Point
    left_mouth_corner: Point
    right_mouth_corner: Point
    upper_lip_center: Point
    lower_lip_center: Point
    left_eye_blink_score: float
    right_eye_blink_score: float
    jaw_open_score: float
    mouth_left_score: float
    mouth_right_score: float
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float

    @property
    def average_blink_score(self) -> float:
        return (
            self.left_eye_blink_score + self.right_eye_blink_score
        ) / 2.0


@dataclass(frozen=True)
class FaceAlignment:
    """Kafa ve ağız hizalama analizinin sonucu."""

    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    mouth_angle_degrees: float
    mouth_open_score: float
    mouth_lateral_difference: float
    head_aligned: bool
    mouth_closed: bool
    mouth_centered: bool
    mouth_aligned: bool

    @property
    def is_acceptable(self) -> bool:
        return self.head_aligned and self.mouth_aligned


class Challenge(str, Enum):
    BLINK = "BLINK"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


@dataclass(frozen=True)
class LivenessResult:
    """JSON dosyasına yazılacak aktif canlılık sonucu."""

    face_detected: bool
    quality_status: str
    liveness_type: str
    challenge_sequence: tuple[Challenge, ...]
    completed_challenges: tuple[Challenge, ...]
    verdict: str
    risk_score: int
    processing_time_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "face_detected": self.face_detected,
            "quality_status": self.quality_status,
            "liveness_type": self.liveness_type,
            "challenge_sequence": [
                challenge.value for challenge in self.challenge_sequence
            ],
            "completed_challenges": [
                challenge.value for challenge in self.completed_challenges
            ],
            "verdict": self.verdict,
            "risk_score": self.risk_score,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass(frozen=True)
class FaceAnalysisResult:
    """Canlılık kararı vermeden üretilen yüz analizi JSON sonucu."""

    face_detected: bool
    quality_status: str
    alignment_status: str
    alignment: FaceAlignment | None
    processing_time_ms: int

    def to_dict(self) -> dict[str, object]:
        head_pose: dict[str, float] | None = None
        mouth: dict[str, object] | None = None

        if self.alignment is not None:
            head_pose = {
                "yaw_degrees": round(self.alignment.yaw_degrees, 2),
                "pitch_degrees": round(self.alignment.pitch_degrees, 2),
                "roll_degrees": round(self.alignment.roll_degrees, 2),
                "aligned": self.alignment.head_aligned,
            }
            mouth = {
                "open_score": round(self.alignment.mouth_open_score, 3),
                "angle_degrees": round(
                    self.alignment.mouth_angle_degrees,
                    2,
                ),
                "lateral_difference": round(
                    self.alignment.mouth_lateral_difference,
                    3,
                ),
                "closed": self.alignment.mouth_closed,
                "centered": self.alignment.mouth_centered,
                "aligned": self.alignment.mouth_aligned,
            }

        return {
            "face_detected": self.face_detected,
            "quality_status": self.quality_status,
            "alignment_status": self.alignment_status,
            "head_pose": head_pose,
            "mouth": mouth,
            "liveness_type": "NOT_EVALUATED",
            "challenge_sequence": [],
            "completed_challenges": [],
            "verdict": "NOT_EVALUATED",
            "risk_score": None,
            "processing_time_ms": self.processing_time_ms,
        }
