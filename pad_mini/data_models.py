from __future__ import annotations

from dataclasses import dataclass


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
class FaceDetection:
    """YuNet'in tek bir yüz için ürettiği anlamlandırılmış sonuç."""

    box: FaceBox
    right_eye: Point
    left_eye: Point
    nose_tip: Point
    right_mouth_corner: Point
    left_mouth_corner: Point
    confidence_score: float

    @property
    def landmark_points(self) -> tuple[Point, Point, Point, Point, Point]:
        return (
            self.right_eye,
            self.left_eye,
            self.nose_tip,
            self.right_mouth_corner,
            self.left_mouth_corner,
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
