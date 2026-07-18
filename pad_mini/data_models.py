class Point:
    """Goruntu uzerindeki bir piksel koordinati."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return (self.x, self.y)


class Color:
    """OpenCV'nin kullandigi BGR renk degeri."""

    def __init__(self, blue, green, red):
        self.blue = blue
        self.green = green
        self.red = red

    def to_tuple(self):
        return (self.blue, self.green, self.red)


class DisplayMessage:
    """Ekranda gosterilecek metin ve renk."""

    def __init__(self, text, color):
        self.text = text
        self.color = color


class FaceBox:
    """Tespit edilen yuzun dikdortgen sinirlari."""

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def get_top_left(self):
        return Point(self.x, self.y)

    def get_bottom_right(self):
        return Point(
            self.x + self.width,
            self.y + self.height,
        )

    def clamp_to_frame(self, frame_width, frame_height):
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(frame_width, self.x + self.width)
        bottom = min(frame_height, self.y + self.height)

        return FaceBox(
            left,
            top,
            max(0, right - left),
            max(0, bottom - top),
        )


class FaceQuality:
    """Bir yuz icin hesaplanan goruntu kalitesi."""

    def __init__(
        self,
        face_area_ratio,
        blur_score,
        brightness,
        face_large_enough,
        sharp_enough,
        brightness_ok,
    ):
        self.face_area_ratio = face_area_ratio
        self.blur_score = blur_score
        self.brightness = brightness
        self.face_large_enough = face_large_enough
        self.sharp_enough = sharp_enough
        self.brightness_ok = brightness_ok

    def is_acceptable(self):
        return (
            self.face_large_enough
            and self.sharp_enough
            and self.brightness_ok
        )


class FaceLandmarkDetection:
    """MediaPipe tarafindan tespit edilen yuz ve landmark verileri."""

    def __init__(
        self,
        box,
        landmarks,
        left_eye_landmarks,
        right_eye_landmarks,
        mouth_landmarks,
        nose_tip,
        left_mouth_corner,
        right_mouth_corner,
        upper_lip_center,
        lower_lip_center,
        left_eye_blink_score,
        right_eye_blink_score,
        jaw_open_score,
        mouth_left_score,
        mouth_right_score,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
    ):
        self.box = box
        self.landmarks = landmarks
        self.left_eye_landmarks = left_eye_landmarks
        self.right_eye_landmarks = right_eye_landmarks
        self.mouth_landmarks = mouth_landmarks
        self.nose_tip = nose_tip
        self.left_mouth_corner = left_mouth_corner
        self.right_mouth_corner = right_mouth_corner
        self.upper_lip_center = upper_lip_center
        self.lower_lip_center = lower_lip_center
        self.left_eye_blink_score = left_eye_blink_score
        self.right_eye_blink_score = right_eye_blink_score
        self.jaw_open_score = jaw_open_score
        self.mouth_left_score = mouth_left_score
        self.mouth_right_score = mouth_right_score
        self.yaw_degrees = yaw_degrees
        self.pitch_degrees = pitch_degrees
        self.roll_degrees = roll_degrees

    def get_average_blink_score(self):
        total_score = (
            self.left_eye_blink_score
            + self.right_eye_blink_score
        )
        return total_score / 2.0


class HeadRotation:
    """Modelin yuz donus matrisinden hesaplanan kafa acilari."""

    def __init__(self, pitch_degrees, yaw_degrees, roll_degrees):
        self.pitch_degrees = pitch_degrees
        self.yaw_degrees = yaw_degrees
        self.roll_degrees = roll_degrees


class FaceAlignment:
    """Kafa ve agiz hizalama analizinin sonucu."""

    def __init__(
        self,
        yaw_degrees,
        pitch_degrees,
        roll_degrees,
        mouth_angle_degrees,
        mouth_open_score,
        mouth_lateral_difference,
        head_aligned,
        mouth_closed,
        mouth_centered,
        mouth_aligned,
    ):
        self.yaw_degrees = yaw_degrees
        self.pitch_degrees = pitch_degrees
        self.roll_degrees = roll_degrees
        self.mouth_angle_degrees = mouth_angle_degrees
        self.mouth_open_score = mouth_open_score
        self.mouth_lateral_difference = mouth_lateral_difference
        self.head_aligned = head_aligned
        self.mouth_closed = mouth_closed
        self.mouth_centered = mouth_centered
        self.mouth_aligned = mouth_aligned

    def is_acceptable(self):
        return self.head_aligned and self.mouth_aligned


class FaceAnalysisResult:
    """JSON dosyasina yazilacak yuz analiz sonucu."""

    def __init__(
        self,
        face_detected,
        quality_status,
        alignment_status,
        alignment,
        processing_time_ms,
    ):
        self.face_detected = face_detected
        self.quality_status = quality_status
        self.alignment_status = alignment_status
        self.alignment = alignment
        self.processing_time_ms = processing_time_ms

    def to_dictionary(self):
        head_pose = None
        mouth = None

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


class FrameProcessingResult:
    """Tek kamera karesi islendikten sonra uretilen veriler."""

    def __init__(self, display_frame, face_image, analysis_result):
        self.display_frame = display_frame
        self.face_image = face_image
        self.analysis_result = analysis_result


class FaceFrameAnalysis:
    """Tek yuz bulunan bir karede hesaplanan ara sonuclar."""

    def __init__(self, detected_face, quality, alignment, face_image):
        self.detected_face = detected_face
        self.quality = quality
        self.alignment = alignment
        self.face_image = face_image
