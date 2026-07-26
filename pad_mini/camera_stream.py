"""Kameradan daima en yeni kareyi alan arka plan okuyucusu."""

import threading
import time

import cv2


class LatestFrameCamera:
    """Eski karelerin kuyrukta birikmesini engelleyen kamera okuyucusu."""

    MAXIMUM_CONSECUTIVE_ERRORS = 10

    def __init__(self, source):
        self.source = source
        self.capture = self._open_capture(source)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_metadata = None
        self.frame_number = 0
        self.consecutive_errors = 0
        self.running = False
        self.reader_thread = None
        self.nominal_fps = self._finite_positive(
            self.capture.get(cv2.CAP_PROP_FPS)
        )
        self.previous_acquisition_monotonic_s = None

    def is_opened(self):
        return self.capture.isOpened()

    def start(self):
        if not self.is_opened() or self.running:
            return

        self.running = True
        self.reader_thread = threading.Thread(
            target=self._read_frames,
            name="camera-reader",
            daemon=True,
        )
        self.reader_thread.start()

    def read_latest(self, previous_frame_number):
        with self.lock:
            if self.latest_frame is None:
                return False, None, previous_frame_number
            if self.frame_number == previous_frame_number:
                return False, None, previous_frame_number

            return True, self.latest_frame, self.frame_number

    def read_latest_with_metadata(self, previous_frame_number):
        """Return the newest frame plus additive capture provenance.

        ``read_latest`` remains unchanged for callers that rely on its
        established three-value return signature.
        """
        with self.lock:
            if self.latest_frame is None:
                return False, None, previous_frame_number, None
            if self.frame_number == previous_frame_number:
                return False, None, previous_frame_number, None

            metadata = dict(self.latest_metadata or {})
            metadata["frames_skipped_by_consumer"] = max(
                0,
                int(self.frame_number - previous_frame_number - 1),
            )
            return True, self.latest_frame, self.frame_number, metadata

    def has_failed(self):
        return (
            not self.running
            and self.consecutive_errors
            >= self.MAXIMUM_CONSECUTIVE_ERRORS
        )

    def release(self):
        self.running = False
        self.capture.release()

        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None

    def _read_frames(self):
        while self.running:
            frame_was_read, frame = self.capture.read()

            if not frame_was_read or frame is None:
                self.consecutive_errors += 1
                if (
                    self.consecutive_errors
                    >= self.MAXIMUM_CONSECUTIVE_ERRORS
                ):
                    self.running = False
                time.sleep(0.05)
                continue

            self.consecutive_errors = 0
            acquisition_monotonic_s = time.monotonic()
            acquisition_wall_time_s = time.time()
            interarrival_s = None
            if self.previous_acquisition_monotonic_s is not None:
                interarrival_s = (
                    acquisition_monotonic_s
                    - self.previous_acquisition_monotonic_s
                )
            self.previous_acquisition_monotonic_s = acquisition_monotonic_s
            source_position_ms = self._finite_nonnegative(
                self.capture.get(cv2.CAP_PROP_POS_MSEC)
            )
            metadata = {
                "frame_number": self.frame_number + 1,
                "acquisition_monotonic_s": acquisition_monotonic_s,
                "acquisition_wall_time_s": acquisition_wall_time_s,
                "interarrival_s": interarrival_s,
                "nominal_fps": self.nominal_fps,
                "source_position_ms": source_position_ms,
                "source_kind": self._source_kind(),
                "timestamp_basis": "decoder_arrival_monotonic",
                "timestamp_reliability": self._timestamp_reliability(
                    interarrival_s
                ),
                "decoded_frame": True,
                "encoded_bytes_available": False,
                "codec_or_jpeg_tables_available": False,
            }
            with self.lock:
                self.latest_frame = frame
                self.latest_metadata = metadata
                self.frame_number += 1

    def _source_kind(self):
        if isinstance(self.source, str):
            if self.source.startswith(("http://", "https://")):
                return "network_stream"
            if self.source.startswith("/dev/video"):
                return "camera_stream"
            return "file_or_stream"
        if isinstance(self.source, int):
            return "camera_stream"
        return "unknown"

    def _timestamp_reliability(self, interarrival_s):
        if interarrival_s is None or self.nominal_fps is None:
            return "low"
        expected = 1.0 / self.nominal_fps
        relative_error = abs(interarrival_s - expected) / max(expected, 1e-9)
        if relative_error <= 0.20:
            return "medium"
        return "low"

    @staticmethod
    def _finite_positive(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0.0 or numeric != numeric or numeric == float("inf"):
            return None
        return numeric

    @staticmethod
    def _finite_nonnegative(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0.0 or numeric != numeric or numeric == float("inf"):
            return None
        return numeric

    def _open_capture(self, source):
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            parameters = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                3000,
            ]
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                parameters,
            )
        elif isinstance(source, str) and source.startswith("/dev/video"):
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(source)

        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return capture
