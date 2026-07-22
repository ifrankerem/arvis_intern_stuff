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
        self.frame_number = 0
        self.consecutive_errors = 0
        self.running = False
        self.reader_thread = None

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
            with self.lock:
                self.latest_frame = frame
                self.frame_number += 1

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
