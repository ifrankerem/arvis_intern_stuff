import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from data_models import FaceBox
from face_roi_tracker import StableFaceROITracker
from model_free_pre_control_application import ModelFreePreControlApplication


class SequenceFaceDetector:
    def __init__(self, detections):
        self.detections = list(detections)
        self.timestamps = []

    def detect_faces(self, _frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        if not self.detections:
            return []
        boxes = self.detections.pop(0)
        return [SimpleNamespace(box=box) for box in boxes]


class StableFaceROITrackerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detection_box_is_smoothed_instead_of_jumping(self):
        detector = SequenceFaceDetector(
            [
                [FaceBox(100, 90, 160, 190)],
                [FaceBox(120, 100, 160, 190)],
            ]
        )
        tracker = StableFaceROITracker(
            detector,
            smoothing_alpha=0.25,
            horizontal_expansion_ratio=0.0,
            vertical_expansion_ratio=0.0,
        )

        first = tracker.update(self.frame, 1.0)
        second = tracker.update(self.frame, 1.1)

        self.assertEqual(first.status, "DETECTED")
        self.assertEqual(second.status, "DETECTED")
        self.assertEqual(second.box.x, 105)
        self.assertEqual(second.box.y, 92)

    def test_short_detection_gap_holds_then_expires(self):
        detector = SequenceFaceDetector(
            [[FaceBox(100, 90, 160, 190)], [], []]
        )
        tracker = StableFaceROITracker(
            detector,
            hold_seconds=0.8,
            horizontal_expansion_ratio=0.0,
            vertical_expansion_ratio=0.0,
        )

        detected = tracker.update(self.frame, 1.0)
        held = tracker.update(self.frame, 1.4)
        expired = tracker.update(self.frame, 1.9)

        self.assertTrue(detected.supported)
        self.assertEqual(held.status, "HELD")
        self.assertTrue(held.supported)
        self.assertAlmostEqual(held.reliability, 0.5)
        self.assertEqual(expired.status, "NO_FACE")
        self.assertFalse(expired.supported)

    def test_detector_timestamps_are_strictly_increasing(self):
        detector = SequenceFaceDetector([[], [], []])
        tracker = StableFaceROITracker(detector)

        tracker.update(self.frame, 2.0)
        tracker.update(self.frame, 2.0)
        tracker.update(self.frame, 1.9)

        self.assertEqual(detector.timestamps, [2000, 2001, 2002])

    def test_multiple_faces_clear_the_previous_hold_state(self):
        detector = SequenceFaceDetector(
            [
                [FaceBox(100, 90, 160, 190)],
                [
                    FaceBox(100, 90, 160, 190),
                    FaceBox(350, 100, 140, 170),
                ],
                [],
            ]
        )
        tracker = StableFaceROITracker(detector, hold_seconds=0.8)

        tracker.update(self.frame, 1.0)
        multiple = tracker.update(self.frame, 1.1)
        after_multiple = tracker.update(self.frame, 1.2)

        self.assertEqual(multiple.status, "MULTIPLE_FACES")
        self.assertFalse(multiple.supported)
        self.assertEqual(after_multiple.status, "NO_FACE")
        self.assertFalse(after_multiple.supported)

    def test_expansion_alone_does_not_force_roi_onto_frame_edge(self):
        detector = SequenceFaceDetector(
            [[FaceBox(20, 15, 600, 450)]]
        )
        tracker = StableFaceROITracker(
            detector,
            horizontal_expansion_ratio=0.10,
            vertical_expansion_ratio=0.12,
            minimum_frame_margin_ratio=0.01,
        )

        result = tracker.update(self.frame, 1.0)

        self.assertGreater(result.box.x, int(self.frame.shape[1] * 0.01))
        self.assertGreater(result.box.y, int(self.frame.shape[0] * 0.01))
        self.assertLess(
            result.box.x + result.box.width,
            self.frame.shape[1] - int(self.frame.shape[1] * 0.01),
        )


class ModelFreeFaceROIIntegrationTests(unittest.TestCase):
    @staticmethod
    def _textured_frame():
        random = np.random.default_rng(81)
        gray = np.clip(
            125 + random.normal(0, 22, (480, 640)),
            0,
            255,
        ).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_detected_roi_replaces_guide_and_missing_face_is_unsupported(self):
        detector = SequenceFaceDetector(
            [[FaceBox(210, 120, 180, 210)], [], []]
        )
        application = ModelFreePreControlApplication(
            face_detector=detector,
            enable_face_detection=True,
            runtime_mode="FAST",
        )
        frame = self._textured_frame()

        application.process_frame(
            frame,
            frame_metadata={"acquisition_monotonic_s": 10.0},
        )
        detected_context = application.latest_context
        detected_tracking = application.latest_face_tracking_result
        application.process_frame(
            frame,
            frame_metadata={"acquisition_monotonic_s": 10.4},
        )
        held_context = application.latest_context
        held_tracking = application.latest_face_tracking_result
        application.process_frame(
            frame,
            frame_metadata={"acquisition_monotonic_s": 11.0},
        )
        missing_context = application.latest_context
        missing_tracking = application.latest_face_tracking_result
        missing_decision = application.latest_precontrol_decision
        application.shutdown()

        self.assertEqual(detected_tracking.status, "DETECTED")
        self.assertEqual(
            detected_context.capture_metadata["face_roi_source"],
            "detected_face_landmarks",
        )
        self.assertNotEqual(
            detected_context.face_bounding_box.width,
            application.create_guide_box(frame).width,
        )
        self.assertEqual(held_tracking.status, "HELD")
        self.assertTrue(held_context.face_quality_valid)
        self.assertEqual(missing_tracking.status, "NO_FACE")
        self.assertFalse(missing_context.face_quality_valid)
        self.assertEqual(
            missing_context.quality_reason,
            "face temporarily not detected",
        )
        self.assertEqual(missing_context.face_bounding_box.get_area(), 0)
        self.assertEqual(
            missing_context.capture_metadata["face_roi_source"],
            "no_detected_face",
        )
        self.assertTrue(
            all(
                not result.available
                for result in application.latest_pre_control_results.values()
            )
        )
        self.assertEqual(
            missing_decision.classification,
            "INSUFFICIENT_QUALITY",
        )
        self.assertFalse(
            missing_context.capture_metadata[
                "face_detector_confidence_used_as_pad_evidence"
            ]
        )

    def test_detected_face_at_frame_edge_is_degraded_not_rejected(self):
        detector = SequenceFaceDetector(
            [[FaceBox(145, 55, 350, 425)]]
        )
        application = ModelFreePreControlApplication(
            face_detector=detector,
            enable_face_detection=True,
            runtime_mode="FAST",
        )
        frame = self._textured_frame()

        application.process_frame(
            frame,
            frame_metadata={"acquisition_monotonic_s": 20.0},
        )
        context = application.latest_context
        results = dict(application.latest_pre_control_results)
        application.shutdown()

        self.assertTrue(context.capture_metadata["face_roi_edge_contact"])
        self.assertTrue(context.face_quality_valid)
        for name in ("fft", "moire", "dct_block", "residual"):
            self.assertTrue(results[name].available, name)
            self.assertIn(
                "Method reliability reduced because the detected face ROI "
                "touches the frame margin",
                results[name].warnings,
            )

    def test_fixed_center_guide_is_not_drawn(self):
        application = ModelFreePreControlApplication(
            enable_face_detection=False
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        guide = application.create_guide_box(frame)
        application.draw_text = lambda *_args, **_kwargs: None
        normal_fft = SimpleNamespace(
            passed=True,
            warning="",
            status="Normal",
            score=None,
            calibrated=False,
        )
        normal_moire = SimpleNamespace(
            warning="",
            status="Normal",
            score=None,
        )
        application.draw_guide(
            frame,
            guide,
            normal_fft,
            normal_moire,
        )
        application.shutdown()

        self.assertEqual(int(np.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()
