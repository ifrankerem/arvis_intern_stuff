import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from camera_stream import LatestFrameCamera
from data_models import FaceBox
from model_free_analysis import (
    ModelFreeAnalysisResult,
    ModelFreePreControlContextBuilder,
)
from precontrol_decision import PreControlDecisionBuilder


class PreControlContractTests(unittest.TestCase):
    def test_roi_topology_preserves_raw_background_and_full_frame(self):
        random = np.random.default_rng(91)
        frame = random.integers(0, 256, (360, 480, 3), dtype=np.uint8)
        context = ModelFreePreControlContextBuilder().build(
            10.0,
            frame,
            frame,
            FaceBox(112, 52, 256, 256),
            capture_metadata={"timestamp_basis": "unit_test"},
        )

        expected = {
            "aligned_face",
            "raw_face",
            "expanded_face",
            "forehead",
            "left_cheek",
            "right_cheek",
            "nose",
            "eyes",
            "background_ring",
            "full_frame",
        }
        self.assertEqual(set(context.rois), expected)
        self.assertEqual(context.capture_metadata["timestamp_basis"], "unit_test")
        self.assertIn(
            "identity_alias_not_aligned",
            context.rois["aligned_face"].transform_history,
        )
        self.assertIsNotNone(context.rois["background_ring"].mask)
        self.assertGreater(
            np.count_nonzero(context.rois["background_ring"].mask),
            0,
        )

    def test_legacy_result_adapts_to_canonical_method_contract(self):
        result = ModelFreeAnalysisResult(
            module_name="Example",
            available=True,
            raw_features={"measurement": 2.5},
            raw_score=62.0,
            confidence=0.4,
            status="Suspicious",
            evidence_family="frequency",
            attack_targets=["replay_screen"],
            reason_codes=["EXAMPLE_ELEVATED"],
            human_explanation="Example measurement is elevated",
            runtime_ms=3.2,
            triggered=True,
        )

        canonical = result.to_method_result().to_dict()

        self.assertEqual(canonical["method_name"], "Example")
        self.assertEqual(canonical["evidence_family"], "frequency")
        self.assertAlmostEqual(canonical["normalized_score"], 0.62)
        self.assertAlmostEqual(canonical["reliability"], 0.4)
        self.assertTrue(canonical["triggered"])
        self.assertEqual(canonical["runtime_ms"], 3.2)

    def test_uncalibrated_low_score_never_becomes_live(self):
        context = SimpleNamespace(face_quality_valid=True, quality_reason=None)
        results = {
            "frequency": self._result("frequency", "frequency", 10.0),
            "compression": self._result(
                "compression", "compression_recapture", 8.0
            ),
            "texture": self._result("texture", "spatial_texture", 12.0),
        }
        combined = SimpleNamespace(available=True, calibrated=False)

        decision = PreControlDecisionBuilder().build(
            results,
            context,
            combined,
        )

        self.assertEqual(decision.classification, "INSUFFICIENT_EVIDENCE")
        self.assertNotEqual(decision.classification, "LIVE")
        self.assertTrue(decision.score_valid)
        self.assertIn("DEPLOYMENT_CALIBRATION_REQUIRED", decision.reason_codes)

    def test_invalid_quality_has_no_invented_score(self):
        context = SimpleNamespace(
            face_quality_valid=False,
            quality_reason="face is blurred",
        )

        decision = PreControlDecisionBuilder().build({}, context, None)

        self.assertEqual(decision.classification, "INSUFFICIENT_QUALITY")
        self.assertIsNone(decision.overall_risk_0_100)
        self.assertFalse(decision.score_valid)

    def test_latest_frame_metadata_api_is_additive(self):
        camera = LatestFrameCamera.__new__(LatestFrameCamera)
        camera.lock = __import__("threading").Lock()
        camera.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        camera.latest_metadata = {
            "timestamp_basis": "decoder_arrival_monotonic",
            "acquisition_monotonic_s": 12.0,
        }
        camera.frame_number = 7

        legacy = camera.read_latest(5)
        extended = camera.read_latest_with_metadata(5)

        self.assertEqual(len(legacy), 3)
        self.assertEqual(len(extended), 4)
        self.assertEqual(extended[3]["frames_skipped_by_consumer"], 1)

    @staticmethod
    def _result(name, family, score):
        return ModelFreeAnalysisResult(
            module_name=name,
            available=True,
            raw_score=score,
            confidence=1.0,
            status="Normal",
            evidence_family=family,
        )


if __name__ == "__main__":
    unittest.main()
