import unittest

import cv2
import numpy as np

from data_models import FaceBox
from model_free_analysis import ModelFreePreControlContextBuilder
from moire_pre_control import MoirePeriodicPatternPreController


class MoireLocalPatchTests(unittest.TestCase):
    @staticmethod
    def _context(periodic_patch_count):
        random = np.random.default_rng(4)
        gray = np.clip(
            125 + random.normal(0, 14, (400, 400)),
            0,
            255,
        ).astype(np.uint8)
        x_coordinates = np.arange(48, dtype=np.float32)[None, :]
        grating = np.tile(
            125.0 + 70.0 * np.sin(2.0 * np.pi * x_coordinates / 8.0),
            (48, 1),
        )
        locations = ((136, 184), (184, 232))
        for x, y in locations[:periodic_patch_count]:
            gray[y : y + 48, x : x + 48] = np.clip(
                grating,
                0,
                255,
            ).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return ModelFreePreControlContextBuilder().build(
            1.0,
            frame,
            frame,
            FaceBox(40, 40, 320, 320),
            roi_semantic_basis="detected_face_landmarks",
            allow_frame_edge_contact=True,
        )

    def test_two_local_periodic_regions_raise_local_moire_score(self):
        analyzer = MoirePeriodicPatternPreController()

        result = analyzer.analyze(self._context(periodic_patch_count=2))

        self.assertTrue(result.available)
        self.assertGreater(
            result.raw_features["local_patch_moire_score"],
            result.raw_features["global_moire_score"],
        )
        self.assertGreaterEqual(
            result.raw_features["local_patch_strong_count"],
            2,
        )
        self.assertGreaterEqual(result.raw_score, 70.0)
        heatmap = analyzer.get_local_heatmap()
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape[:2], (320, 320))

    def test_one_patch_or_noise_does_not_satisfy_spatial_vote(self):
        one_patch = MoirePeriodicPatternPreController().analyze(
            self._context(periodic_patch_count=1)
        )
        noise = MoirePeriodicPatternPreController().analyze(
            self._context(periodic_patch_count=0)
        )

        self.assertEqual(
            one_patch.raw_features["local_patch_moire_score"],
            0.0,
        )
        self.assertEqual(
            noise.raw_features["local_patch_moire_score"],
            0.0,
        )

    def test_local_pattern_requires_temporal_persistence_for_warning(self):
        analyzer = MoirePeriodicPatternPreController()
        context = self._context(periodic_patch_count=2)

        results = [analyzer.analyze(context) for _ in range(6)]

        self.assertEqual(results[0].status, "Analysis Uncertain")
        self.assertEqual(results[-1].status, "Suspicious")


if __name__ == "__main__":
    unittest.main()
