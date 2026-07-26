import unittest

import cv2
import numpy as np

from data_models import FaceBox
from model_free_analysis import ModelFreePreControlContextBuilder
from periodicity_pre_control import PeriodicityPreController


class PeriodicityPreControlTests(unittest.TestCase):
    def _analyze(self, gray):
        frame = cv2.cvtColor(
            np.clip(gray, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )
        context = ModelFreePreControlContextBuilder().build(
            0.0,
            frame,
            frame,
            FaceBox(32, 32, 256, 256),
        )
        return PeriodicityPreController().analyze(context)

    def test_increasing_periodic_amplitude_increases_measurement(self):
        random = np.random.default_rng(44)
        yy, xx = np.indices((320, 320))
        base = 125.0 + random.normal(0.0, 8.0, (320, 320))
        weak = base + 5.0 * np.sin(2.0 * np.pi * xx / 12.0)
        strong = base + 25.0 * np.sin(2.0 * np.pi * xx / 12.0)

        baseline_result = self._analyze(base)
        weak_result = self._analyze(weak)
        strong_result = self._analyze(strong)

        self.assertTrue(baseline_result.available)
        self.assertTrue(weak_result.available)
        self.assertTrue(strong_result.available)
        self.assertGreater(
            weak_result.raw_features["autocorrelation_peak_ratio"],
            baseline_result.raw_features["autocorrelation_peak_ratio"],
        )
        self.assertGreater(
            strong_result.raw_features["autocorrelation_peak_ratio"],
            weak_result.raw_features["autocorrelation_peak_ratio"],
        )
        self.assertGreater(strong_result.raw_score, weak_result.raw_score)
        self.assertGreater(
            strong_result.raw_features["patch_periodic_vote_ratio"],
            weak_result.raw_features["patch_periodic_vote_ratio"],
        )

    def test_rotated_periodic_pattern_remains_detectable(self):
        random = np.random.default_rng(45)
        yy, xx = np.indices((320, 320))
        angle = np.deg2rad(37.0)
        coordinate = xx * np.cos(angle) + yy * np.sin(angle)
        gray = (
            125.0
            + 28.0 * np.sin(2.0 * np.pi * coordinate / 10.0)
            + random.normal(0.0, 4.0, (320, 320))
        )

        result = self._analyze(gray)

        self.assertTrue(result.available)
        self.assertGreater(result.raw_score, 55.0)
        self.assertGreater(
            result.raw_features["patch_periodic_vote_ratio"],
            0.50,
        )
        self.assertIn(
            "autocorrelation_peak_coordinate",
            result.debug_data,
        )

    def test_small_or_invalid_input_is_unsupported_without_score(self):
        gray = np.full((80, 80), 125, dtype=np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        context = ModelFreePreControlContextBuilder().build(
            0.0,
            frame,
            frame,
            FaceBox(5, 5, 70, 70),
        )

        result = PeriodicityPreController().analyze(context)

        self.assertFalse(result.supported)
        self.assertIsNone(result.score)
        self.assertEqual(result.reliability, 0.0)
        self.assertIn("PERIODICITY_UNSUPPORTED", result.reason_codes)

    def test_debug_maps_correspond_to_computed_domains(self):
        yy, xx = np.indices((320, 320))
        gray = 127.0 + 35.0 * np.sin(2.0 * np.pi * xx / 8.0)
        frame = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        context = ModelFreePreControlContextBuilder().build(
            0.0,
            frame,
            frame,
            FaceBox(32, 32, 256, 256),
        )
        analyzer = PeriodicityPreController()

        result = analyzer.analyze(context)
        images = analyzer.get_debug_images()

        self.assertTrue(result.available)
        self.assertEqual(
            set(images),
            {
                "autocorrelation_map",
                "cepstrum_map",
                "patch_periodicity_heatmap",
            },
        )
        self.assertTrue(all(image.size > 0 for image in images.values()))


if __name__ == "__main__":
    unittest.main()
