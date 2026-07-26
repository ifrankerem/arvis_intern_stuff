import time
import unittest

import cv2
import numpy as np

import config
from data_models import FaceBox
from dct_block_pre_control import DCTBlockAnalysisPreController
from model_free_analysis import ModelFreePreControlContextBuilder


class DCTBlockAnalysisTests(unittest.TestCase):
    def _context_for_crop(self, crop):
        frame_gray = np.full((400, 400), 100, dtype=np.uint8)
        frame_gray[72:328, 72:328] = crop
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        aligned = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        return ModelFreePreControlContextBuilder().build(
            time.time(),
            frame,
            frame,
            FaceBox(72, 72, 256, 256),
            aligned_face_crop=aligned,
            pose_alignment_valid=True,
        )

    def test_keeps_required_scores_and_debug_images(self):
        random = np.random.default_rng(13)
        crop = np.clip(
            128 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        analyzer = DCTBlockAnalysisPreController()
        result = analyzer.analyze(self._context_for_crop(crop))

        self.assertTrue(result.available)
        self.assertFalse(result.calibrated)
        for name in (
            "dct_band_anomaly_score",
            "coefficient_sparsity_score",
            "blockiness_score",
            "local_dct_inconsistency_score",
            "final_dct_score",
        ):
            self.assertIn(name, result.raw_features)
            self.assertGreaterEqual(result.raw_features[name], 0.0)
            self.assertLessEqual(result.raw_features[name], 100.0)

        images = analyzer.get_debug_images()
        self.assertEqual(images["dct_band_energy_map"].shape, (256, 256, 3))
        self.assertEqual(
            images["block_boundary_visualization"].shape,
            (256, 256, 3),
        )
        self.assertEqual(images["blockiness_heatmap"].shape, (256, 256, 3))
        self.assertFalse(result.debug_data["encoded_jpeg_bytes_available"])

    def test_detects_strong_eight_pixel_boundary_periodicity(self):
        random = np.random.default_rng(2)
        y_coordinates, x_coordinates = np.indices((256, 256))
        block_offsets = random.integers(-45, 46, (32, 32))
        offsets = np.repeat(
            np.repeat(block_offsets, 8, axis=0),
            8,
            axis=1,
        )
        texture = (
            18 * np.sin(x_coordinates * 1.2)
            + 15 * np.cos(y_coordinates * 1.05)
            + random.normal(0, 4, (256, 256))
        )
        crop = np.clip(128 + offsets + texture, 0, 255).astype(np.uint8)
        analyzer = DCTBlockAnalysisPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_DCT_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertGreater(
            result.raw_features["horizontal_block_boundary_ratio"],
            1.5,
        )
        self.assertGreater(
            result.raw_features["vertical_block_boundary_ratio"],
            1.5,
        )
        self.assertEqual(
            result.status,
            "Compression-like block structure detected",
        )

    def test_blurred_crop_is_unavailable_via_shared_quality_gate(self):
        crop = np.full((256, 256), 128, dtype=np.uint8)
        analyzer = DCTBlockAnalysisPreController()
        result = analyzer.analyze(self._context_for_crop(crop))

        self.assertFalse(result.available)
        self.assertEqual(result.status, "Analysis unavailable")
        self.assertIsNone(result.score)

    def test_detects_inner_patch_smoothing_inconsistency(self):
        random = np.random.default_rng(9)
        crop = np.clip(
            125 + random.normal(0, 22, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        crop[96:160, 96:160] = cv2.GaussianBlur(
            crop[96:160, 96:160],
            (15, 15),
            5,
        )
        analyzer = DCTBlockAnalysisPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_DCT_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertGreater(
            result.raw_features["local_dct_inconsistency_score"],
            config.EXPERIMENTAL_DCT_LOCAL_INCONSISTENCY_SCORE,
        )
        self.assertEqual(result.status, "Local DCT inconsistency")


if __name__ == "__main__":
    unittest.main()
