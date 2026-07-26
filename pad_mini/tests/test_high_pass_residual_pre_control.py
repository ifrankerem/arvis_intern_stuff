from contextlib import redirect_stdout
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import config
from data_models import FaceBox
from high_pass_residual_pre_control import HighPassResidualPreController
from model_free_analysis import ModelFreePreControlContextBuilder
from model_free_pre_control_application import ModelFreePreControlApplication


class HighPassResidualAnalysisTests(unittest.TestCase):
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

    def _normal_crop(self, seed=31):
        random = np.random.default_rng(seed)
        y_coordinates, x_coordinates = np.indices((256, 256))
        return np.clip(
            125
            + 15 * np.sin(x_coordinates / 15.0)
            + 12 * np.cos(y_coordinates / 21.0)
            + random.normal(0, 6, (256, 256)),
            0,
            255,
        ).astype(np.uint8)

    def test_keeps_required_features_signed_residuals_and_debug_images(self):
        analyzer = HighPassResidualPreController()
        result = analyzer.analyze(
            self._context_for_crop(self._normal_crop())
        )

        self.assertTrue(result.available)
        self.assertFalse(result.calibrated)
        for name in (
            "residual_variance",
            "residual_mean_absolute_deviation",
            "residual_rms_energy",
            "residual_entropy",
            "residual_kurtosis",
            "positive_negative_residual_balance",
            "laplacian_variance",
            "gradient_energy",
            "high_frequency_edge_density",
            "local_residual_consistency",
            "patch_residual_energy_variation",
            "gaussian_residual_score",
            "laplacian_score",
            "gradient_score",
            "local_residual_inconsistency_score",
            "final_residual_score",
        ):
            self.assertIn(name, result.raw_features)

        self.assertEqual(analyzer.latest_gaussian_residual.dtype, np.float32)
        self.assertLess(float(np.min(analyzer.latest_gaussian_residual)), 0.0)
        self.assertGreater(float(np.max(analyzer.latest_gaussian_residual)), 0.0)
        self.assertEqual(analyzer.latest_laplacian_response.dtype, np.float32)

        images = analyzer.get_debug_images()
        self.assertEqual(set(images), {
            "gaussian_residual",
            "laplacian",
            "gradient_magnitude",
            "patch_residual_energy_map",
        })
        self.assertEqual(images["gaussian_residual"].shape, (256, 256))
        self.assertEqual(images["laplacian"].shape, (256, 256))
        self.assertEqual(images["gradient_magnitude"].shape, (256, 256))
        self.assertEqual(
            images["patch_residual_energy_map"].shape,
            (256, 256, 3),
        )
        for image in images.values():
            self.assertEqual(image.dtype, np.uint8)

    def test_temporal_normal_status_and_uncalibrated_confidence_cap(self):
        analyzer = HighPassResidualPreController()
        context = self._context_for_crop(self._normal_crop(seed=32))
        for _ in range(config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertEqual(result.status, "Normal residual structure")
        self.assertEqual(result.raw_score, result.raw_features[
            "final_residual_score"
        ])
        self.assertIsNotNone(result.stabilized_score)
        self.assertLessEqual(
            result.confidence,
            config.EXPERIMENTAL_RESIDUAL_MAXIMUM_CONFIDENCE,
        )

    def test_excessive_high_frequency_residual_is_not_called_fraud(self):
        random = np.random.default_rng(33)
        crop = np.clip(
            125 + random.normal(0, 55, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        analyzer = HighPassResidualPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertEqual(result.status, "Excessive high-frequency residual")
        self.assertGreater(
            result.raw_features["residual_energy_anomaly_direction"],
            0.15,
        )
        self.assertEqual(result.attack_type, "none")
        self.assertTrue(result.debug_data["supporting_evidence_only"])
        self.assertIn("noise or sharpening", " ".join(result.warnings))

    def test_smooth_inner_face_can_produce_low_sided_anomaly(self):
        y_coordinates, x_coordinates = np.indices((256, 256))
        crop = np.clip(
            125
            + 15 * np.sin(x_coordinates / 40.0)
            + 15 * np.cos(y_coordinates / 45.0),
            0,
            255,
        ).astype(np.uint8)
        border = (
            (x_coordinates < 30)
            | (x_coordinates > 225)
            | (y_coordinates < 35)
            | (y_coordinates > 225)
        )
        crop[border] = np.clip(
            125 + 25 * ((x_coordinates[border] + y_coordinates[border]) % 2),
            0,
            255,
        ).astype(np.uint8)
        analyzer = HighPassResidualPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertEqual(result.status, "Abnormally smooth residual")
        self.assertLess(
            result.raw_features["residual_energy_anomaly_direction"],
            -0.15,
        )

    def test_repeated_inner_smoothing_is_local_inconsistency(self):
        random = np.random.default_rng(34)
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
        analyzer = HighPassResidualPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertEqual(result.status, "Local residual inconsistency")
        self.assertGreaterEqual(
            result.raw_features["local_residual_inconsistency_score"],
            config.EXPERIMENTAL_RESIDUAL_LOCAL_STATUS_SCORE,
        )

    def test_invalid_frame_does_not_enter_temporal_history(self):
        analyzer = HighPassResidualPreController()
        valid_result = analyzer.analyze(
            self._context_for_crop(self._normal_crop(seed=35))
        )
        history_length = len(analyzer.score_history)
        invalid_result = analyzer.analyze(
            self._context_for_crop(
                np.full((256, 256), 128, dtype=np.uint8)
            )
        )

        self.assertTrue(valid_result.available)
        self.assertFalse(invalid_result.available)
        self.assertEqual(invalid_result.status, "Analysis unavailable")
        self.assertIsNone(invalid_result.score)
        self.assertEqual(len(analyzer.score_history), history_length)

    def test_low_light_noise_reduces_confidence_and_returns_uncertain(self):
        random = np.random.default_rng(36)
        crop = np.clip(
            62 + random.normal(0, 20, (256, 256)),
            3,
            180,
        ).astype(np.uint8)
        analyzer = HighPassResidualPreController()
        context = self._context_for_crop(crop)
        for _ in range(config.EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY):
            result = analyzer.analyze(context)

        self.assertTrue(result.available)
        self.assertEqual(result.status, "Analysis uncertain")
        self.assertLess(
            result.confidence,
            config.EXPERIMENTAL_RESIDUAL_MAXIMUM_CONFIDENCE,
        )
        self.assertIn("low light", " ".join(result.warnings))

    def test_compatible_bona_fide_calibration_is_applied(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "high_pass_residual_analysis": {
                            "feature_profiles": {
                                "gaussian_residual_rms_energy": {
                                    "mean": 5.0,
                                    "standard_deviation": 2.0,
                                    "normal_z_limit": 2.0,
                                    "weight": 1.0,
                                },
                                "laplacian_variance": {
                                    "mean": 2800.0,
                                    "standard_deviation": 1000.0,
                                    "weight": 1.0,
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                config,
                "MODEL_FREE_CALIBRATION_FILE_PATH",
                calibration_path,
            ):
                result = HighPassResidualPreController().analyze(
                    self._context_for_crop(self._normal_crop(seed=37))
                )

        self.assertTrue(result.available)
        self.assertTrue(result.calibrated)
        self.assertTrue(result.raw_features["calibrated_component_scores"])

    def test_debug_export_saves_four_images_and_feature_report(self):
        random = np.random.default_rng(38)
        gray = np.clip(
            125 + random.normal(0, 18, (480, 640)),
            0,
            255,
        ).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with mock.patch.object(
                config,
                "RESIDUAL_DEBUG_DIRECTORY",
                output_directory,
            ):
                application = ModelFreePreControlApplication()
                application.process_frame(frame)
                result = application.latest_pre_control_results["residual"]
                with redirect_stdout(io.StringIO()):
                    saved = application._save_residual_debug(
                        "test_timestamp",
                        application.latest_context,
                    )
                application.shutdown()

            self.assertTrue(result.available)
            self.assertTrue(saved)
            self.assertEqual(len(list(output_directory.glob("*.png"))), 4)
            report_paths = list(
                output_directory.glob("residual_feature_report_*.json")
            )
            self.assertEqual(len(report_paths), 1)
            report = json.loads(report_paths[0].read_text(encoding="utf-8"))
            self.assertTrue(
                report["preprocessing"]
                ["signed_residuals_retained_for_analysis"]
            )
            self.assertIn("No Noiseprint", " ".join(report["limitations"]))


if __name__ == "__main__":
    unittest.main()
