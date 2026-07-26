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
from model_free_analysis import ModelFreePreControlContextBuilder
from model_free_pre_control_application import ModelFreePreControlApplication
import wavelet_pre_control
from wavelet_pre_control import WaveletAnalysisPreController


class _FakeWaveletDefinition:
    dec_len = 2


class _FakePyWavelets:
    @staticmethod
    def Wavelet(_name):
        return _FakeWaveletDefinition()

    @staticmethod
    def dwt_max_level(size, _filter_length):
        return int(np.floor(np.log2(size)))

    @staticmethod
    def dwt2(values, _wavelet, mode=None):
        del mode
        values = np.asarray(values, dtype=np.float64)
        top_left = values[0::2, 0::2]
        top_right = values[0::2, 1::2]
        bottom_left = values[1::2, 0::2]
        bottom_right = values[1::2, 1::2]
        approximation = (
            top_left + top_right + bottom_left + bottom_right
        ) / 2.0
        horizontal = (
            top_left + top_right - bottom_left - bottom_right
        ) / 2.0
        vertical = (
            top_left - top_right + bottom_left - bottom_right
        ) / 2.0
        diagonal = (
            top_left - top_right - bottom_left + bottom_right
        ) / 2.0
        return approximation, (horizontal, vertical, diagonal)


class WaveletAnalysisTests(unittest.TestCase):
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

    def _dependency_available_patch(self):
        return mock.patch.multiple(
            wavelet_pre_control,
            pywt=_FakePyWavelets,
            PYWAVELETS_AVAILABLE=True,
        )

    def test_missing_dependency_returns_unavailable_without_score(self):
        random = np.random.default_rng(21)
        crop = np.clip(
            125 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        with mock.patch.multiple(
            wavelet_pre_control,
            pywt=None,
            PYWAVELETS_AVAILABLE=False,
        ):
            result = WaveletAnalysisPreController().analyze(
                self._context_for_crop(crop)
            )

        self.assertFalse(result.available)
        self.assertEqual(result.status, "Analysis unavailable")
        self.assertIsNone(result.score)
        self.assertEqual(result.debug_data["dependency"], "PyWavelets")
        self.assertFalse(result.debug_data["dependency_available"])
        self.assertIn("PyWavelets dependency", result.warning)

    def test_two_level_features_scores_and_heatmap(self):
        random = np.random.default_rng(22)
        crop = np.clip(
            125 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        with self._dependency_available_patch():
            analyzer = WaveletAnalysisPreController()
            result = analyzer.analyze(self._context_for_crop(crop))
            subbands = analyzer.get_debug_subbands()
            heatmap = analyzer.get_anomaly_heatmap()

        self.assertTrue(result.available)
        self.assertFalse(result.calibrated)
        for name in (
            "wavelet_energy_score",
            "directional_wavelet_score",
            "local_wavelet_inconsistency_score",
            "final_wavelet_score",
        ):
            self.assertIn(name, result.raw_features)
            self.assertGreaterEqual(result.raw_features[name], 0.0)
            self.assertLessEqual(result.raw_features[name], 100.0)

        self.assertEqual(set(subbands), {1, 2})
        self.assertEqual(set(subbands[1]), {"LL", "LH", "HL", "HH"})
        self.assertEqual(subbands[1]["LL"].shape, (128, 128))
        self.assertEqual(subbands[2]["LL"].shape, (64, 64))
        self.assertEqual(heatmap.shape, (256, 256, 3))
        self.assertIn("not neural-network attention", result.debug_data[
            "heatmap_interpretation"
        ])

    def test_invalid_frame_is_not_added_to_temporal_history(self):
        random = np.random.default_rng(23)
        valid_crop = np.clip(
            125 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        invalid_crop = np.full((256, 256), 128, dtype=np.uint8)
        with self._dependency_available_patch():
            analyzer = WaveletAnalysisPreController()
            valid_result = analyzer.analyze(
                self._context_for_crop(valid_crop)
            )
            history_length = len(analyzer.score_history)
            invalid_result = analyzer.analyze(
                self._context_for_crop(invalid_crop)
            )

        self.assertTrue(valid_result.available)
        self.assertFalse(invalid_result.available)
        self.assertEqual(history_length, 1)
        self.assertEqual(len(analyzer.score_history), history_length)

    def test_severe_clipping_is_unavailable(self):
        random = np.random.default_rng(24)
        crop = np.clip(
            125 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        crop[:, 82:174] = 255
        with self._dependency_available_patch():
            result = WaveletAnalysisPreController().analyze(
                self._context_for_crop(crop)
            )

        self.assertFalse(result.available)
        self.assertEqual(result.status, "Analysis unavailable")
        self.assertGreaterEqual(
            result.raw_features["clipped_pixel_ratio"],
            config.WAVELET_UNAVAILABLE_CLIPPING_RATIO,
        )

    def test_debug_export_writes_raw_and_normalized_subbands(self):
        random = np.random.default_rng(25)
        gray = np.clip(
            125 + random.normal(0, 18, (480, 640)),
            0,
            255,
        ).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with self._dependency_available_patch(), mock.patch.object(
                config,
                "WAVELET_DEBUG_DIRECTORY",
                output_directory,
            ):
                application = ModelFreePreControlApplication(
                    enable_face_detection=False
                )
                application.process_frame(frame)
                result = application.latest_pre_control_results["wavelet"]
                with redirect_stdout(io.StringIO()):
                    saved = application._save_wavelet_debug(
                        "test_timestamp",
                        application.latest_context,
                    )
                application.shutdown()

            self.assertTrue(result.available)
            self.assertTrue(saved)
            self.assertEqual(len(list(output_directory.glob("*_raw_*.npy"))), 8)
            self.assertEqual(
                len(list(output_directory.glob("*_normalized_*.png"))),
                8,
            )
            self.assertEqual(
                len(list(output_directory.glob("wavelet_anomaly_heatmap_*.png"))),
                1,
            )
            self.assertEqual(
                len(list(output_directory.glob("wavelet_feature_report_*.json"))),
                1,
            )

    def test_repeated_inner_smoothing_is_local_detail_inconsistency(self):
        random = np.random.default_rng(26)
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
        with self._dependency_available_patch():
            analyzer = WaveletAnalysisPreController()
            context = self._context_for_crop(crop)
            for _ in range(config.EXPERIMENTAL_WAVELET_MINIMUM_HISTORY):
                result = analyzer.analyze(context)

        self.assertGreaterEqual(
            result.raw_features["local_wavelet_inconsistency_score"],
            config.EXPERIMENTAL_WAVELET_LOCAL_STATUS_SCORE,
        )
        self.assertEqual(result.status, "Local detail inconsistency")

    def test_strong_oriented_texture_is_directional_anomaly(self):
        _y_coordinates, x_coordinates = np.indices((256, 256))
        random = np.random.default_rng(27)
        crop = np.clip(
            128
            + 45 * np.sin(2 * np.pi * x_coordinates / 8.0)
            + random.normal(0, 3, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        with self._dependency_available_patch():
            analyzer = WaveletAnalysisPreController()
            context = self._context_for_crop(crop)
            for _ in range(config.EXPERIMENTAL_WAVELET_MINIMUM_HISTORY):
                result = analyzer.analyze(context)

        self.assertGreaterEqual(
            result.raw_features["directional_wavelet_score"],
            config.EXPERIMENTAL_WAVELET_DIRECTIONAL_STATUS_SCORE,
        )
        self.assertEqual(result.status, "Directional wavelet anomaly")

    def test_compatible_calibration_profile_is_applied(self):
        random = np.random.default_rng(28)
        crop = np.clip(
            125 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "wavelet_analysis": {
                            "wavelet_name": config.WAVELET_NAME,
                            "decomposition_levels": (
                                config.WAVELET_DECOMPOSITION_LEVELS
                            ),
                            "feature_profiles": {
                                "global_detail_sparsity_mean": {
                                    "mean": 0.5,
                                    "standard_deviation": 0.5,
                                    "weight": 1.0,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self._dependency_available_patch(), mock.patch.object(
                config,
                "MODEL_FREE_CALIBRATION_FILE_PATH",
                calibration_path,
            ):
                result = WaveletAnalysisPreController().analyze(
                    self._context_for_crop(crop)
                )

        self.assertTrue(result.available)
        self.assertTrue(result.calibrated)
        self.assertIsNotNone(
            result.raw_features["calibrated_feature_anomaly_score"]
        )


if __name__ == "__main__":
    unittest.main()
