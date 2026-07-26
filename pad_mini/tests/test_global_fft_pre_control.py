import unittest
from unittest import mock

import cv2
import numpy as np

import config
from data_models import FaceBox
from global_fft_pre_control import GlobalFFTPreController
from model_free_analysis import ModelFreePreControlContextBuilder


class GlobalFFTPreControlTests(unittest.TestCase):
    def _context(self):
        random = np.random.default_rng(77)
        yy, xx = np.indices((320, 320))
        gray = np.clip(
            127.0
            + 28.0 * np.sin(2.0 * np.pi * xx / 10.0)
            + random.normal(0.0, 7.0, (320, 320)),
            0.0,
            255.0,
        ).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return ModelFreePreControlContextBuilder().build(
            0.0,
            frame,
            frame,
            FaceBox(32, 32, 256, 256),
        )

    def test_returns_extended_raw_spectral_measurements(self):
        result = GlobalFFTPreController().analyze(self._context())
        self.assertTrue(result.available)
        for name in (
            "spectral_flatness",
            "spectral_rolloff_85_radius",
            "spectral_kurtosis_pearson",
            "spectral_slope_fit_mad",
            "dominant_non_dc_peak_ratio",
            "dominant_symmetric_pair_energy_ratio",
            "dominant_pair_amplitude_symmetry",
        ):
            self.assertIn(name, result.raw_features)
            self.assertTrue(np.isfinite(result.raw_features[name]))
        self.assertGreaterEqual(result.raw_features["spectral_flatness"], 0.0)
        self.assertLessEqual(result.raw_features["spectral_flatness"], 1.0)

    def test_hann_hamming_tukey_and_none_windows_are_supported(self):
        windows = {}
        for name in ("hann", "hamming", "tukey", "none"):
            with mock.patch.object(config, "MODEL_FREE_FFT_WINDOW_TYPE", name):
                builder = ModelFreePreControlContextBuilder()
                windows[name] = builder.fft_window
        self.assertTrue(all(window.shape == (256, 256) for window in windows.values()))
        self.assertAlmostEqual(float(windows["none"].min()), 1.0)
        self.assertLess(float(windows["hann"][0, 0]), 1e-8)
        self.assertGreater(float(windows["hamming"][0, 0]), 0.0)
        self.assertAlmostEqual(float(windows["tukey"][0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
