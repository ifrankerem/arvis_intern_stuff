import unittest

import cv2
import numpy as np

from data_models import FaceBox
from model_free_analysis import ModelFreePreControlContextBuilder
from radial_angular_pre_control import RadialAngularSpectrumPreController


class SyntheticFrequencyBehaviorTests(unittest.TestCase):
    def _dominant_angle(self, grating_angle_degrees):
        yy, xx = np.indices((320, 320))
        angle = np.deg2rad(grating_angle_degrees)
        coordinate = xx * np.cos(angle) + yy * np.sin(angle)
        gray = 127.0 + 80.0 * np.sin(2.0 * np.pi * coordinate / 8.0)
        frame = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        context = ModelFreePreControlContextBuilder().build(
            0.0,
            frame,
            frame,
            FaceBox(32, 32, 256, 256),
        )
        result = RadialAngularSpectrumPreController().analyze(context)
        self.assertTrue(result.available)
        return result.raw_features["dominant_frequency_angle_degrees"]

    def test_rotating_grating_rotates_angular_spectrum_peak(self):
        angle_0 = self._dominant_angle(0.0)
        angle_60 = self._dominant_angle(60.0)
        axial_difference = abs(angle_60 - angle_0) % 180.0
        axial_difference = min(axial_difference, 180.0 - axial_difference)
        self.assertAlmostEqual(axial_difference, 60.0, delta=7.0)


if __name__ == "__main__":
    unittest.main()
