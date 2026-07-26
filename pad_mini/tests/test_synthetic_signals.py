import unittest

import cv2
import numpy as np

from synthetic_signals import (
    add_eight_pixel_block_steps,
    add_periodic_residual,
    checkerboard,
    chromaticity_shift,
    clip_colors,
    irregular_timestamps,
    jpeg_recompress,
    nonplanar_motion_sequence,
    planar_motion_sequence,
    pulse_rgb_signals,
    resize_roundtrip,
    screen_lattice,
    sharpen,
    sinusoidal_grating,
    temporal_sinusoid,
)


class SyntheticSignalFixtureTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(202)
        self.noise = np.clip(
            127 + random.normal(0, 18, (256, 256)),
            0,
            255,
        ).astype(np.uint8)

    def test_periodic_spatial_fixtures_have_expected_sizes(self):
        for image in (
            sinusoidal_grating(),
            checkerboard(),
            screen_lattice(),
            add_periodic_residual(self.noise),
            add_eight_pixel_block_steps(self.noise),
        ):
            self.assertEqual(image.shape, (256, 256))
            self.assertEqual(image.dtype, np.uint8)

    def test_blur_sharpen_and_resize_change_gradient_energy_predictably(self):
        blurred = cv2.GaussianBlur(self.noise, (0, 0), 2.5)
        sharpened = sharpen(self.noise, amount=1.5)
        resized = resize_roundtrip(self.noise, scale=0.45)
        lap = lambda image: float(cv2.Laplacian(image, cv2.CV_64F).var())
        self.assertLess(lap(blurred), lap(self.noise))
        self.assertGreater(lap(sharpened), lap(self.noise))
        self.assertLess(lap(resized), lap(self.noise))

    def test_jpeg_and_double_jpeg_fixtures_are_decoded_not_encoded_evidence(self):
        q90 = jpeg_recompress(self.noise, [90])
        q40 = jpeg_recompress(self.noise, [40])
        double = jpeg_recompress(self.noise, [85, 45])
        self.assertEqual(q90.shape, self.noise.shape)
        self.assertGreater(
            float(np.mean(np.abs(q40.astype(float) - self.noise))),
            float(np.mean(np.abs(q90.astype(float) - self.noise))),
        )
        self.assertEqual(double.dtype, np.uint8)

    def test_color_clipping_and_chromaticity_shift_are_controlled(self):
        bgr = cv2.cvtColor(self.noise, cv2.COLOR_GRAY2BGR)
        shifted = chromaticity_shift(bgr, blue_gain=1.3, red_gain=0.7)
        clipped = clip_colors(shifted, 40, 200)
        self.assertGreater(float(shifted[:, :, 0].mean()), float(shifted[:, :, 2].mean()))
        self.assertGreaterEqual(int(clipped.min()), 40)
        self.assertLessEqual(int(clipped.max()), 200)

    def test_temporal_flicker_and_pulse_signals_have_requested_peak(self):
        timestamps = np.arange(180, dtype=float) / 30.0
        signal = temporal_sinusoid(timestamps, 4.0)
        frequencies = np.fft.rfftfreq(signal.size, d=1.0 / 30.0)
        peak = frequencies[1 + np.argmax(np.abs(np.fft.rfft(signal))[1:])]
        self.assertAlmostEqual(peak, 4.0, places=5)
        rgb = pulse_rgb_signals(timestamps, pulse_hz=1.2)
        self.assertEqual(rgb.shape, (180, 3))
        self.assertGreater(np.std(rgb[:, 1]), np.std(rgb[:, 2]))
        self.assertTrue(np.any(np.diff(irregular_timestamps()) > 2.0 / 30.0))

    def test_planar_and_nonplanar_motion_sequences_are_distinct(self):
        base = checkerboard(128, 8)
        planar = planar_motion_sequence(base, frame_count=5)
        nonplanar = nonplanar_motion_sequence(base, frame_count=5)
        self.assertEqual(len(planar), 5)
        self.assertEqual(len(nonplanar), 5)
        difference = np.mean(
            np.abs(planar[-1].astype(float) - nonplanar[-1].astype(float))
        )
        self.assertGreater(difference, 0.5)


if __name__ == "__main__":
    unittest.main()
