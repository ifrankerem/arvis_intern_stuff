import unittest

from deterministic_normalization import (
    FeatureCalibration,
    normalize_feature,
    robust_z,
    smoothstep,
)


class DeterministicNormalizationTests(unittest.TestCase):
    def test_robust_z_uses_median_and_mad(self):
        self.assertAlmostEqual(
            robust_z(12.9652, 10.0, 2.0),
            1.0,
            places=4,
        )

    def test_high_suspicion_mapping_is_monotonic_and_bounded(self):
        calibration = FeatureCalibration(
            direction="high",
            median_reference=10.0,
            mad_reference=1.0,
            z_start=1.0,
            z_full=4.0,
        )
        values = [normalize_feature(value, calibration) for value in (9, 11, 13, 20)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_low_and_outside_directions_are_explicit(self):
        low = FeatureCalibration(
            direction="low",
            median_reference=10.0,
            mad_reference=1.0,
            z_start=1.0,
            z_full=4.0,
        )
        outside = FeatureCalibration(
            direction="outside",
            valid_interval=(5.0, 15.0),
            transition_width=5.0,
        )
        self.assertGreater(normalize_feature(2.0, low), normalize_feature(9.0, low))
        self.assertEqual(normalize_feature(10.0, outside), 0.0)
        self.assertGreater(normalize_feature(18.0, outside), 0.0)

    def test_smoothstep_rejects_reversed_bounds(self):
        with self.assertRaises(ValueError):
            smoothstep(1.0, 2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
