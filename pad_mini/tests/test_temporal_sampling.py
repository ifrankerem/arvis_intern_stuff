import unittest

import numpy as np

from temporal_sampling import assess_temporal_sampling


class TemporalSamplingTests(unittest.TestCase):
    def test_regular_timestamps_support_temporal_analysis(self):
        timestamps = np.arange(90, dtype=np.float64) / 30.0
        assessment = assess_temporal_sampling(timestamps)
        self.assertTrue(assessment.supported)
        self.assertGreater(assessment.reliability, 0.95)
        self.assertAlmostEqual(assessment.estimated_fps, 30.0, places=4)

    def test_dropped_and_irregular_timestamps_reduce_reliability(self):
        regular = np.arange(90, dtype=np.float64) / 30.0
        irregular_intervals = np.full(89, 1.0 / 30.0)
        irregular_intervals[::8] *= 3.0
        irregular = np.concatenate(([0.0], np.cumsum(irregular_intervals)))

        regular_result = assess_temporal_sampling(regular)
        irregular_result = assess_temporal_sampling(irregular)

        self.assertLess(irregular_result.reliability, regular_result.reliability)
        self.assertIn("dropped-frame gaps are present", irregular_result.warnings)

    def test_insufficient_frames_are_explicitly_unsupported(self):
        timestamps = np.arange(8, dtype=np.float64) / 30.0
        assessment = assess_temporal_sampling(timestamps)
        self.assertFalse(assessment.supported)
        self.assertIn("too few frames", assessment.warnings)

    def test_duplicate_timestamp_is_unsupported(self):
        assessment = assess_temporal_sampling([0.0, 0.1, 0.1, 0.2])
        self.assertFalse(assessment.supported)
        self.assertEqual(assessment.reliability, 0.0)


if __name__ == "__main__":
    unittest.main()
