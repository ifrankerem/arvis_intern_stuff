import unittest

from benchmark_metrics import evaluate_pad_scores


class BenchmarkMetricTests(unittest.TestCase):
    def test_apcer_bpcer_acer_and_abstention_are_separate(self):
        records = [
            {"label": "attack", "attack_species": "replay", "score": 80, "score_valid": True, "classification": "SUSPICIOUS", "runtime_ms": 10},
            {"label": "attack", "attack_species": "replay", "score": 40, "score_valid": True, "classification": "SUSPICIOUS", "runtime_ms": 12},
            {"label": "bona_fide", "attack_species": "none", "score": 60, "score_valid": True, "classification": "SUSPICIOUS", "runtime_ms": 8},
            {"label": "bona_fide", "attack_species": "none", "score": 20, "score_valid": True, "classification": "INSUFFICIENT_EVIDENCE", "runtime_ms": 9},
            {"label": "attack", "attack_species": "print", "score": None, "score_valid": False, "classification": "UNSUPPORTED_CAPTURE", "runtime_ms": 2},
        ]
        metrics = evaluate_pad_scores(records, threshold=50.0)
        self.assertAlmostEqual(metrics["APCER"], 0.5)
        self.assertAlmostEqual(metrics["BPCER"], 0.5)
        self.assertAlmostEqual(metrics["ACER"], 0.5)
        self.assertAlmostEqual(metrics["abstention_rate"], 0.4)
        self.assertAlmostEqual(metrics["unsupported_rate"], 0.2)
        self.assertEqual(metrics["per_attack_confusion"]["replay"]["missed"], 1)


if __name__ == "__main__":
    unittest.main()
