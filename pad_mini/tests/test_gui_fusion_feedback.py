from types import SimpleNamespace
import unittest

from application_gui import FaceQualityGui


class GuiFusionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.gui = FaceQualityGui.__new__(FaceQualityGui)

    def _feedback(self, status):
        return self.gui._plain_language_fusion_feedback(
            SimpleNamespace(status=status)
        )

    def test_normal_feedback_is_plain_and_non_absolute(self):
        text, color = self._feedback("Normal mathematical evidence")

        self.assertIn("belirgin", text)
        self.assertIn("izi görülmedi", text)
        self.assertIn("canlılık kanıtı değildir", text)
        self.assertEqual(color, FaceQualityGui.SUCCESS)

    def test_weak_feedback_mentions_possible_attack_and_retry(self):
        text, color = self._feedback("Weak anomaly evidence")

        self.assertIn("saldırısı olabilir", text)
        self.assertIn("tekrar deneyin", text)
        self.assertEqual(color, FaceQualityGui.WARNING)

    def test_suspicious_feedback_is_a_cautious_attack_warning(self):
        text, color = self._feedback(
            "Suspicious mathematical evidence"
        )

        self.assertIn("UYARI", text)
        self.assertIn("saldırısı olabilir", text)
        self.assertIn("kesin bir karar değildir", text)
        self.assertEqual(color, FaceQualityGui.ERROR)

    def test_inconclusive_feedback_gives_actionable_capture_guidance(self):
        text, color = self._feedback("Inconclusive")

        self.assertIn("yeterli güvenilir veri yok", text)
        self.assertIn("iyi ışıkta", text)
        self.assertEqual(color, FaceQualityGui.WARNING)


class GuiLiveFrequencyTests(unittest.TestCase):
    def setUp(self):
        self.gui = FaceQualityGui.__new__(FaceQualityGui)

    def test_live_frequency_metrics_include_bands_and_direction(self):
        results = {
            "fft": SimpleNamespace(
                available=True,
                raw_features={
                    "low_frequency_energy_ratio": 0.51,
                    "middle_frequency_energy_ratio": 0.32,
                    "high_frequency_energy_ratio": 0.17,
                    "spectral_centroid": 0.2842,
                },
            ),
            "radial_angular": SimpleNamespace(
                available=True,
                raw_features={
                    "dominant_radial_frequency": 0.375,
                    "dominant_frequency_angle_degrees": 42.25,
                },
            ),
        }

        text = self.gui._format_live_frequency_metrics(results)

        self.assertIn("Düşük bant: 51.0%", text)
        self.assertIn("Orta bant: 32.0%", text)
        self.assertIn("Yüksek bant: 17.0%", text)
        self.assertIn("Spektral merkez: 0.284", text)
        self.assertIn("Baskın frekans: 0.375", text)
        self.assertIn("Baskın yön: 42.2°", text)

    def test_unavailable_fft_returns_capture_guidance(self):
        results = {
            "fft": SimpleNamespace(available=False, raw_features={})
        }

        text = self.gui._format_live_frequency_metrics(results)

        self.assertIn("FFT verisi bekleniyor", text)
        self.assertIn("iyi ışıkta", text)

    def test_non_finite_frequency_metric_is_not_displayed(self):
        self.assertEqual(
            self.gui._format_frequency_metric(float("nan")),
            "--",
        )


if __name__ == "__main__":
    unittest.main()
