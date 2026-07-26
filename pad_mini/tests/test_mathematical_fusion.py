from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import config
from application_gui import FaceQualityGui
from data_models import FaceBox
from mathematical_fusion import MathematicalFusionController
from model_free_analysis import ModelFreeAnalysisResult
from model_free_pre_control_application import ModelFreePreControlApplication


class MathematicalFusionTests(unittest.TestCase):
    MODULE_NAMES = (
        "fft",
        "moire",
        "radial_angular",
        "dct_block",
        "wavelet",
        "residual",
    )

    def _context(self, valid=True):
        return SimpleNamespace(
            face_quality_valid=valid,
            quality_reason=None if valid else "quality gate failed",
            face_bounding_box=FaceBox(10, 10, 256, 256),
            blur_value=180.0,
            brightness_value=125.0,
            exposure_valid=True,
            pose_alignment_valid=True,
            face_dimensions=(256, 256),
            analysis_dimensions=(256, 256),
        )

    def _context_with_frame(self, frame):
        context = self._context()
        context.analysis_frame = frame
        context.face_bounding_box = FaceBox(220, 80, 200, 200)
        return context

    def _result(
        self,
        module_name,
        score,
        confidence=1.0,
        available=True,
        features=None,
        status="Normal",
        debug_data=None,
        calibrated=False,
    ):
        details = {"quality_status": "Sufficient", "possible_attack": "none"}
        details.update(debug_data or {})
        return ModelFreeAnalysisResult(
            module_name=module_name,
            available=available,
            raw_features=dict(features or {}),
            raw_score=score if available else None,
            stabilized_score=score if available else None,
            confidence=confidence if available else 0.0,
            status=status if available else "Analysis unavailable",
            evidence=[],
            warnings=[],
            debug_data=details,
            calibrated=calibrated,
        )

    def _results(self, scores=None, confidences=None):
        scores = scores or {}
        confidences = confidences or {}
        return {
            name: self._result(
                name,
                scores.get(name, 0.0),
                confidences.get(name, 1.0),
            )
            for name in self.MODULE_NAMES
        }

    def test_fft_modules_enter_final_fusion_as_one_correlated_family(self):
        scores = {
            "fft": 100.0,
            "moire": 100.0,
            "radial_angular": 100.0,
            "dct_block": 0.0,
            "wavelet": 0.0,
            "residual": 0.0,
        }
        result = MathematicalFusionController().analyze(
            self._results(scores),
            self._context(),
        )

        self.assertAlmostEqual(result.raw_features["fft_family_score"], 100.0)
        self.assertAlmostEqual(
            result.raw_features["local_transform_score"],
            0.0,
        )
        self.assertAlmostEqual(
            result.raw_score,
            100.0
            * config.MATHEMATICAL_FUSION_CONFIG["group_weights"][
                "fft_family"
            ],
        )
        self.assertTrue(
            result.debug_data[
                "shared_fft_family_counted_as_one_final_group"
            ]
        )

    def test_unavailable_module_is_excluded_instead_of_scored_as_zero(self):
        results = self._results(
            {
                "moire": 50.0,
                "radial_angular": 100.0,
                "dct_block": 0.0,
                "wavelet": 0.0,
                "residual": 0.0,
            }
        )
        results["fft"] = self._result("fft", None, available=False)
        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        expected_fft_family = 75.0
        self.assertAlmostEqual(
            result.raw_features["fft_family_score"],
            expected_fft_family,
        )
        contribution = result.raw_features["module_contributions"]["fft"]
        self.assertFalse(contribution["included"])
        self.assertEqual(contribution["exclusion_reason"], "module unavailable")
        self.assertEqual(result.raw_features["valid_module_count"], 5)

    def test_confidence_changes_the_weighted_group_score(self):
        results = self._results(
            {
                "fft": 100.0,
                "moire": 0.0,
                "radial_angular": 0.0,
            },
            {"fft": 0.1, "moire": 1.0, "radial_angular": 1.0},
        )
        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        weights = config.MATHEMATICAL_FUSION_CONFIG["module_weights"]
        expected = (
            100.0 * weights["fft"] * 0.1
            / (
                weights["fft"] * 0.1
                + weights["moire"]
                + weights["radial_angular"]
            )
        )
        self.assertAlmostEqual(result.raw_features["fft_family_score"], expected)

    def test_available_but_uncertain_quality_module_is_excluded(self):
        results = self._results()
        results["wavelet"] = self._result(
            "wavelet",
            100.0,
            debug_data={"quality_status": "Uncertain"},
        )
        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        wavelet = result.raw_features["module_contributions"]["wavelet"]
        self.assertFalse(wavelet["included"])
        self.assertEqual(
            wavelet["exclusion_reason"],
            "module quality is uncertain or invalid",
        )
        self.assertEqual(result.raw_features["valid_module_count"], 5)
        self.assertEqual(result.raw_features["local_transform_score"], 0.0)

    def test_supported_severe_clipping_is_preserved_as_presentation_evidence(self):
        results = self._results()
        results["fft"] = self._result(
            "fft",
            0.0,
            features={
                "middle_frequency_energy_ratio": 0.105,
                "spectral_entropy": 0.63,
            },
        )
        results["dct_block"] = self._result(
            "dct_block",
            14.0,
            features={"local_dct_inconsistency_score": 51.0},
        )
        for name in ("wavelet", "residual"):
            results[name] = self._result(
                name,
                None,
                available=False,
                features={"clipped_pixel_ratio": 0.32},
            )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(results, self._context())

        cue = result.raw_features["presentation_artifact"]
        self.assertTrue(cue["support_is_sufficient"])
        self.assertGreater(
            result.raw_features["presentation_artifact_score"],
            75.0,
        )
        self.assertLess(result.raw_score, 25.0)
        self.assertGreater(
            result.raw_features["score_summary"][
                "temporal_decision_score"
            ],
            75.0,
        )
        self.assertEqual(result.status, "High mathematical risk")
        self.assertEqual(result.attack_type, "presentation_attack")
        self.assertIn("Severe highlight clipping", " ".join(result.evidence))

    def test_severe_clipping_alone_does_not_claim_an_attack(self):
        results = self._results()
        for name in ("wavelet", "residual"):
            results[name] = self._result(
                name,
                None,
                available=False,
                features={"clipped_pixel_ratio": 0.32},
            )

        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        cue = result.raw_features["presentation_artifact"]
        self.assertFalse(cue["support_is_sufficient"])
        self.assertEqual(cue["presentation_artifact_score"], 0.0)
        self.assertEqual(result.raw_score, 0.0)

    def test_broadband_display_texture_uses_three_independent_families(self):
        results = self._results()
        results["fft"] = self._result(
            "fft",
            0.0,
            features={
                "middle_frequency_energy_ratio": 0.16,
                "spectral_entropy": 0.686,
                "high_to_low_energy_ratio": 0.075,
            },
        )
        results["dct_block"] = self._result(
            "dct_block",
            3.0,
            features={
                "middle_frequency_ac_energy_ratio": 0.15,
                "high_frequency_ac_energy_ratio": 0.035,
                "near_zero_ac_coefficient_ratio": 0.40,
            },
        )
        results["wavelet"] = self._result(
            "wavelet",
            6.0,
            features={
                "clipped_pixel_ratio": 0.0,
                "global_detail_sparsity_mean": 0.23,
            },
        )
        results["residual"] = self._result(
            "residual",
            15.0,
            features={
                "clipped_pixel_ratio": 0.0,
                "gaussian_residual_rms_energy": 9.2,
                "laplacian_variance": 8200.0,
                "high_frequency_edge_density": 0.72,
            },
        )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(results, self._context())

        cue = result.raw_features["presentation_artifact"]
        self.assertTrue(cue["broadband_support_is_sufficient"])
        self.assertEqual(cue["presentation_mode"], "broadband_display_texture")
        self.assertGreater(cue["presentation_artifact_score"], 75.0)
        self.assertEqual(result.status, "High mathematical risk")
        self.assertIn("Broadband display texture", " ".join(result.evidence))

    def test_fft_broadband_texture_needs_transform_corroboration(self):
        results = self._results()
        results["fft"] = self._result(
            "fft",
            0.0,
            features={
                "middle_frequency_energy_ratio": 0.17,
                "spectral_entropy": 0.70,
                "high_to_low_energy_ratio": 0.08,
            },
        )

        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        cue = result.raw_features["presentation_artifact"]
        self.assertFalse(cue["broadband_support_is_sufficient"])
        self.assertEqual(cue["presentation_artifact_score"], 0.0)

    def test_partial_broadband_evidence_is_weak_instead_of_zero(self):
        results = self._results()
        results["fft"] = self._result(
            "fft",
            0.0,
            features={
                "middle_frequency_energy_ratio": 0.173,
                "spectral_entropy": 0.679,
                "high_to_low_energy_ratio": 0.045,
            },
        )
        results["dct_block"] = self._result(
            "dct_block",
            3.0,
            features={
                "middle_frequency_ac_energy_ratio": 0.091,
                "high_frequency_ac_energy_ratio": 0.014,
                "near_zero_ac_coefficient_ratio": 0.457,
            },
        )
        results["wavelet"] = self._result(
            "wavelet",
            6.0,
            features={"global_detail_sparsity_mean": 0.334},
        )
        results["residual"] = self._result(
            "residual",
            15.0,
            features={
                "gaussian_residual_rms_energy": 6.615,
                "laplacian_variance": 3888.0,
                "high_frequency_edge_density": 0.519,
            },
        )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(results, self._context())

        cue = result.raw_features["presentation_artifact"]
        self.assertTrue(cue["partial_support_is_sufficient"])
        self.assertEqual(cue["presentation_mode"], "partial_broadband_texture")
        self.assertGreater(cue["presentation_artifact_score"], 25.0)
        self.assertLess(cue["presentation_artifact_score"], 50.0)
        self.assertEqual(result.status, "Weak anomaly evidence")
        self.assertFalse(analyzer.warning_is_active)

    def test_visible_screen_border_does_not_affect_fusion(self):
        frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (80, 35),
            (560, 330),
            (230, 230, 230),
            5,
        )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(
                self._results(),
                self._context_with_frame(frame),
            )

        cue = result.raw_features["presentation_artifact"]
        self.assertEqual(
            cue["frame_structure"],
            {
                "enabled": False,
                "available": False,
                "included_in_fusion": False,
                "score": None,
                "exclusion_reason": (
                    "Screen-border evidence disabled by configuration"
                ),
            },
        )
        self.assertEqual(cue["presentation_artifact_score"], 0.0)
        self.assertEqual(result.raw_score, 0.0)
        self.assertEqual(
            result.raw_features["score_summary"]["current_frame_score"],
            0.0,
        )
        self.assertFalse(analyzer.warning_is_active)
        self.assertEqual(result.attack_type, "none")
        self.assertNotIn("border", " ".join(result.evidence).lower())

    def test_parallel_background_lines_do_not_affect_fusion(self):
        frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        for y_coordinate in range(15, 350, 20):
            cv2.line(
                frame,
                (0, y_coordinate),
                (639, y_coordinate),
                (230, 230, 230),
                2,
            )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(
                self._results(),
                self._context_with_frame(frame),
            )

        cue = result.raw_features["presentation_artifact"]
        self.assertEqual(cue["presentation_artifact_score"], 0.0)
        self.assertFalse(cue["frame_structure"]["included_in_fusion"])
        self.assertEqual(result.raw_score, 0.0)
        self.assertFalse(analyzer.warning_is_active)
        self.assertEqual(result.attack_type, "none")

    def test_face_region_mathematical_replay_evidence_needs_no_border(self):
        results = self._results()
        results["fft"] = self._result(
            "fft",
            80.0,
            features={
                "middle_frequency_energy_ratio": 0.18,
                "spectral_entropy": 0.70,
                "high_to_low_energy_ratio": 0.086,
            },
        )
        results["dct_block"] = self._result(
            "dct_block",
            75.0,
            features={
                "middle_frequency_ac_energy_ratio": 0.15,
                "high_frequency_ac_energy_ratio": 0.035,
                "near_zero_ac_coefficient_ratio": 0.40,
            },
        )
        results["wavelet"] = self._result(
            "wavelet",
            78.0,
            features={"global_detail_sparsity_mean": 0.23},
        )
        results["residual"] = self._result(
            "residual",
            82.0,
            features={
                "gaussian_residual_rms_energy": 9.2,
                "laplacian_variance": 8200.0,
                "high_frequency_edge_density": 0.72,
            },
        )
        results["moire"] = self._result(
            "moire",
            90.0,
            features={
                "periodic_peak_score": 0.9,
                "symmetric_peak_score": 0.9,
                "directional_concentration_score": 0.8,
            },
        )
        results["radial_angular"] = self._result(
            "radial_angular",
            80.0,
            status="Directional concentration detected",
            debug_data={"raw_angular_score": 80.0},
        )

        analyzer = MathematicalFusionController()
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            result = analyzer.analyze(results, self._context())

        self.assertNotIn("analysis_frame", vars(self._context()))
        self.assertTrue(
            result.raw_features["presentation_artifact"][
                "broadband_support_is_sufficient"
            ]
        )
        self.assertFalse(
            result.raw_features["presentation_artifact"][
                "frame_structure"
            ]["included_in_fusion"]
        )
        self.assertTrue(analyzer.warning_is_active)
        self.assertEqual(result.attack_type, "presentation_attack")
        self.assertIn(
            result.status,
            ("Suspicious mathematical evidence", "High mathematical risk"),
        )

    def test_intermittent_presentation_hits_survive_short_focus_drops(self):
        high_results = self._results()
        high_results["fft"] = self._result(
            "fft",
            0.0,
            features={
                "middle_frequency_energy_ratio": 0.18,
                "spectral_entropy": 0.70,
                "high_to_low_energy_ratio": 0.086,
            },
        )
        high_results["dct_block"] = self._result(
            "dct_block",
            3.0,
            features={
                "middle_frequency_ac_energy_ratio": 0.132,
                "high_frequency_ac_energy_ratio": 0.029,
                "near_zero_ac_coefficient_ratio": 0.435,
            },
        )
        high_results["wavelet"] = self._result(
            "wavelet",
            6.0,
            features={"global_detail_sparsity_mean": 0.291},
        )
        high_results["residual"] = self._result(
            "residual",
            15.0,
            features={
                "gaussian_residual_rms_energy": 8.485,
                "laplacian_variance": 6970.0,
                "high_frequency_edge_density": 0.621,
            },
        )
        low_results = self._results()
        analyzer = MathematicalFusionController()

        for results in (
            high_results,
            low_results,
            high_results,
            low_results,
            high_results,
        ):
            result = analyzer.analyze(results, self._context())

        self.assertTrue(analyzer.warning_is_active)
        self.assertTrue(analyzer.presentation_warning_is_active)
        self.assertEqual(result.attack_type, "presentation_attack")
        self.assertEqual(
            result.raw_features["presentation_recent_suspicious_hit_count"],
            3,
        )

        recovery_count = (
            config.EXPERIMENTAL_PRESENTATION_REQUIRED_RECOVERY_FRAMES
        )
        for _ in range(recovery_count - 1):
            result = analyzer.analyze(low_results, self._context())
        self.assertTrue(analyzer.warning_is_active)
        self.assertIn(
            result.status,
            ("Suspicious mathematical evidence", "High mathematical risk"),
        )

        result = analyzer.analyze(low_results, self._context())
        self.assertFalse(analyzer.warning_is_active)
        self.assertFalse(analyzer.presentation_warning_is_active)

    def test_too_few_valid_modules_is_inconclusive_and_not_history(self):
        results = self._results()
        for name in ("radial_angular", "dct_block", "wavelet", "residual"):
            results[name] = self._result(name, None, available=False)
        analyzer = MathematicalFusionController()
        result = analyzer.analyze(results, self._context())

        self.assertFalse(result.available)
        self.assertEqual(result.status, "Inconclusive")
        self.assertIsNone(result.raw_score)
        self.assertEqual(len(analyzer.score_history), 0)

    def test_temporal_warning_requires_streak_and_recovers_with_hysteresis(self):
        analyzer = MathematicalFusionController()
        high_results = self._results(
            {name: 80.0 for name in self.MODULE_NAMES}
        )
        for _ in range(config.MATHEMATICAL_FUSION_CONFIG["minimum_history"]):
            high_result = analyzer.analyze(high_results, self._context())

        self.assertEqual(high_result.status, "High mathematical risk")
        self.assertTrue(analyzer.warning_is_active)
        self.assertGreaterEqual(
            analyzer.consecutive_suspicious_frames,
            config.MATHEMATICAL_FUSION_CONFIG[
                "required_suspicious_frames"
            ],
        )
        self.assertIn(
            "possible replay or print attack",
            high_result.warning,
        )

        low_results = self._results()
        for _ in range(
            config.MATHEMATICAL_FUSION_CONFIG["required_recovery_frames"]
        ):
            recovery_result = analyzer.analyze(low_results, self._context())

        self.assertFalse(analyzer.warning_is_active)
        self.assertEqual(
            analyzer.consecutive_recovery_frames,
            config.MATHEMATICAL_FUSION_CONFIG[
                "required_recovery_frames"
            ],
        )
        self.assertNotIn(
            recovery_result.status,
            ("Suspicious mathematical evidence", "High mathematical risk"),
        )

    def test_invalid_quality_frame_does_not_enter_history(self):
        analyzer = MathematicalFusionController()
        analyzer.analyze(self._results(), self._context())
        history_length = len(analyzer.score_history)
        result = analyzer.analyze(self._results(), self._context(valid=False))

        self.assertEqual(result.status, "Inconclusive")
        self.assertEqual(len(analyzer.score_history), history_length)
        self.assertFalse(
            result.debug_data["valid_quality_frame_added_to_history"]
        )

    def test_evidence_is_generated_only_from_measured_features(self):
        results = self._results({name: 70.0 for name in self.MODULE_NAMES})
        results["moire"] = self._result(
            "moire",
            70.0,
            features={
                "periodic_peak_score": 0.9,
                "symmetric_peak_score": 0.9,
                "directional_concentration_score": 0.9,
            },
        )
        results["radial_angular"] = self._result(
            "radial_angular",
            70.0,
            status="Directional concentration detected",
            debug_data={"raw_angular_score": 75.0},
        )
        results["dct_block"] = self._result(
            "dct_block",
            70.0,
            features={
                "blockiness_score": 70.0,
                "local_dct_inconsistency_score": 0.0,
            },
        )
        results["wavelet"] = self._result(
            "wavelet",
            70.0,
            features={
                "local_wavelet_inconsistency_score": 70.0,
                "directional_wavelet_score": 0.0,
            },
        )
        results["residual"] = self._result(
            "residual",
            70.0,
            features={
                "local_residual_inconsistency_score": 0.0,
                "gaussian_residual_score": 70.0,
                "laplacian_score": 0.0,
                "gradient_score": 0.0,
                "residual_energy_direction_label": "high",
            },
        )

        result = MathematicalFusionController().analyze(
            results,
            self._context(),
        )

        for evidence in (
            "Strong symmetric periodic FFT peaks",
            "Directional frequency concentration",
            "8x8 block discontinuity",
            "Localized wavelet detail inconsistency",
            "Abnormal high-pass residual energy: excessive",
        ):
            self.assertIn(evidence, result.evidence)
        self.assertNotIn("Local DCT coefficient inconsistency", result.evidence)

    def test_missing_calibration_is_explicitly_uncalibrated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "model_free_calibration.json"
            with mock.patch.object(
                config,
                "MODEL_FREE_CALIBRATION_FILE_PATH",
                missing,
            ):
                result = MathematicalFusionController().analyze(
                    self._results(),
                    self._context(),
                )

        self.assertFalse(result.calibrated)
        self.assertEqual(result.status, "Uncalibrated")
        self.assertEqual(result.debug_data["scoring_mode"], "experimental")

    def test_compatible_calibration_maps_score_and_overrides_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = (
                Path(temporary_directory) / "model_free_calibration.json"
            )
            calibration_path.write_text(
                json.dumps(
                    {
                        "mathematical_fusion": {
                            "bona_fide_baseline": {
                                "combined_mathematical_risk_score": {
                                    "mean": 10.0,
                                    "standard_deviation": 5.0,
                                }
                            },
                            "score_mapping": {
                                "z_score_start": 1.0,
                                "z_score_full": 5.0,
                            },
                            "module_weights": {
                                name: 1.0 for name in self.MODULE_NAMES
                            },
                            "group_weights": {
                                "fft_family": 1.0,
                                "local_transform": 1.0,
                            },
                            "status_thresholds": {
                                "weak_anomaly_score": 20.0,
                                "suspicious_score": 50.0,
                                "high_risk_score": 80.0,
                                "recovery_score": 40.0,
                            },
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
                result = MathematicalFusionController().analyze(
                    self._results(
                        {name: 20.0 for name in self.MODULE_NAMES}
                    ),
                    self._context(),
                )

        self.assertTrue(result.calibrated)
        self.assertAlmostEqual(
            result.raw_features["pre_mapping_combined_score"],
            20.0,
        )
        self.assertAlmostEqual(result.raw_score, 25.0)
        self.assertEqual(result.debug_data["scoring_mode"], "calibrated")

    def test_reporting_uses_each_canonical_score_field(self):
        score_summary = {
            "current_frame_score": 44.31,
            "rolling_median": 56.57,
            "temporal_percentile": 62.63,
            "temporal_decision_score": 62.63,
            "display_score": 62.63,
        }
        report = {
            "export_timestamp": "test",
            "quality": {
                "blur": 100.0,
                "brightness": 120.0,
                "valid": True,
            },
            "modules": {},
            "group_scores": {
                "fft_family_score": 40.0,
                "fft_family_confidence": 1.0,
                "local_transform_score": 48.0,
                "local_transform_confidence": 1.0,
            },
            "score_summary": score_summary,
            "combined_score": {
                **score_summary,
                "calibrated": False,
                "status": "Suspicious mathematical evidence",
            },
            "fusion": {
                "raw_features": {
                    "consecutive_suspicious_frame_count": 3,
                    "consecutive_recovery_frame_count": 0,
                }
            },
            "final_evidence": [],
            "final_warnings": [],
        }
        application = ModelFreePreControlApplication()
        text_report = application._human_readable_report(report)
        application.shutdown()

        expected_lines = (
            "Current frame risk: 44.31",
            "Rolling median: 56.57",
            "Temporal percentile: 62.63",
            "Final temporal decision: 62.63",
            "Displayed risk: 62.63",
        )
        for expected_line in expected_lines:
            self.assertIn(expected_line, text_report)
        self.assertNotIn("Rolling median: 62.63", text_report)
        self.assertEqual(application._format_report_score(None), "N/A")

        gui = FaceQualityGui.__new__(FaceQualityGui)
        self.assertEqual(
            gui._fusion_score_summary_lines(score_summary),
            list(expected_lines),
        )
        self.assertEqual(gui._format_fusion_score(None), "N/A")

    def test_incompatible_temporal_schema_clears_old_history(self):
        analyzer = MathematicalFusionController()
        analyzer.score_history.extend([78.0] * 5)
        analyzer.presentation_score_history.extend([78.0] * 5)
        analyzer.warning_is_active = True
        analyzer.presentation_warning_is_active = True
        analyzer.temporal_schema_signature = (1, 1, True)

        result = analyzer.analyze(self._results(), self._context())

        self.assertEqual(len(analyzer.score_history), 1)
        self.assertEqual(list(analyzer.score_history), [0.0])
        self.assertEqual(list(analyzer.presentation_score_history), [0.0])
        self.assertFalse(analyzer.warning_is_active)
        self.assertFalse(analyzer.presentation_warning_is_active)
        self.assertEqual(
            result.raw_features["score_summary"],
            {
                "current_frame_score": 0.0,
                "rolling_median": 0.0,
                "temporal_percentile": 0.0,
                "temporal_decision_score": 0.0,
                "display_score": 0.0,
            },
        )

        application = ModelFreePreControlApplication()
        self.assertEqual(len(application.fusion_controller.score_history), 0)
        self.assertEqual(
            len(application.fusion_controller.presentation_score_history),
            0,
        )
        application.shutdown()

    def test_unified_debug_export_contains_all_modules_json_and_text(self):
        random = np.random.default_rng(51)
        gray = np.clip(
            125 + random.normal(0, 18, (480, 640)),
            0,
            255,
        ).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            debug_root = Path(temporary_directory)
            with mock.patch.object(
                config,
                "MODEL_FREE_DEBUG_DIRECTORY",
                debug_root,
            ):
                application = ModelFreePreControlApplication()
                for _ in range(5):
                    application.process_frame(frame)
                with redirect_stdout(io.StringIO()):
                    saved = application.save_debug_sample()
                application.shutdown()

            folders = [path for path in debug_root.iterdir() if path.is_dir()]
            self.assertTrue(saved)
            self.assertEqual(len(folders), 1)
            output = folders[0]
            json_path = output / "model_free_analysis_report.json"
            text_path = output / "model_free_analysis_report.txt"
            self.assertTrue(json_path.exists())
            self.assertTrue(text_path.exists())
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["schema_version"],
                config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION,
            )
            self.assertEqual(
                set(report["score_summary"]),
                {
                    "current_frame_score",
                    "rolling_median",
                    "temporal_percentile",
                    "temporal_decision_score",
                    "display_score",
                },
            )
            self.assertEqual(set(report["modules"]), set(self.MODULE_NAMES))
            for module in report["modules"].values():
                self.assertIn("raw_features", module)
                self.assertIn("raw_score", module)
                self.assertIn("stabilized_score", module)
            self.assertIn("fft_family_score", report["group_scores"])
            self.assertIn("local_transform_score", report["group_scores"])
            self.assertIn("final_evidence", report)
            frame_structure = report["fusion"]["raw_features"][
                "presentation_artifact"
            ]["frame_structure"]
            self.assertEqual(
                frame_structure,
                {
                    "enabled": False,
                    "available": False,
                    "included_in_fusion": False,
                    "score": None,
                    "exclusion_reason": (
                        "Screen-border evidence disabled by configuration"
                    ),
                },
            )
            self.assertFalse(
                any(
                    name.startswith("EXPERIMENTAL_PRESENTATION_FRAME_")
                    for name in report["configuration_thresholds"]
                )
            )
            module_visualizations = {
                item["module"] for item in report["visualizations"]
            }
            self.assertTrue(set(self.MODULE_NAMES) <= module_visualizations)
            text_report = text_path.read_text(encoding="utf-8")
            self.assertIn(
                "MODEL-FREE PRECONTROL MATHEMATICAL REPORT",
                text_report,
            )
            for label in (
                "Current frame risk:",
                "Rolling median:",
                "Temporal percentile:",
                "Final temporal decision:",
                "Displayed risk:",
            ):
                self.assertIn(label, text_report)


if __name__ == "__main__":
    unittest.main()
