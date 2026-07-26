"""Two-stage, model-free fusion of the six mathematical PreControl modules."""

from collections import deque
import copy
import json
import math

import numpy as np

import config
from model_free_analysis import FusionScoreSummary, ModelFreeAnalysisResult


class MathematicalFusionController:
    """Fuse correlated module scores without treating missing data as zero."""

    MODULE_NAME = "Combined Mathematical Risk"

    def __init__(self):
        self.experimental_config = copy.deepcopy(
            config.MATHEMATICAL_FUSION_CONFIG
        )
        self.calibration = self._load_calibration()
        self.active_config = self._active_configuration()
        self.score_history = deque(
            maxlen=int(self.active_config["history_size"])
        )
        self.presentation_score_history = deque(
            maxlen=config.EXPERIMENTAL_PRESENTATION_TEMPORAL_HISTORY_SIZE
        )
        self.consecutive_suspicious_frames = 0
        self.consecutive_recovery_frames = 0
        self.warning_is_active = False
        self.presentation_warning_is_active = False
        self.invalid_streak = 0
        self.previous_region = None
        self.temporal_schema_signature = (
            self._current_temporal_schema_signature()
        )

    def analyze(self, module_results, context):
        """Return one final result while retaining both family calculations."""
        self._reset_if_temporal_schema_changed()
        self._handle_region_change(
            context.face_bounding_box if context is not None else None
        )
        if context is None:
            return self._inconclusive(
                "Shared model-free analysis context is unavailable"
            )
        if not context.face_quality_valid:
            return self._inconclusive(
                context.quality_reason or "Shared face quality gate failed"
            )

        observations = self._module_observations(module_results)
        group_results = {}
        for group_name, member_names in self.active_config[
            "module_groups"
        ].items():
            group_results[group_name] = self._fuse_group(
                group_name,
                member_names,
                observations,
            )

        valid_modules = [
            name
            for name, observation in observations.items()
            if observation["included"]
        ]
        coverage_reasons = self._coverage_reasons(
            valid_modules,
            group_results,
        )
        raw_features = self._base_raw_features(
            observations,
            group_results,
            valid_modules,
        )
        if coverage_reasons:
            return self._inconclusive(
                "; ".join(coverage_reasons),
                raw_features=raw_features,
                observations=observations,
            )

        pre_mapping_score, group_contributions = self._fuse_groups(
            group_results
        )
        if pre_mapping_score is None:
            return self._inconclusive(
                "The available group confidence is insufficient for fusion",
                raw_features=raw_features,
                observations=observations,
            )

        presentation_features = self._presentation_artifact_features(
            observations
        )
        # Presentation support is derived only from the six face-region
        # modules and is kept in its own temporal channel. It must not replace
        # the weighted two-family current-frame score.
        base_group_score = pre_mapping_score

        combined_score = self._map_calibrated_score(pre_mapping_score)
        raw_features.update(
            {
                "base_group_combined_score": base_group_score,
                "presentation_artifact": presentation_features,
                "presentation_artifact_score": presentation_features[
                    "presentation_artifact_score"
                ],
                "pre_mapping_combined_score": pre_mapping_score,
                "combined_mathematical_risk_score": combined_score,
                "group_contributions": group_contributions,
            }
        )
        return self._stabilize(
            combined_score,
            raw_features,
            observations,
            group_results,
            context,
        )

    def reset(self):
        self._reset_temporal_state()
        self.previous_region = None
        self.temporal_schema_signature = (
            self._current_temporal_schema_signature()
        )

    def get_configuration_snapshot(self):
        snapshot = copy.deepcopy(self.active_config)
        snapshot.update(
            {
                "analysis_schema_version": (
                    config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION
                ),
                "fusion_configuration_version": (
                    config.MODEL_FREE_FUSION_CONFIGURATION_VERSION
                ),
                "frame_structure_enabled": False,
            }
        )
        return snapshot

    def get_calibration_summary(self):
        return {
            "path": str(config.MODEL_FREE_CALIBRATION_FILE_PATH),
            "file_exists": config.MODEL_FREE_CALIBRATION_FILE_PATH.exists(),
            "fusion_calibration_compatible": self.calibration is not None,
            "scoring_mode": (
                "calibrated" if self.calibration else "experimental"
            ),
        }

    def _module_observations(self, module_results):
        expected_names = tuple(
            name
            for members in self.active_config["module_groups"].values()
            for name in members
        )
        observations = {}
        for name in expected_names:
            result = module_results.get(name)
            observation = {
                "included": False,
                "exclusion_reason": None,
                "score": None,
                "raw_score": None,
                "stabilized_score": None,
                "confidence": None,
                "weight": float(
                    self.active_config["module_weights"].get(name, 0.0)
                ),
                "effective_weight": 0.0,
                "status": None,
                "calibrated": False,
                "quality_status": None,
                "result": result,
            }
            observations[name] = observation
            if result is None:
                observation["exclusion_reason"] = "result missing"
                continue

            observation["status"] = result.status
            observation["calibrated"] = bool(result.calibrated)
            observation["raw_score"] = self._finite_number(
                result.raw_score
            )
            observation["stabilized_score"] = self._finite_number(
                result.stabilized_score
            )
            observation["quality_status"] = str(
                result.debug_data.get("quality_status", "Unknown")
            )
            if not result.available:
                observation["exclusion_reason"] = "module unavailable"
                continue
            if observation["raw_score"] is None:
                observation["exclusion_reason"] = "raw score unavailable"
                continue
            if not self._quality_status_is_valid(
                observation["quality_status"]
            ):
                observation["exclusion_reason"] = (
                    "module quality is uncertain or invalid"
                )
                continue

            confidence = self._finite_number(result.confidence)
            if confidence is None:
                observation["exclusion_reason"] = "confidence unavailable"
                continue
            confidence = float(np.clip(confidence, 0.0, 1.0))
            observation["confidence"] = confidence
            minimum_confidence = float(
                self.active_config["minimum_effective_confidence"]
            )
            if confidence < minimum_confidence:
                observation["exclusion_reason"] = (
                    "confidence below fusion minimum"
                )
                continue
            if observation["weight"] <= 0.0:
                observation["exclusion_reason"] = (
                    "module fusion weight is not positive"
                )
                continue

            observation["score"] = float(
                np.clip(observation["raw_score"], 0.0, 100.0)
            )
            observation["effective_weight"] = (
                observation["weight"] * confidence
            )
            observation["included"] = True
        return observations

    def _quality_status_is_valid(self, quality_status):
        normalized = quality_status.strip().lower()
        if normalized in ("uncertain", "unknown", ""):
            return False
        invalid_terms = (
            "unavailable",
            "invalid",
            "failed",
            "too low",
            "too small",
            "empty",
            "missing",
        )
        return not any(term in normalized for term in invalid_terms)

    def _fuse_group(self, group_name, member_names, observations):
        included = [
            name for name in member_names if observations[name]["included"]
        ]
        denominator = sum(
            observations[name]["effective_weight"] for name in included
        )
        if denominator <= 0.0:
            score = None
        else:
            score = sum(
                observations[name]["score"]
                * observations[name]["effective_weight"]
                for name in included
            ) / denominator

        base_weight_sum = sum(
            observations[name]["weight"] for name in included
        )
        group_confidence = (
            denominator / base_weight_sum if base_weight_sum > 0.0 else 0.0
        )
        return {
            "name": group_name,
            "score": None if score is None else float(score),
            "confidence": float(np.clip(group_confidence, 0.0, 1.0)),
            "included_modules": included,
            "valid_module_count": len(included),
            "effective_weight_sum": float(denominator),
        }

    def _coverage_reasons(self, valid_modules, group_results):
        reasons = []
        minimum_total = int(self.active_config["minimum_valid_modules"])
        if len(valid_modules) < minimum_total:
            reasons.append(
                "only %d/%d required modules are valid"
                % (len(valid_modules), minimum_total)
            )
        group_minimums = self.active_config[
            "minimum_valid_modules_per_group"
        ]
        for group_name, minimum in group_minimums.items():
            count = group_results[group_name]["valid_module_count"]
            if count < int(minimum):
                reasons.append(
                    "%s has only %d/%d required valid modules"
                    % (group_name, count, int(minimum))
                )
        return reasons

    def _fuse_groups(self, group_results):
        numerator = 0.0
        denominator = 0.0
        contributions = {}
        for group_name, group in group_results.items():
            score = group["score"]
            group_weight = float(
                self.active_config["group_weights"].get(group_name, 0.0)
            )
            confidence = float(group["confidence"])
            effective_weight = group_weight * confidence
            included = score is not None and effective_weight > 0.0
            contributions[group_name] = {
                "score": score,
                "confidence": confidence,
                "weight": group_weight,
                "effective_weight": effective_weight,
                "included": included,
            }
            if not included:
                continue
            numerator += float(score) * effective_weight
            denominator += effective_weight
        if denominator <= 0.0:
            return None, contributions
        return float(np.clip(numerator / denominator, 0.0, 100.0)), contributions

    def _map_calibrated_score(self, score):
        if self.calibration is None:
            return float(score)
        baseline = self.calibration["combined_baseline"]
        mapping = self.calibration["score_mapping"]
        standard_deviation = baseline["standard_deviation"]
        start = baseline["mean"] + mapping["z_score_start"] * standard_deviation
        full = baseline["mean"] + mapping["z_score_full"] * standard_deviation
        if full <= start:
            return float(score)
        mapped = 100.0 * (float(score) - start) / (full - start)
        return float(np.clip(mapped, 0.0, 100.0))

    def _stabilize(
        self,
        raw_score,
        raw_features,
        observations,
        group_results,
        context,
    ):
        self.invalid_streak = 0
        self.score_history.append(raw_score)
        rolling_median = float(np.median(self.score_history))
        presentation_score = self._finite_number(
            raw_features.get("presentation_artifact_score")
        )
        if presentation_score is None:
            presentation_score = 0.0
        self.presentation_score_history.append(presentation_score)

        suspicious_threshold = float(self.active_config["suspicious_score"])
        recovery_threshold = float(self.active_config["recovery_score"])
        weak_threshold = float(self.active_config["weak_anomaly_score"])

        presentation_history = list(self.presentation_score_history)
        activation_window = int(
            config.EXPERIMENTAL_PRESENTATION_ACTIVATION_WINDOW
        )
        recent_presentation = presentation_history[-activation_window:]
        presentation_percentile = float(
            np.percentile(
                recent_presentation,
                config.EXPERIMENTAL_PRESENTATION_TEMPORAL_PERCENTILE,
            )
        )
        presentation_suspicious_hits = sum(
            score >= suspicious_threshold for score in recent_presentation
        )
        presentation_weak_hits = sum(
            score >= weak_threshold for score in recent_presentation
        )

        if raw_score >= suspicious_threshold:
            self.consecutive_suspicious_frames += 1
            self.consecutive_recovery_frames = 0
        elif (
            raw_score <= recovery_threshold
            and presentation_score < weak_threshold
        ):
            self.consecutive_recovery_frames += 1
            self.consecutive_suspicious_frames = 0
        else:
            self.consecutive_suspicious_frames = 0
            self.consecutive_recovery_frames = 0

        history_ready = (
            len(self.score_history) >= int(self.active_config["minimum_history"])
        )
        generic_activation = (
            history_ready
            and self.consecutive_suspicious_frames
            >= int(self.active_config["required_suspicious_frames"])
            and rolling_median >= suspicious_threshold
        )
        presentation_activation = (
            history_ready
            and presentation_suspicious_hits
            >= config.EXPERIMENTAL_PRESENTATION_REQUIRED_SUSPICIOUS_HITS
            and presentation_percentile >= suspicious_threshold
        )
        if not self.warning_is_active:
            if presentation_activation:
                self.warning_is_active = True
                self.presentation_warning_is_active = True
            elif generic_activation:
                self.warning_is_active = True
                self.presentation_warning_is_active = False
        elif presentation_activation:
            self.presentation_warning_is_active = True

        recovery_count = int(self.active_config["required_recovery_frames"])
        if self.presentation_warning_is_active:
            recovery_count = max(
                recovery_count,
                config.EXPERIMENTAL_PRESENTATION_REQUIRED_RECOVERY_FRAMES,
            )
        if (
            self.warning_is_active
            and self.consecutive_recovery_frames >= recovery_count
        ):
            recent_median = float(
                np.median(list(self.score_history)[-recovery_count:])
            )
            if recent_median <= recovery_threshold:
                self.warning_is_active = False
                self.presentation_warning_is_active = False

        stable_score = rolling_median
        presentation_used_for_decision = (
            self.presentation_warning_is_active
            or presentation_weak_hits
            >= config.EXPERIMENTAL_PRESENTATION_REQUIRED_WEAK_HITS
        )
        if presentation_used_for_decision:
            stable_score = max(stable_score, presentation_percentile)

        if not history_ready:
            status = "Inconclusive" if self.calibration else "Uncalibrated"
        elif self.warning_is_active:
            if stable_score >= float(self.active_config["high_risk_score"]):
                status = "High mathematical risk"
            else:
                status = "Suspicious mathematical evidence"
        elif stable_score >= weak_threshold:
            status = "Weak anomaly evidence"
        else:
            status = "Normal mathematical evidence"

        evidence = self._create_evidence(
            observations,
            raw_features.get("presentation_artifact"),
        )
        warnings = []
        if status in (
            "Suspicious mathematical evidence",
            "High mathematical risk",
        ):
            warnings.append(
                "WARNING: Mathematical image evidence is consistent with "
                "a possible replay or print attack."
            )

        combined_confidence = self._combined_confidence(group_results)
        if not self.calibration:
            combined_confidence = min(
                combined_confidence,
                float(self.active_config["uncalibrated_confidence_cap"]),
            )
        score_summary = FusionScoreSummary(
            current_frame_score=self._finite_number(
                raw_features.get("combined_mathematical_risk_score")
            ),
            rolling_median=rolling_median,
            temporal_percentile=presentation_percentile,
            temporal_decision_score=stable_score,
            display_score=stable_score,
        ).to_dict()
        raw_features.update(
            {
                "rolling_median": rolling_median,
                "temporal_decision_score": stable_score,
                "presentation_temporal_percentile": (
                    presentation_percentile
                ),
                "presentation_recent_suspicious_hit_count": (
                    presentation_suspicious_hits
                ),
                "presentation_recent_weak_hit_count": presentation_weak_hits,
                "presentation_temporal_used_for_decision": (
                    presentation_used_for_decision
                ),
                "presentation_warning_active": (
                    self.presentation_warning_is_active
                ),
                "history_length": len(self.score_history),
                "consecutive_suspicious_frame_count": (
                    self.consecutive_suspicious_frames
                ),
                "consecutive_recovery_frame_count": (
                    self.consecutive_recovery_frames
                ),
                "warning_active": self.warning_is_active,
                "score_summary": score_summary,
                "analysis_schema_version": (
                    config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION
                ),
                "fusion_configuration_version": (
                    config.MODEL_FREE_FUSION_CONFIGURATION_VERSION
                ),
            }
        )
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=True,
            raw_features=raw_features,
            raw_score=raw_score,
            stabilized_score=stable_score,
            confidence=combined_confidence,
            status=status,
            evidence=evidence,
            warnings=warnings,
            debug_data={
                "possible_attack": (
                    "presentation_attack"
                    if self.warning_is_active
                    and (
                        self.presentation_warning_is_active
                        or presentation_score > 0.0
                    )
                    else "none"
                ),
                "quality_status": "Sufficient",
                "scoring_mode": (
                    "calibrated" if self.calibration else "experimental"
                ),
                "calibration": self.get_calibration_summary(),
                "two_stage_fusion": True,
                "shared_fft_family_counted_as_one_final_group": True,
                "valid_quality_frame_added_to_history": True,
                "quality": self._context_quality(context),
            },
            calibrated=self.calibration is not None,
        )

    def _combined_confidence(self, group_results):
        numerator = 0.0
        denominator = 0.0
        for group_name, group in group_results.items():
            if group["score"] is None:
                continue
            weight = float(self.active_config["group_weights"][group_name])
            numerator += float(group["confidence"]) * weight
            denominator += weight
        return float(
            np.clip(numerator / denominator, 0.0, 1.0)
            if denominator > 0.0
            else 0.0
        )

    def _base_raw_features(
        self,
        observations,
        group_results,
        valid_modules,
    ):
        serializable_observations = {}
        for name, observation in observations.items():
            serializable_observations[name] = {
                key: value
                for key, value in observation.items()
                if key != "result"
            }
        return {
            "fft_family_score": group_results["fft_family"]["score"],
            "fft_family_confidence": group_results[
                "fft_family"
            ]["confidence"],
            "local_transform_score": group_results[
                "local_transform"
            ]["score"],
            "local_transform_confidence": group_results[
                "local_transform"
            ]["confidence"],
            "valid_module_count": len(valid_modules),
            "valid_modules": list(valid_modules),
            "module_contributions": serializable_observations,
            "group_details": copy.deepcopy(group_results),
        }

    def _inconclusive(
        self,
        reason,
        raw_features=None,
        observations=None,
    ):
        self.invalid_streak += 1
        if self.invalid_streak >= int(
            self.active_config["invalid_reset_frames"]
        ):
            self._reset_temporal_state()
        evidence = [reason]
        if observations:
            excluded = [
                "%s: %s" % (name, observation["exclusion_reason"])
                for name, observation in observations.items()
                if not observation["included"]
            ]
            evidence.extend(excluded)
        rolling_median = (
            float(np.median(self.score_history))
            if self.score_history
            else None
        )
        temporal_percentile = (
            float(
                np.percentile(
                    self.presentation_score_history,
                    config.EXPERIMENTAL_PRESENTATION_TEMPORAL_PERCENTILE,
                )
            )
            if self.presentation_score_history
            else None
        )
        features = dict(raw_features or {})
        features.update(
            {
                "score_summary": FusionScoreSummary(
                    current_frame_score=None,
                    rolling_median=rolling_median,
                    temporal_percentile=temporal_percentile,
                    temporal_decision_score=None,
                    display_score=rolling_median,
                ).to_dict(),
                "analysis_schema_version": (
                    config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION
                ),
                "fusion_configuration_version": (
                    config.MODEL_FREE_FUSION_CONFIGURATION_VERSION
                ),
            }
        )
        return ModelFreeAnalysisResult(
            module_name=self.MODULE_NAME,
            available=False,
            raw_features=features,
            raw_score=None,
            stabilized_score=rolling_median,
            confidence=0.0,
            status="Inconclusive",
            evidence=evidence,
            warnings=[],
            debug_data={
                "possible_attack": "none",
                "quality_status": reason,
                "scoring_mode": (
                    "calibrated" if self.calibration else "experimental"
                ),
                "calibration": self.get_calibration_summary(),
                "valid_quality_frame_added_to_history": False,
                "history_length": len(self.score_history),
                "consecutive_suspicious_frame_count": (
                    self.consecutive_suspicious_frames
                ),
                "consecutive_recovery_frame_count": (
                    self.consecutive_recovery_frames
                ),
            },
            calibrated=self.calibration is not None,
        )

    def _create_evidence(self, observations, presentation_features=None):
        evidence = []
        module_threshold = float(
            self.active_config["module_evidence_score"]
        )

        fft = observations["fft"]
        if fft["included"] and fft["score"] >= module_threshold:
            evidence.append("Global FFT spectral distribution anomaly")

        moire = observations["moire"]
        if moire["included"]:
            features = moire["result"].raw_features
            periodic = self._feature_number(features, "periodic_peak_score")
            symmetric = self._feature_number(
                features,
                "symmetric_peak_score",
            )
            directional = self._feature_number(
                features,
                "directional_concentration_score",
            )
            if (
                periodic
                >= config.EXPERIMENTAL_MOIRE_SUPPORTING_PERIODIC_SCORE
                and symmetric
                >= config.EXPERIMENTAL_MOIRE_SYMMETRY_EVIDENCE_SCORE
            ):
                evidence.append("Strong symmetric periodic FFT peaks")
            elif periodic >= config.EXPERIMENTAL_MOIRE_PERIODIC_EVIDENCE_SCORE:
                evidence.append("Strong periodic FFT peaks")
            if (
                periodic
                >= config.EXPERIMENTAL_MOIRE_SUPPORTING_PERIODIC_SCORE
                and directional
                >= config.EXPERIMENTAL_MOIRE_DIRECTION_EVIDENCE_SCORE
            ):
                evidence.append("Periodic FFT peaks have directional concentration")

        radial = observations["radial_angular"]
        if radial["included"]:
            radial_result = radial["result"]
            angular_score = self._feature_number(
                radial_result.debug_data,
                "raw_angular_score",
            )
            if (
                radial_result.status == "Directional concentration detected"
                or angular_score
                >= config.EXPERIMENTAL_RADIAL_DIRECTIONAL_STATUS_SCORE
            ):
                evidence.append("Directional frequency concentration")
            if radial_result.status == "Narrow-band spectral anomaly":
                evidence.append("Narrow-band radial frequency concentration")

        dct = observations["dct_block"]
        if dct["included"]:
            features = dct["result"].raw_features
            if self._feature_number(features, "blockiness_score") >= (
                config.EXPERIMENTAL_DCT_BLOCK_STRUCTURE_SCORE
            ):
                evidence.append("8x8 block discontinuity")
            if self._feature_number(
                features,
                "local_dct_inconsistency_score",
            ) >= config.EXPERIMENTAL_DCT_LOCAL_INCONSISTENCY_SCORE:
                evidence.append("Local DCT coefficient inconsistency")

        wavelet = observations["wavelet"]
        if wavelet["included"]:
            features = wavelet["result"].raw_features
            if self._feature_number(
                features,
                "local_wavelet_inconsistency_score",
            ) >= config.EXPERIMENTAL_WAVELET_LOCAL_STATUS_SCORE:
                evidence.append("Localized wavelet detail inconsistency")
            if self._feature_number(
                features,
                "directional_wavelet_score",
            ) >= config.EXPERIMENTAL_WAVELET_DIRECTIONAL_STATUS_SCORE:
                evidence.append("Directional wavelet detail anomaly")

        residual = observations["residual"]
        if residual["included"]:
            features = residual["result"].raw_features
            if self._feature_number(
                features,
                "local_residual_inconsistency_score",
            ) >= config.EXPERIMENTAL_RESIDUAL_LOCAL_STATUS_SCORE:
                evidence.append("Localized high-pass residual inconsistency")
            residual_components = (
                self._feature_number(features, "gaussian_residual_score"),
                self._feature_number(features, "laplacian_score"),
                self._feature_number(features, "gradient_score"),
            )
            if max(residual_components) >= (
                config.EXPERIMENTAL_RESIDUAL_ENERGY_STATUS_SCORE
            ):
                direction = str(
                    features.get("residual_energy_direction_label", "balanced")
                )
                if direction == "high":
                    evidence.append("Abnormal high-pass residual energy: excessive")
                elif direction == "low":
                    evidence.append("Abnormal high-pass residual energy: unusually low")
                else:
                    evidence.append("Abnormal high-pass residual energy")

        if presentation_features:
            evidence_threshold = float(
                self.active_config["module_evidence_score"]
            )
            if (
                presentation_features.get(
                    "clipping_support_is_sufficient"
                )
                and presentation_features.get(
                    "clipping_presentation_score",
                    0.0,
                )
                >= evidence_threshold
            ):
                evidence.append(
                    "Severe highlight clipping is supported by independent "
                    "DCT/FFT presentation artifacts"
                )
            if (
                presentation_features.get(
                    "broadband_support_is_sufficient"
                )
                and presentation_features.get(
                    "broadband_presentation_score",
                    0.0,
                )
                >= evidence_threshold
            ):
                evidence.append(
                    "Broadband display texture is repeated across FFT, DCT "
                    "and fine-detail transforms"
                )
        if not evidence:
            evidence.append("No included module crossed its evidence threshold")
        return list(dict.fromkeys(evidence))

    def _presentation_artifact_features(self, observations):
        """Preserve severe clipping as cross-method supporting evidence.

        Wavelet/residual correctly refuse to interpret texture after severe
        clipping. Their measured clipping ratio is still useful when an
        independent DCT or FFT observation supports it. The max clipping ratio
        is used so the same photometric cue is never counted twice.
        """
        clipping_ratios = {}
        for name in ("wavelet", "residual"):
            result = observations[name]["result"]
            ratio = None
            if result is not None:
                ratio = self._finite_number(
                    result.raw_features.get("clipped_pixel_ratio")
                )
            clipping_ratios[name] = ratio

        finite_clipping = [
            ratio for ratio in clipping_ratios.values() if ratio is not None
        ]
        clipping_ratio = max(finite_clipping, default=0.0)
        clipping_score = self._linear_unit_score(
            clipping_ratio,
            config.EXPERIMENTAL_PRESENTATION_CLIPPING_START,
            config.EXPERIMENTAL_PRESENTATION_CLIPPING_FULL,
        )

        dct_result = observations["dct_block"]["result"]
        dct_local_value = 0.0
        if dct_result is not None:
            dct_local_value = self._feature_number(
                dct_result.raw_features,
                "local_dct_inconsistency_score",
            )
        dct_support = self._linear_unit_score(
            dct_local_value,
            config.EXPERIMENTAL_PRESENTATION_DCT_LOCAL_START,
            config.EXPERIMENTAL_PRESENTATION_DCT_LOCAL_FULL,
        )

        fft_result = observations["fft"]["result"]
        fft_middle_value = 0.0
        fft_entropy_value = 0.0
        fft_high_to_low_value = 0.0
        if fft_result is not None:
            fft_middle_value = self._feature_number(
                fft_result.raw_features,
                "middle_frequency_energy_ratio",
            )
            fft_entropy_value = self._feature_number(
                fft_result.raw_features,
                "spectral_entropy",
            )
            fft_high_to_low_value = self._feature_number(
                fft_result.raw_features,
                "high_to_low_energy_ratio",
            )
        fft_middle_support = self._linear_unit_score(
            fft_middle_value,
            config.EXPERIMENTAL_PRESENTATION_FFT_MIDDLE_START,
            config.EXPERIMENTAL_PRESENTATION_FFT_MIDDLE_FULL,
        )
        fft_entropy_support = self._linear_unit_score(
            fft_entropy_value,
            config.EXPERIMENTAL_PRESENTATION_FFT_ENTROPY_START,
            config.EXPERIMENTAL_PRESENTATION_FFT_ENTROPY_FULL,
        )
        fft_high_to_low_support = self._linear_unit_score(
            fft_high_to_low_value,
            config.EXPERIMENTAL_PRESENTATION_FFT_HIGH_TO_LOW_START,
            config.EXPERIMENTAL_PRESENTATION_FFT_HIGH_TO_LOW_FULL,
        )
        fft_support = (
            fft_middle_support
            + fft_entropy_support
            + fft_high_to_low_support
        ) / 3.0

        clipping_support_score = (
            config.EXPERIMENTAL_PRESENTATION_DCT_SUPPORT_WEIGHT * dct_support
            + config.EXPERIMENTAL_PRESENTATION_FFT_SUPPORT_WEIGHT
            * fft_support
        )
        clipping_support_is_sufficient = (
            clipping_score > 0.0
            and clipping_support_score
            >= config.EXPERIMENTAL_PRESENTATION_MINIMUM_SUPPORT
        )
        if clipping_support_is_sufficient:
            clipping_presentation_score = 100.0 * clipping_score * (
                config.EXPERIMENTAL_PRESENTATION_CLIPPING_BASE_WEIGHT
                + config.EXPERIMENTAL_PRESENTATION_SUPPORT_WEIGHT
                * clipping_support_score
            )
        else:
            clipping_presentation_score = 0.0

        dct_middle_value = 0.0
        dct_high_value = 0.0
        dct_near_zero_value = 1.0
        if dct_result is not None:
            dct_middle_value = self._feature_number(
                dct_result.raw_features,
                "middle_frequency_ac_energy_ratio",
            )
            dct_high_value = self._feature_number(
                dct_result.raw_features,
                "high_frequency_ac_energy_ratio",
            )
            measured_near_zero = self._finite_number(
                dct_result.raw_features.get(
                    "near_zero_ac_coefficient_ratio"
                )
            )
            if measured_near_zero is not None:
                dct_near_zero_value = measured_near_zero
        dct_broadband_support = np.mean(
            (
                self._linear_unit_score(
                    dct_middle_value,
                    config.EXPERIMENTAL_PRESENTATION_DCT_MIDDLE_START,
                    config.EXPERIMENTAL_PRESENTATION_DCT_MIDDLE_FULL,
                ),
                self._linear_unit_score(
                    dct_high_value,
                    config.EXPERIMENTAL_PRESENTATION_DCT_HIGH_START,
                    config.EXPERIMENTAL_PRESENTATION_DCT_HIGH_FULL,
                ),
                self._inverse_linear_unit_score(
                    dct_near_zero_value,
                    config.EXPERIMENTAL_PRESENTATION_DCT_DENSE_COEFFICIENT_START,
                    config.EXPERIMENTAL_PRESENTATION_DCT_DENSE_COEFFICIENT_FULL,
                ),
            )
        )

        residual_result = observations["residual"]["result"]
        residual_rms_value = 0.0
        laplacian_variance_value = 0.0
        edge_density_value = 0.0
        if residual_result is not None:
            residual_rms_value = self._feature_number(
                residual_result.raw_features,
                "gaussian_residual_rms_energy",
            )
            laplacian_variance_value = self._feature_number(
                residual_result.raw_features,
                "laplacian_variance",
            )
            edge_density_value = self._feature_number(
                residual_result.raw_features,
                "high_frequency_edge_density",
            )
        residual_broadband_support = np.mean(
            (
                self._linear_unit_score(
                    residual_rms_value,
                    config.EXPERIMENTAL_PRESENTATION_RESIDUAL_RMS_START,
                    config.EXPERIMENTAL_PRESENTATION_RESIDUAL_RMS_FULL,
                ),
                self._linear_unit_score(
                    laplacian_variance_value,
                    config.EXPERIMENTAL_PRESENTATION_LAPLACIAN_VARIANCE_START,
                    config.EXPERIMENTAL_PRESENTATION_LAPLACIAN_VARIANCE_FULL,
                ),
                self._linear_unit_score(
                    edge_density_value,
                    config.EXPERIMENTAL_PRESENTATION_EDGE_DENSITY_START,
                    config.EXPERIMENTAL_PRESENTATION_EDGE_DENSITY_FULL,
                ),
            )
        )

        wavelet_result = observations["wavelet"]["result"]
        wavelet_sparsity_value = 1.0
        if wavelet_result is not None:
            measured_sparsity = self._finite_number(
                wavelet_result.raw_features.get(
                    "global_detail_sparsity_mean"
                )
            )
            if measured_sparsity is not None:
                wavelet_sparsity_value = measured_sparsity
        wavelet_broadband_support = self._inverse_linear_unit_score(
            wavelet_sparsity_value,
            config.EXPERIMENTAL_PRESENTATION_WAVELET_DENSE_DETAIL_START,
            config.EXPERIMENTAL_PRESENTATION_WAVELET_DENSE_DETAIL_FULL,
        )
        transform_support = (
            config.EXPERIMENTAL_PRESENTATION_TRANSFORM_DCT_WEIGHT
            * dct_broadband_support
            + config.EXPERIMENTAL_PRESENTATION_TRANSFORM_RESIDUAL_WEIGHT
            * residual_broadband_support
            + config.EXPERIMENTAL_PRESENTATION_TRANSFORM_WAVELET_WEIGHT
            * wavelet_broadband_support
        )
        transform_support_count = sum(
            value
            >= config.EXPERIMENTAL_PRESENTATION_BROADBAND_COMPONENT_MINIMUM
            for value in (
                dct_broadband_support,
                residual_broadband_support,
                wavelet_broadband_support,
            )
        )
        broadband_support_score = (
            config.EXPERIMENTAL_PRESENTATION_BROADBAND_FFT_WEIGHT
            * fft_support
            + config.EXPERIMENTAL_PRESENTATION_BROADBAND_TRANSFORM_WEIGHT
            * transform_support
        )
        broadband_support_is_sufficient = (
            fft_support
            >= config.EXPERIMENTAL_PRESENTATION_BROADBAND_FFT_MINIMUM
            and transform_support
            >= config.EXPERIMENTAL_PRESENTATION_BROADBAND_TRANSFORM_MINIMUM
            and transform_support_count
            >= config.EXPERIMENTAL_PRESENTATION_BROADBAND_MINIMUM_SUPPORT_COUNT
        )
        broadband_presentation_score = 0.0
        if broadband_support_is_sufficient:
            broadband_minimum = (
                config.EXPERIMENTAL_PRESENTATION_BROADBAND_MINIMUM_SCORE
            )
            broadband_presentation_score = broadband_minimum + (
                100.0 - broadband_minimum
            ) * self._linear_unit_score(
                broadband_support_score,
                config.EXPERIMENTAL_PRESENTATION_BROADBAND_SUPPORT_START,
                config.EXPERIMENTAL_PRESENTATION_BROADBAND_SUPPORT_FULL,
            )

        partial_support_count = sum(
            value
            >= config.EXPERIMENTAL_PRESENTATION_PARTIAL_COMPONENT_MINIMUM
            for value in (
                dct_broadband_support,
                residual_broadband_support,
                wavelet_broadband_support,
            )
        )
        partial_support_is_sufficient = (
            not broadband_support_is_sufficient
            and fft_support
            >= config.EXPERIMENTAL_PRESENTATION_PARTIAL_FFT_MINIMUM
            and transform_support
            >= config.EXPERIMENTAL_PRESENTATION_PARTIAL_TRANSFORM_MINIMUM
            and partial_support_count
            >= config.EXPERIMENTAL_PRESENTATION_PARTIAL_MINIMUM_SUPPORT_COUNT
        )
        partial_presentation_score = 0.0
        if partial_support_is_sufficient:
            partial_minimum = (
                config.EXPERIMENTAL_PRESENTATION_PARTIAL_MINIMUM_SCORE
            )
            partial_maximum = (
                config.EXPERIMENTAL_PRESENTATION_PARTIAL_MAXIMUM_SCORE
            )
            partial_presentation_score = partial_minimum + (
                partial_maximum - partial_minimum
            ) * self._linear_unit_score(
                broadband_support_score,
                config.EXPERIMENTAL_PRESENTATION_PARTIAL_SUPPORT_START,
                config.EXPERIMENTAL_PRESENTATION_PARTIAL_SUPPORT_FULL,
            )
        broadband_presentation_score = max(
            broadband_presentation_score,
            partial_presentation_score,
        )

        presentation_score = max(
            clipping_presentation_score,
            broadband_presentation_score,
        )
        support_is_sufficient = (
            clipping_support_is_sufficient
            or broadband_support_is_sufficient
            or partial_support_is_sufficient
        )
        if broadband_presentation_score > clipping_presentation_score:
            presentation_mode = (
                "broadband_display_texture"
                if broadband_support_is_sufficient
                else "partial_broadband_texture"
            )
        elif clipping_presentation_score > 0.0:
            presentation_mode = "supported_clipping"
        else:
            presentation_mode = "none"

        return {
            "clipped_pixel_ratios": clipping_ratios,
            "maximum_clipped_pixel_ratio": clipping_ratio,
            "clipping_score": float(100.0 * clipping_score),
            "dct_local_inconsistency_value": dct_local_value,
            "dct_support_score": float(100.0 * dct_support),
            "fft_middle_frequency_energy_ratio": fft_middle_value,
            "fft_spectral_entropy": fft_entropy_value,
            "fft_high_to_low_energy_ratio": fft_high_to_low_value,
            "fft_support_score": float(100.0 * fft_support),
            "clipping_support_score": float(
                100.0 * clipping_support_score
            ),
            "clipping_support_is_sufficient": bool(
                clipping_support_is_sufficient
            ),
            "clipping_presentation_score": float(
                np.clip(clipping_presentation_score, 0.0, 100.0)
            ),
            "dct_broadband_support_score": float(
                100.0 * dct_broadband_support
            ),
            "residual_broadband_support_score": float(
                100.0 * residual_broadband_support
            ),
            "wavelet_broadband_support_score": float(
                100.0 * wavelet_broadband_support
            ),
            "transform_broadband_support_score": float(
                100.0 * transform_support
            ),
            "transform_broadband_support_count": int(
                transform_support_count
            ),
            "broadband_support_score": float(
                100.0 * broadband_support_score
            ),
            "broadband_support_is_sufficient": bool(
                broadband_support_is_sufficient
            ),
            "partial_support_count": int(partial_support_count),
            "partial_support_is_sufficient": bool(
                partial_support_is_sufficient
            ),
            "partial_presentation_score": float(
                np.clip(partial_presentation_score, 0.0, 100.0)
            ),
            "broadband_presentation_score": float(
                np.clip(broadband_presentation_score, 0.0, 100.0)
            ),
            "frame_structure": {
                "enabled": False,
                "available": False,
                "included_in_fusion": False,
                "score": None,
                "exclusion_reason": (
                    "Screen-border evidence disabled by configuration"
                ),
            },
            "combined_support_score": float(
                100.0
                * max(
                    clipping_support_score,
                    broadband_support_score,
                )
            ),
            "support_is_sufficient": bool(support_is_sufficient),
            "presentation_mode": presentation_mode,
            "presentation_artifact_score": float(
                np.clip(presentation_score, 0.0, 100.0)
            ),
            "interpretation": (
                "supporting presentation evidence; not a standalone "
                "authenticity verdict"
            ),
        }

    def _linear_unit_score(self, value, start, full):
        if full <= start:
            return 0.0
        return float(np.clip((value - start) / (full - start), 0.0, 1.0))

    def _inverse_linear_unit_score(self, value, start, full):
        if full >= start:
            return 0.0
        return float(np.clip((start - value) / (start - full), 0.0, 1.0))

    def _load_calibration(self):
        path = config.MODEL_FREE_CALIBRATION_FILE_PATH
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as calibration_file:
                document = json.load(calibration_file)
            section = document.get("mathematical_fusion")
            if not isinstance(section, dict):
                return None
            baselines = section.get("bona_fide_baseline")
            mapping = section.get("score_mapping")
            if not isinstance(baselines, dict) or not isinstance(mapping, dict):
                return None
            baseline = self._validated_baseline(
                baselines.get("combined_mathematical_risk_score")
            )
            validated_mapping = self._validated_score_mapping(mapping)
            if baseline is None or validated_mapping is None:
                return None
            return {
                "combined_baseline": baseline,
                "score_mapping": validated_mapping,
                "module_weights": self._validated_weight_overrides(
                    section.get("module_weights"),
                    self.experimental_config["module_weights"],
                ),
                "group_weights": self._validated_weight_overrides(
                    section.get("group_weights"),
                    self.experimental_config["group_weights"],
                ),
                "status_thresholds": self._validated_threshold_overrides(
                    section.get("status_thresholds")
                ),
                "source_section": section,
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print("Mathematical fusion calibration could not be loaded: " + str(error))
            return None

    def _validated_baseline(self, baseline):
        if not isinstance(baseline, dict):
            return None
        try:
            mean = float(baseline["mean"])
            standard_deviation = float(baseline["standard_deviation"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            return None
        if standard_deviation <= 0.0:
            return None
        return {"mean": mean, "standard_deviation": standard_deviation}

    def _validated_score_mapping(self, mapping):
        try:
            start = float(mapping["z_score_start"])
            full = float(mapping["z_score_full"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(start) or not math.isfinite(full) or full <= start:
            return None
        return {"z_score_start": start, "z_score_full": full}

    def _validated_weight_overrides(self, overrides, defaults):
        if not isinstance(overrides, dict):
            return None
        validated = {}
        try:
            for name in defaults:
                value = float(overrides[name])
                if not math.isfinite(value) or value <= 0.0:
                    return None
                validated[name] = value
        except (KeyError, TypeError, ValueError):
            return None
        return validated

    def _validated_threshold_overrides(self, thresholds):
        if not isinstance(thresholds, dict):
            return None
        names = (
            "weak_anomaly_score",
            "suspicious_score",
            "high_risk_score",
            "recovery_score",
        )
        try:
            values = {name: float(thresholds[name]) for name in names}
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values.values()):
            return None
        if not (
            0.0 <= values["weak_anomaly_score"]
            < values["suspicious_score"]
            < values["high_risk_score"]
            <= 100.0
        ):
            return None
        if not 0.0 <= values["recovery_score"] < values["suspicious_score"]:
            return None
        return values

    def _active_configuration(self):
        active = copy.deepcopy(self.experimental_config)
        if self.calibration is None:
            return active
        if self.calibration["module_weights"] is not None:
            active["module_weights"] = self.calibration["module_weights"]
        if self.calibration["group_weights"] is not None:
            active["group_weights"] = self.calibration["group_weights"]
        if self.calibration["status_thresholds"] is not None:
            active.update(self.calibration["status_thresholds"])
        return active

    def _current_temporal_schema_signature(self):
        return (
            int(config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION),
            int(config.MODEL_FREE_FUSION_CONFIGURATION_VERSION),
            bool(config.MODEL_FREE_FRAME_STRUCTURE_ENABLED),
        )

    def _reset_if_temporal_schema_changed(self):
        current_signature = self._current_temporal_schema_signature()
        if current_signature == self.temporal_schema_signature:
            return
        self._reset_temporal_state()
        self.previous_region = None
        self.temporal_schema_signature = current_signature

    def _handle_region_change(self, face_box):
        current = self._region_tuple(face_box)
        if current is None:
            return
        if self.previous_region is not None:
            overlap = self._intersection_over_union(
                self.previous_region,
                current,
            )
            if overlap < float(
                self.active_config["region_iou_reset_threshold"]
            ):
                self._reset_temporal_state()
        self.previous_region = current

    def _reset_temporal_state(self):
        self.score_history.clear()
        self.presentation_score_history.clear()
        self.consecutive_suspicious_frames = 0
        self.consecutive_recovery_frames = 0
        self.warning_is_active = False
        self.presentation_warning_is_active = False
        self.invalid_streak = 0

    def _region_tuple(self, face_box):
        if face_box is None:
            return None
        return (
            int(face_box.x),
            int(face_box.y),
            int(face_box.width),
            int(face_box.height),
        )

    def _intersection_over_union(self, first, second):
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second
        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = (
            first_width * first_height
            + second_width * second_height
            - intersection
        )
        return intersection / union if union > 0 else 0.0

    def _context_quality(self, context):
        return {
            "valid": context.face_quality_valid,
            "reason": context.quality_reason,
            "blur": context.blur_value,
            "brightness": context.brightness_value,
            "exposure_valid": context.exposure_valid,
            "pose_alignment_valid": context.pose_alignment_valid,
            "face_dimensions": context.face_dimensions,
            "analysis_dimensions": context.analysis_dimensions,
        }

    def _feature_number(self, mapping, name):
        value = self._finite_number(mapping.get(name))
        return value if value is not None else 0.0

    def _finite_number(self, value):
        if isinstance(value, (bool, np.bool_)):
            return None
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None


MathematicalFusionPreController = MathematicalFusionController
