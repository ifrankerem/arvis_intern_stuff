"""Attack-specific, uncertainty-aware deterministic PreControl decisions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import config


FINAL_STATES = {
    "LIVE",
    "SUSPICIOUS",
    "HIGH_RISK",
    "INSUFFICIENT_QUALITY",
    "INSUFFICIENT_EVIDENCE",
    "UNSUPPORTED_CAPTURE",
}


@dataclass(frozen=True)
class EvidenceFamilyResult:
    family_name: str
    supported: bool
    score_0_100: Optional[float]
    reliability: float
    coverage: float
    methods: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "family_name": self.family_name,
            "supported": self.supported,
            "score_0_100": self.score_0_100,
            "reliability": self.reliability,
            "coverage": self.coverage,
            "methods": list(self.methods),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class AttackRiskResult:
    attack_name: str
    supported: bool
    score_0_100: Optional[float]
    reliability: float
    contributing_families: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "attack_name": self.attack_name,
            "supported": self.supported,
            "score_0_100": self.score_0_100,
            "reliability": self.reliability,
            "contributing_families": list(self.contributing_families),
        }


@dataclass(frozen=True)
class PreControlDecision:
    overall_risk_0_100: Optional[float]
    overall_reliability: float
    uncertainty_0_100: float
    risk_interval_0_100: List[Optional[float]]
    classification: str
    attack_scores: Dict[str, AttackRiskResult]
    family_scores: Dict[str, EvidenceFamilyResult]
    reason_codes: List[str]
    human_explanation: str
    calibrated: bool
    score_valid: bool
    runtime_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "overall_risk_0_100": self.overall_risk_0_100,
            "overall_reliability": self.overall_reliability,
            "uncertainty_0_100": self.uncertainty_0_100,
            "risk_interval_0_100": list(self.risk_interval_0_100),
            "classification": self.classification,
            "attack_scores": {
                name: value.to_dict()
                for name, value in self.attack_scores.items()
            },
            "family_scores": {
                name: value.to_dict()
                for name, value in self.family_scores.items()
            },
            "reason_codes": list(self.reason_codes),
            "human_explanation": self.human_explanation,
            "calibrated": self.calibrated,
            "score_valid": self.score_valid,
            "runtime_ms": self.runtime_ms,
            "warnings": list(self.warnings),
        }


class PreControlDecisionBuilder:
    """Fuse method results in two deterministic, reliability-aware stages."""

    def __init__(self, decision_config=None):
        self.config = dict(
            decision_config or config.PRECONTROL_ATTACK_FUSION_CONFIG
        )

    def build(self, method_results, context, combined_result, runtime_ms=0.0):
        if context is None:
            return self._unsupported("Shared analysis context is unavailable", runtime_ms)
        if not context.face_quality_valid:
            reason = context.quality_reason or "Image quality gate failed"
            return self._quality_failure(reason, runtime_ms)

        families = self._family_results(method_results)
        attacks = {
            attack_name: self._attack_result(
                attack_name,
                weights,
                families,
            )
            for attack_name, weights in self.config[
                "attack_family_weights"
            ].items()
        }
        supported_attacks = [
            result
            for result in attacks.values()
            if result.supported and result.score_0_100 is not None
        ]
        if not supported_attacks:
            return self._unsupported(
                "No attack score has sufficient supported evidence",
                runtime_ms,
                families=families,
                attacks=attacks,
            )

        highest = max(supported_attacks, key=lambda value: value.score_0_100)
        overall_score = float(np.clip(highest.score_0_100, 0.0, 100.0))
        overall_reliability = float(np.clip(highest.reliability, 0.0, 1.0))
        uncertainty = 100.0 * (1.0 - overall_reliability)
        lower = overall_score * overall_reliability
        upper = overall_score + (100.0 - overall_score) * (
            1.0 - overall_reliability
        )
        calibrated = bool(
            combined_result is not None
            and combined_result.available
            and combined_result.calibrated
        )
        supported_family_count = sum(
            1 for family in families.values() if family.supported
        )
        classification, state_codes = self._classification(
            overall_score,
            overall_reliability,
            supported_family_count,
            calibrated,
        )
        method_codes = []
        for result in method_results.values():
            if result.triggered:
                method_codes.extend(result.reason_codes)
        reason_codes = self._unique(method_codes + state_codes)
        target_label = highest.attack_name.removesuffix("_score")
        explanation = self._explanation(
            classification,
            target_label,
            overall_score,
            overall_reliability,
            families,
            calibrated,
        )
        warnings = []
        if not calibrated:
            warnings.append(
                "Scores use experimental mappings; LIVE is disabled until calibration"
            )
        if overall_reliability < self.config["minimum_live_reliability"]:
            warnings.append("Evidence coverage is insufficient for a LIVE claim")
        return PreControlDecision(
            overall_risk_0_100=overall_score,
            overall_reliability=overall_reliability,
            uncertainty_0_100=uncertainty,
            risk_interval_0_100=[lower, upper],
            classification=classification,
            attack_scores=attacks,
            family_scores=families,
            reason_codes=reason_codes,
            human_explanation=explanation,
            calibrated=calibrated,
            score_valid=True,
            runtime_ms=max(0.0, float(runtime_ms)),
            warnings=warnings,
        )

    def _family_results(self, method_results):
        grouped = {}
        for method_key, result in method_results.items():
            family = result.evidence_family or "unassigned"
            grouped.setdefault(family, []).append((method_key, result))

        families = {}
        for family, expected_count in self.config[
            "expected_family_methods"
        ].items():
            results = grouped.get(family, [])
            supported = [
                (method_key, result)
                for method_key, result in results
                if result.available
                and result.score is not None
                and result.reliability > 0.0
            ]
            coverage = min(1.0, len(supported) / max(1.0, float(expected_count)))
            weighted = [
                (
                    result,
                    float(
                        self.config.get("method_weights", {}).get(
                            method_key,
                            1.0,
                        )
                    )
                    * result.reliability,
                )
                for method_key, result in supported
            ]
            denominator = sum(weight for _result, weight in weighted)
            score = (
                sum(result.score * weight for result, weight in weighted)
                / denominator
                if denominator > 0.0
                else None
            )
            mean_reliability = (
                sum(result.reliability for _key, result in supported)
                / max(1.0, float(len(supported)))
            )
            family_reliability = float(
                np.clip(mean_reliability * coverage, 0.0, 1.0)
            )
            codes = self._unique(
                code
                for _key, result in supported
                if result.triggered
                for code in result.reason_codes
            )
            families[family] = EvidenceFamilyResult(
                family_name=family,
                supported=score is not None,
                score_0_100=(
                    None if score is None else float(np.clip(score, 0.0, 100.0))
                ),
                reliability=family_reliability,
                coverage=coverage,
                methods=[result.module_name for _key, result in supported],
                reason_codes=codes,
            )
        return families

    def _attack_result(self, attack_name, weights, families):
        numerator = 0.0
        effective_weight = 0.0
        configured_weight = sum(float(value) for value in weights.values())
        contributors = []
        for family_name, weight in weights.items():
            family = families.get(family_name)
            if family is None or not family.supported:
                continue
            effective = float(weight) * family.reliability
            numerator += family.score_0_100 * effective
            effective_weight += effective
            contributors.append(family_name)
        score = numerator / effective_weight if effective_weight > 0.0 else None
        reliability = (
            effective_weight / configured_weight
            if configured_weight > 0.0
            else 0.0
        )
        return AttackRiskResult(
            attack_name=attack_name,
            supported=score is not None,
            score_0_100=(
                None if score is None else float(np.clip(score, 0.0, 100.0))
            ),
            reliability=float(np.clip(reliability, 0.0, 1.0)),
            contributing_families=contributors,
        )

    def _classification(
        self,
        score,
        reliability,
        supported_family_count,
        calibrated,
    ):
        if (
            supported_family_count < self.config["minimum_supported_families"]
            or reliability < self.config["minimum_decision_reliability"]
        ):
            return "INSUFFICIENT_EVIDENCE", ["INSUFFICIENT_EVIDENCE_COVERAGE"]
        if not calibrated:
            if score >= self.config["suspicious_score"]:
                return "SUSPICIOUS", ["UNCALIBRATED_SUSPICIOUS_EVIDENCE"]
            return "INSUFFICIENT_EVIDENCE", ["DEPLOYMENT_CALIBRATION_REQUIRED"]
        if score >= self.config["high_risk_score"]:
            return "HIGH_RISK", ["MULTI_FAMILY_HIGH_RISK"]
        if score >= self.config["suspicious_score"]:
            return "SUSPICIOUS", ["MULTI_FAMILY_SUSPICIOUS"]
        if reliability >= self.config["minimum_live_reliability"]:
            return "LIVE", ["CALIBRATED_LOW_RISK_WITH_COVERAGE"]
        return "INSUFFICIENT_EVIDENCE", ["INSUFFICIENT_EVIDENCE_COVERAGE"]

    def _quality_failure(self, reason, runtime_ms):
        return PreControlDecision(
            overall_risk_0_100=None,
            overall_reliability=0.0,
            uncertainty_0_100=100.0,
            risk_interval_0_100=[None, None],
            classification="INSUFFICIENT_QUALITY",
            attack_scores={},
            family_scores={},
            reason_codes=["INSUFFICIENT_IMAGE_QUALITY"],
            human_explanation=reason,
            calibrated=False,
            score_valid=False,
            runtime_ms=max(0.0, float(runtime_ms)),
            warnings=["No risk score was invented for an invalid frame"],
        )

    def _unsupported(
        self,
        reason,
        runtime_ms,
        families=None,
        attacks=None,
    ):
        return PreControlDecision(
            overall_risk_0_100=None,
            overall_reliability=0.0,
            uncertainty_0_100=100.0,
            risk_interval_0_100=[None, None],
            classification="UNSUPPORTED_CAPTURE",
            attack_scores=dict(attacks or {}),
            family_scores=dict(families or {}),
            reason_codes=["UNSUPPORTED_CAPTURE"],
            human_explanation=reason,
            calibrated=False,
            score_valid=False,
            runtime_ms=max(0.0, float(runtime_ms)),
            warnings=["No risk score was invented for unsupported evidence"],
        )

    def _explanation(
        self,
        classification,
        target,
        score,
        reliability,
        families,
        calibrated,
    ):
        contributors = sorted(
            (
                family
                for family in families.values()
                if family.supported and family.score_0_100 is not None
            ),
            key=lambda item: item.score_0_100,
            reverse=True,
        )
        names = ", ".join(item.family_name for item in contributors[:3])
        calibration_note = (
            "Deployment calibration is active."
            if calibrated
            else "Deployment calibration is absent, so LIVE is withheld."
        )
        return (
            f"{classification}: highest supported risk is {target} at "
            f"{score:.1f}/100 with reliability {reliability:.2f}. "
            f"Contributing evidence families: {names or 'none'}. "
            f"{calibration_note}"
        )

    @staticmethod
    def _unique(values):
        unique = []
        for value in values:
            if value and value not in unique:
                unique.append(value)
        return unique
