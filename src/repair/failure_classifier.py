# src/repair/failure_classifier.py
from __future__ import annotations

from typing import Dict, Any, List


class FailureClassifier:
    """
    Classify scene generation failures into actionable repair categories.

    Input sources:
    - score_report from consistency scorer
    - drift_report from drift detector / scorer
    - generation_result metadata
    - scene_packet metadata

    Output:
    {
        "ok": True,
        "scene_id": "...",
        "primary_failure": "identity_drift",
        "failure_tags": [...],
        "severity": "medium",
        "repair_strategy_hint": "identity_lock_repair",
        "downstream_risk": True,
        "needs_full_regeneration": False,
        "needs_reference_reprioritization": True,
        "needs_prompt_rewrite": True,
        "needs_motion_suppression": False,
        "needs_transition_repair": False,
    }
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        repair_cfg = self._safe_dict(self.config.get("repair", {}))
        classifier_cfg = self._safe_dict(
            repair_cfg.get("failure_classifier", self.config.get("failure_classifier", {}))
        )

        self.high_severity_threshold = float(
            classifier_cfg.get("high_severity_threshold", 0.45)
        )
        self.full_regeneration_threshold = float(
            classifier_cfg.get("full_regeneration_threshold", 0.30)
        )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def classify(
        self,
        score_report: Dict[str, Any] = None,
        drift_report: Dict[str, Any] = None,
        generation_result: Dict[str, Any] = None,
        scene_packet: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        score_report = self._safe_dict(score_report)
        drift_report = self._safe_dict(drift_report)
        generation_result = self._safe_dict(generation_result)
        scene_packet = self._safe_dict(scene_packet)

        scene_id = self._safe_text(
            generation_result.get("scene_id", "")
            or scene_packet.get("scene_id", "")
        )

        overall_score = self._extract_score(score_report)
        hard_fail = bool(score_report.get("hard_fail", False))
        scorer_failure_tags = self._safe_list(score_report.get("failure_tags", []))
        drift_flags = self._safe_list(drift_report.get("drift_flags", []))
        severity = self._normalize_severity(
            self._safe_text(drift_report.get("severity", "low")),
            overall_score,
            hard_fail,
        )

        failure_tags = self._merge_tags(scorer_failure_tags, drift_flags)

        if not bool(generation_result.get("ok", True)):
            failure_tags = self._merge_tags(failure_tags, ["generation_failure"])
            severity = "high"

        primary_failure = self._pick_primary_failure(
            failure_tags=failure_tags,
            severity=severity,
            hard_fail=hard_fail,
            overall_score=overall_score,
        )

        strategy_hint = self._map_primary_failure_to_strategy(primary_failure)

        needs_full_regeneration = self._needs_full_regeneration(
            primary_failure=primary_failure,
            severity=severity,
            overall_score=overall_score,
            hard_fail=hard_fail,
        )

        needs_reference_reprioritization = self._needs_reference_reprioritization(
            failure_tags
        )
        needs_prompt_rewrite = self._needs_prompt_rewrite(
            failure_tags=failure_tags,
            severity=severity,
        )
        needs_motion_suppression = self._needs_motion_suppression(failure_tags)
        needs_transition_repair = self._needs_transition_repair(failure_tags)
        downstream_risk = self._has_downstream_risk(failure_tags, severity)

        return {
            "ok": True,
            "scene_id": scene_id,
            "primary_failure": primary_failure,
            "failure_tags": failure_tags,
            "severity": severity,
            "repair_strategy_hint": strategy_hint,
            "downstream_risk": downstream_risk,
            "needs_full_regeneration": needs_full_regeneration,
            "needs_reference_reprioritization": needs_reference_reprioritization,
            "needs_prompt_rewrite": needs_prompt_rewrite,
            "needs_motion_suppression": needs_motion_suppression,
            "needs_transition_repair": needs_transition_repair,
            "score": overall_score,
            "hard_fail": hard_fail,
        }

    # ------------------------------------------------------------------
    # logic
    # ------------------------------------------------------------------

    def _pick_primary_failure(
        self,
        failure_tags: List[str],
        severity: str,
        hard_fail: bool,
        overall_score: float,
    ) -> str:
        tags = [self._safe_text(x).lower() for x in failure_tags]

        priority = [
            "generation_failure",
            "identity_drift",
            "location_drift",
            "transition_break",
            "transition_drift",
            "prop_loss",
            "props_drift",
            "style_drift",
            "motion_drift",
            "low_score",
        ]

        for tag in priority:
            if tag in tags:
                return tag

        if hard_fail:
            return "hard_fail"
        if overall_score < self.full_regeneration_threshold:
            return "low_score"
        if severity == "high":
            return "severe_drift"
        return "general_drift"

    def _map_primary_failure_to_strategy(self, primary_failure: str) -> str:
        mapping = {
            "generation_failure": "full_scene_regeneration",
            "identity_drift": "identity_lock_repair",
            "location_drift": "location_lock_repair",
            "transition_break": "transition_repair",
            "transition_drift": "transition_repair",
            "prop_loss": "prop_repair",
            "props_drift": "prop_repair",
            "style_drift": "style_repair",
            "motion_drift": "motion_stabilization_repair",
            "low_score": "prompt_rewrite_and_regenerate",
            "hard_fail": "full_scene_regeneration",
            "severe_drift": "full_scene_regeneration",
            "general_drift": "continuity_strengthened_retry",
        }
        return mapping.get(primary_failure, "continuity_strengthened_retry")

    def _needs_full_regeneration(
        self,
        primary_failure: str,
        severity: str,
        overall_score: float,
        hard_fail: bool,
    ) -> bool:
        if hard_fail:
            return True
        if primary_failure in {"generation_failure", "hard_fail", "severe_drift"}:
            return True
        if overall_score < self.full_regeneration_threshold:
            return True
        if severity == "high" and primary_failure in {"identity_drift", "location_drift", "transition_break"}:
            return True
        return False

    def _needs_reference_reprioritization(self, failure_tags: List[str]) -> bool:
        tags = [self._safe_text(x).lower() for x in failure_tags]
        target_tags = {
            "identity_drift",
            "location_drift",
            "transition_break",
            "transition_drift",
            "prop_loss",
            "props_drift",
        }
        return any(tag in target_tags for tag in tags)

    def _needs_prompt_rewrite(self, failure_tags: List[str], severity: str) -> bool:
        tags = [self._safe_text(x).lower() for x in failure_tags]
        if severity in {"medium", "high"}:
            return True
        target_tags = {
            "style_drift",
            "motion_drift",
            "low_score",
            "general_drift",
        }
        return any(tag in target_tags for tag in tags)

    def _needs_motion_suppression(self, failure_tags: List[str]) -> bool:
        tags = [self._safe_text(x).lower() for x in failure_tags]
        return any(tag in {"motion_drift", "transition_break", "transition_drift"} for tag in tags)

    def _needs_transition_repair(self, failure_tags: List[str]) -> bool:
        tags = [self._safe_text(x).lower() for x in failure_tags]
        return any(tag in {"transition_break", "transition_drift"} for tag in tags)

    def _has_downstream_risk(self, failure_tags: List[str], severity: str) -> bool:
        tags = [self._safe_text(x).lower() for x in failure_tags]
        if severity == "high":
            return True
        risky = {
            "identity_drift",
            "location_drift",
            "transition_break",
            "transition_drift",
        }
        return any(tag in risky for tag in tags)

    def _normalize_severity(
        self,
        severity: str,
        overall_score: float,
        hard_fail: bool,
    ) -> str:
        severity = self._safe_text(severity).lower()

        if hard_fail:
            return "high"

        if overall_score < self.full_regeneration_threshold:
            return "high"
        if overall_score < self.high_severity_threshold:
            return "medium"

        if severity in {"none", "low", "medium", "high"}:
            return severity
        return "low"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _extract_score(self, score_report: Dict[str, Any]) -> float:
        score_report = self._safe_dict(score_report)
        if "overall_score" in score_report:
            return self._safe_float(score_report.get("overall_score", 0.0), 0.0)
        if "score" in score_report:
            return self._safe_float(score_report.get("score", 0.0), 0.0)
        return 0.0

    def _merge_tags(self, *tag_lists: List[Any]) -> List[str]:
        out = []
        seen = set()

        for tag_list in tag_lists:
            for item in tag_list or []:
                text = self._safe_text(item)
                if not text:
                    continue
                low = text.lower()
                if low not in seen:
                    seen.add(low)
                    out.append(text)

        return out

    def _safe_text(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {}

    def _safe_list(self, x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return []

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default