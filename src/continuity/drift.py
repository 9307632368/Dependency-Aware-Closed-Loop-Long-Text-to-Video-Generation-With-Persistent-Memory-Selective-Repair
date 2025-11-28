# src/continuity/drift.py
from __future__ import annotations

from typing import Dict, Any, List


class DriftDetector:
    """
    Drift detector built on top of score reports.

    Purpose:
    - normalize drift information into a consistent structure
    - provide retry / repair decision signals
    - classify whether drift is:
        low
        medium
        high
    - expose scene-level decision helpers for pipeline use

    This file is intentionally lightweight and compatible with the
    rewritten `ConsistencyScorer`.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        drift_cfg = self._safe_dict(
            continuity_cfg.get("drift", self.config.get("drift", {}))
        )

        self.retry_threshold = float(drift_cfg.get("retry_threshold", 0.60))
        self.repair_threshold = float(drift_cfg.get("repair_threshold", 0.50))
        self.hard_fail_threshold = float(drift_cfg.get("hard_fail_threshold", 0.35))

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def detect(
        self,
        score_report: Dict[str, Any] = None,
        scene_packet: Dict[str, Any] = None,
        generation_result: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        score_report = self._safe_dict(score_report)
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)

        drift_report = self._safe_dict(score_report.get("drift_report", {}))
        drift_flags = self._dedupe_keep_order(
            [self._safe_text(x) for x in self._safe_list(drift_report.get("drift_flags", []))]
        )

        severity = self._safe_text(drift_report.get("severity", "low")).lower()
        overall_score = self._extract_score(score_report)
        hard_fail = bool(score_report.get("hard_fail", False))
        accepted = bool(score_report.get("accepted", False))

        # strengthen decision if generation itself failed
        if not bool(generation_result.get("ok", True)):
            hard_fail = True
            if "generation_failure" not in drift_flags:
                drift_flags.append("generation_failure")
            severity = "high"

        retry_recommended = False
        repair_recommended = False
        downstream_risk = False

        if overall_score < self.retry_threshold:
            retry_recommended = True

        if overall_score < self.repair_threshold:
            repair_recommended = True

        if hard_fail or overall_score < self.hard_fail_threshold:
            repair_recommended = True

        if any(
            flag in drift_flags
            for flag in [
                "identity_drift",
                "location_drift",
                "transition_drift",
                "transition_break",
            ]
        ):
            downstream_risk = True

        if severity in {"medium", "high"}:
            retry_recommended = True

        if severity == "high":
            repair_recommended = True

        decision = self._build_decision(
            accepted=accepted,
            hard_fail=hard_fail,
            retry_recommended=retry_recommended,
            repair_recommended=repair_recommended,
            severity=severity,
        )

        return {
            "ok": True,
            "scene_id": self._safe_text(scene_packet.get("scene_id", "")),
            "score": overall_score,
            "accepted": accepted,
            "hard_fail": hard_fail,
            "severity": severity if severity else "low",
            "drift_flags": drift_flags,
            "retry_recommended": retry_recommended,
            "repair_recommended": repair_recommended,
            "downstream_risk": downstream_risk,
            "decision": decision,
        }

    def should_retry(self, drift_report: Dict[str, Any]) -> bool:
        drift_report = self._safe_dict(drift_report)
        if bool(drift_report.get("hard_fail", False)):
            return False
        return bool(drift_report.get("retry_recommended", False))

    def should_repair(self, drift_report: Dict[str, Any]) -> bool:
        drift_report = self._safe_dict(drift_report)
        return bool(drift_report.get("repair_recommended", False))

    def should_invalidate_downstream(self, drift_report: Dict[str, Any]) -> bool:
        drift_report = self._safe_dict(drift_report)
        return bool(drift_report.get("downstream_risk", False))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_decision(
        self,
        accepted: bool,
        hard_fail: bool,
        retry_recommended: bool,
        repair_recommended: bool,
        severity: str,
    ) -> str:
        if hard_fail:
            return "hard_fail"
        if repair_recommended:
            return "repair"
        if retry_recommended:
            return "retry"
        if accepted:
            return "accept"
        if severity == "medium":
            return "retry"
        return "accept"

    def _extract_score(self, score_report: Dict[str, Any]) -> float:
        score_report = self._safe_dict(score_report)
        if "overall_score" in score_report:
            return self._safe_float(score_report.get("overall_score", 0.0), 0.0)
        if "score" in score_report:
            return self._safe_float(score_report.get("score", 0.0), 0.0)
        return 0.0

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        out = []
        seen = set()

        for item in items or []:
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


# backward-compatible functional helpers
_default_detector = DriftDetector()


def detect_drift(
    score_report: Dict[str, Any] = None,
    scene_packet: Dict[str, Any] = None,
    generation_result: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return _default_detector.detect(
        score_report=score_report,
        scene_packet=scene_packet,
        generation_result=generation_result,
    )


def should_retry(drift_report: Dict[str, Any]) -> bool:
    return _default_detector.should_retry(drift_report)


def should_repair(drift_report: Dict[str, Any]) -> bool:
    return _default_detector.should_repair(drift_report)


def should_invalidate_downstream(drift_report: Dict[str, Any]) -> bool:
    return _default_detector.should_invalidate_downstream(drift_report)