# src/repair/scene_repair.py
from __future__ import annotations

import copy
from typing import Dict, Any, Optional

from src.repair.failure_classifier import FailureClassifier
from src.repair.repair_policy import RepairPolicy


class SceneRepairEngine:
    """
    Scene-level repair orchestrator.

    This module connects:
    - failure classification
    - repair policy construction
    - prompt bundle modification
    - repaired scene generation via SceneGenerator

    Main goal:
    take a failed / weak scene result and produce a repair attempt
    using a structured closed-loop flow instead of ad-hoc retry text.

    Expected usage:
        engine = SceneRepairEngine(config)
        repair_out = engine.repair_scene(
            scene_generator=scene_generator,
            continuity_package=continuity_package,
            scene_packet=scene_packet,
            original_scene_output=scene_output,
        )

    Output:
    {
        "ok": True/False,
        "scene_id": "...",
        "classification": {...},
        "repair_plan": {...},
        "repaired_prompt_bundle": {...},
        "repair_result": {...},
        "accepted": bool,
        "downstream_actions": {...},
    }
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        self.failure_classifier = FailureClassifier(self.config)
        self.repair_policy = RepairPolicy(self.config)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def repair_scene(
        self,
        scene_generator: Any,
        continuity_package: Dict[str, Any],
        scene_packet: Dict[str, Any],
        original_scene_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        continuity_package = self._safe_dict(continuity_package)
        scene_packet = self._safe_dict(scene_packet)
        original_scene_output = self._safe_dict(original_scene_output)

        scene_id = self._safe_text(
            original_scene_output.get("scene_id", "")
            or continuity_package.get("scene_id", "")
            or scene_packet.get("scene_id", "")
        )

        prompt_bundle = self._safe_dict(original_scene_output.get("prompt_bundle", {}))
        score_report = self._safe_dict(original_scene_output.get("score_report", {}))
        drift_report = self._safe_dict(original_scene_output.get("drift_report", {}))
        generation_result = self._safe_dict(original_scene_output.get("generation_result", {}))

        classification = self.failure_classifier.classify(
            score_report=score_report,
            drift_report=drift_report,
            generation_result=generation_result,
            scene_packet=scene_packet,
        )

        repair_plan = self.repair_policy.build_plan(
            classification=classification,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
            generation_result=generation_result,
        )

        repaired_prompt_bundle = self.repair_policy.apply_plan_to_prompt_bundle(
            prompt_bundle=prompt_bundle,
            repair_plan=repair_plan,
        )

        repaired_continuity_package = self._merge_repaired_prompt_bundle(
            continuity_package=continuity_package,
            repaired_prompt_bundle=repaired_prompt_bundle,
        )

        repair_result = self._run_repair_generation(
            scene_generator=scene_generator,
            repaired_continuity_package=repaired_continuity_package,
            fallback_prompt_bundle=repaired_prompt_bundle,
        )

        accepted = bool(repair_result.get("accepted", False))

        return {
            "ok": bool(repair_result.get("ok", False)),
            "scene_id": scene_id,
            "classification": classification,
            "repair_plan": repair_plan,
            "repaired_prompt_bundle": repaired_prompt_bundle,
            "repair_result": repair_result,
            "accepted": accepted,
            "downstream_actions": self._safe_dict(repair_plan.get("downstream_actions", {})),
        }

    # ------------------------------------------------------------------
    # core helpers
    # ------------------------------------------------------------------

    def _merge_repaired_prompt_bundle(
        self,
        continuity_package: Dict[str, Any],
        repaired_prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        continuity_package = copy.deepcopy(self._safe_dict(continuity_package))
        repaired_prompt_bundle = copy.deepcopy(self._safe_dict(repaired_prompt_bundle))

        continuity_package["prompt_bundle_override"] = repaired_prompt_bundle

        # also copy key fields directly for easier backend / generator access
        continuity_package["control_weights"] = copy.deepcopy(
            self._safe_dict(repaired_prompt_bundle.get("control_weights", {}))
        )
        continuity_package["reference_bundle"] = copy.deepcopy(
            self._safe_dict(repaired_prompt_bundle.get("reference_bundle", {}))
        )
        continuity_package["generation_params"] = copy.deepcopy(
            self._safe_dict(repaired_prompt_bundle.get("generation_params", {}))
        )
        continuity_package["retry_index"] = int(
            self._safe_dict(repaired_prompt_bundle.get("prompt_metadata", {})).get("retry_index", 0) or 0
        )
        continuity_package["failure_tags"] = self._safe_list(
            self._safe_dict(repaired_prompt_bundle.get("prompt_metadata", {})).get("failure_tags", [])
        )

        return continuity_package

    def _run_repair_generation(
        self,
        scene_generator: Any,
        repaired_continuity_package: Dict[str, Any],
        fallback_prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        if scene_generator is None:
            return {
                "ok": False,
                "accepted": False,
                "error": "scene_generator is not available for repair",
                "prompt_bundle": fallback_prompt_bundle,
            }

        # Preferred path: generator understands repair bundle override.
        try:
            out = scene_generator.generate_scene(repaired_continuity_package)
            out = self._safe_dict(out)
            if "prompt_bundle" not in out or not out.get("prompt_bundle"):
                out["prompt_bundle"] = fallback_prompt_bundle
            return out
        except Exception as e:
            return {
                "ok": False,
                "accepted": False,
                "error": f"scene repair generation failed: {e}",
                "prompt_bundle": fallback_prompt_bundle,
            }

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def _safe_text(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {}

    def _safe_list(self, x: Any):
        if isinstance(x, list):
            return x
        return []