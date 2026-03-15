# src/repair/repair_policy.py
from __future__ import annotations

import copy
from typing import Dict, Any, List


class RepairPolicy:
    """
    Convert failure classification into an actionable repair plan.

    This module decides:
    - whether to retry or fully regenerate
    - how to strengthen prompts
    - how to modify control weights
    - whether to reprioritize references
    - whether to suppress motion
    - whether downstream scenes should be invalidated

    Output:
    {
        "ok": True,
        "scene_id": "...",
        "strategy": "identity_lock_repair",
        "use_full_regeneration": False,
        "use_repair_backend": True,
        "retry_index": 2,
        "failure_tags": [...],
        "prompt_actions": {...},
        "control_actions": {...},
        "reference_actions": {...},
        "generation_actions": {...},
        "downstream_actions": {...},
    }
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        repair_cfg = self._safe_dict(self.config.get("repair", {}))
        policy_cfg = self._safe_dict(
            repair_cfg.get("policy", self.config.get("repair_policy", {}))
        )

        self.max_retry_index = int(policy_cfg.get("max_retry_index", 3))
        self.default_guidance_boost = float(policy_cfg.get("default_guidance_boost", 0.4))
        self.default_step_boost = int(policy_cfg.get("default_step_boost", 2))
        self.default_reference_boost = float(policy_cfg.get("default_reference_boost", 0.1))

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def build_plan(
        self,
        classification: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        scene_packet: Dict[str, Any] = None,
        generation_result: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        classification = self._safe_dict(classification)
        prompt_bundle = self._safe_dict(prompt_bundle)
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)

        scene_id = self._safe_text(
            classification.get("scene_id", "")
            or prompt_bundle.get("scene_id", "")
            or scene_packet.get("scene_id", "")
        )
        strategy = self._safe_text(classification.get("repair_strategy_hint", "continuity_strengthened_retry"))
        failure_tags = self._safe_list(classification.get("failure_tags", []))
        severity = self._safe_text(classification.get("severity", "low"))
        primary_failure = self._safe_text(classification.get("primary_failure", ""))

        current_retry_index = int(
            self._safe_dict(prompt_bundle.get("prompt_metadata", {})).get("retry_index", 0) or 0
        )
        next_retry_index = min(self.max_retry_index, current_retry_index + 1)

        use_full_regeneration = bool(classification.get("needs_full_regeneration", False))
        use_repair_backend = True
        if use_full_regeneration and primary_failure == "generation_failure":
            use_repair_backend = False

        plan = {
            "ok": True,
            "scene_id": scene_id,
            "strategy": strategy,
            "use_full_regeneration": use_full_regeneration,
            "use_repair_backend": use_repair_backend,
            "retry_index": next_retry_index,
            "failure_tags": failure_tags,
            "prompt_actions": self._build_prompt_actions(
                classification=classification,
                prompt_bundle=prompt_bundle,
            ),
            "control_actions": self._build_control_actions(
                classification=classification,
                prompt_bundle=prompt_bundle,
            ),
            "reference_actions": self._build_reference_actions(
                classification=classification,
                prompt_bundle=prompt_bundle,
            ),
            "generation_actions": self._build_generation_actions(
                classification=classification,
                prompt_bundle=prompt_bundle,
            ),
            "downstream_actions": self._build_downstream_actions(
                classification=classification,
                scene_packet=scene_packet,
            ),
        }

        # extra top-level hints for convenience
        plan["severity"] = severity
        plan["primary_failure"] = primary_failure
        plan["dependent_on_previous"] = bool(scene_packet.get("dependent_on_previous", False))

        return plan

    def apply_plan_to_prompt_bundle(
        self,
        prompt_bundle: Dict[str, Any],
        repair_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply repair decisions to prompt bundle so SceneGenerator/backend can reuse it.
        """
        prompt_bundle = copy.deepcopy(self._safe_dict(prompt_bundle))
        repair_plan = self._safe_dict(repair_plan)

        prompt_actions = self._safe_dict(repair_plan.get("prompt_actions", {}))
        control_actions = self._safe_dict(repair_plan.get("control_actions", {}))
        reference_actions = self._safe_dict(repair_plan.get("reference_actions", {}))
        generation_actions = self._safe_dict(repair_plan.get("generation_actions", {}))

        # prompt metadata
        prompt_metadata = self._safe_dict(prompt_bundle.get("prompt_metadata", {}))
        prompt_metadata["retry_index"] = int(repair_plan.get("retry_index", 0) or 0)
        prompt_metadata["repair_mode"] = True
        prompt_metadata["failure_tags"] = self._safe_list(repair_plan.get("failure_tags", []))
        prompt_metadata["repair_strategy"] = self._safe_text(repair_plan.get("strategy", ""))
        prompt_bundle["prompt_metadata"] = prompt_metadata

        # retry context
        retry_context = self._safe_dict(prompt_bundle.get("retry_context", {}))
        retry_context["retry_index"] = int(repair_plan.get("retry_index", 0) or 0)
        retry_context["failure_tags"] = self._safe_list(repair_plan.get("failure_tags", []))
        retry_context["repair_strategy"] = self._safe_text(repair_plan.get("strategy", ""))
        retry_context["downstream_actions"] = self._safe_dict(repair_plan.get("downstream_actions", {}))
        prompt_bundle["retry_context"] = retry_context

        # prompt text fields
        extra_prompt_lines = self._safe_list(prompt_actions.get("extra_prompt_lines", []))
        extra_negative_terms = self._safe_list(prompt_actions.get("extra_negative_terms", []))

        for key in ["positive_prompt", "model_prompt", "repair_prompt", "continuity_prompt"]:
            old_text = self._safe_text(prompt_bundle.get(key, ""))
            prompt_bundle[key] = self._append_sentences(old_text, extra_prompt_lines)

        old_negative = self._safe_text(prompt_bundle.get("negative_prompt", ""))
        prompt_bundle["negative_prompt"] = self._append_csv_terms(old_negative, extra_negative_terms)

        if self._safe_text(prompt_bundle.get("prompt", "")):
            prompt_bundle["prompt"] = self._append_sentences(
                self._safe_text(prompt_bundle.get("prompt", "")),
                extra_prompt_lines,
            )

        # control weights
        control_weights = copy.deepcopy(self._safe_dict(prompt_bundle.get("control_weights", {})))
        for key, delta in self._safe_dict(control_actions.get("weight_deltas", {})).items():
            current = self._safe_float(control_weights.get(key, 0.0), 0.0)
            control_weights[key] = self._clamp01(current + self._safe_float(delta, 0.0))

        prompt_bundle["control_weights"] = control_weights

        # generation params
        generation_params = copy.deepcopy(self._safe_dict(prompt_bundle.get("generation_params", {})))
        if "guidance_scale_delta" in generation_actions:
            generation_params["guidance_scale"] = self._safe_float(
                generation_params.get("guidance_scale", 5.0), 5.0
            ) + self._safe_float(generation_actions.get("guidance_scale_delta", 0.0), 0.0)

        if "num_inference_steps_delta" in generation_actions:
            generation_params["num_inference_steps"] = int(
                self._safe_float(generation_params.get("num_inference_steps", 20), 20)
                + self._safe_float(generation_actions.get("num_inference_steps_delta", 0.0), 0.0)
            )

        if "reference_strength_delta" in generation_actions:
            generation_params["reference_strength"] = self._clamp01(
                self._safe_float(generation_params.get("reference_strength", 0.70), 0.70)
                + self._safe_float(generation_actions.get("reference_strength_delta", 0.0), 0.0)
            )

        if generation_actions.get("suppress_motion", False):
            generation_params["motion_scheduler_bias"] = "stable"
            if "motion_strength" in control_weights:
                control_weights["motion_strength"] = max(
                    0.15,
                    self._safe_float(control_weights.get("motion_strength", 0.5), 0.5) - 0.12,
                )

        prompt_bundle["generation_params"] = generation_params
        prompt_bundle["control_weights"] = control_weights

        # reference policy
        reference_bundle = copy.deepcopy(self._safe_dict(prompt_bundle.get("reference_bundle", {})))
        reprioritize_to = self._safe_text(reference_actions.get("reprioritize_to", ""))

        if reprioritize_to == "identity":
            path = self._extract_role_reference(reference_bundle, "character_refs")
            if path:
                reference_bundle["primary_reference_type"] = "identity_reference"
                reference_bundle["primary_reference_path"] = path
                reference_bundle["primary_reference_reason"] = "repair policy identity reprioritization"

        elif reprioritize_to == "location":
            loc = self._safe_dict(reference_bundle.get("location_ref", {}))
            path = self._safe_text(loc.get("path", ""))
            if path:
                reference_bundle["primary_reference_type"] = "location_reference"
                reference_bundle["primary_reference_path"] = path
                reference_bundle["primary_reference_reason"] = "repair policy location reprioritization"

        elif reprioritize_to == "transition":
            path = self._safe_text(reference_bundle.get("previous_scene_keyframe", ""))
            if path:
                reference_bundle["primary_reference_type"] = "transition_reference"
                reference_bundle["primary_reference_path"] = path
                reference_bundle["primary_reference_reason"] = "repair policy transition reprioritization"

        elif reprioritize_to == "prop":
            path = self._extract_role_reference(reference_bundle, "prop_refs")
            if path:
                reference_bundle["primary_reference_type"] = "prop_reference"
                reference_bundle["primary_reference_path"] = path
                reference_bundle["primary_reference_reason"] = "repair policy prop reprioritization"

        prompt_bundle["reference_bundle"] = reference_bundle

        return prompt_bundle

    # ------------------------------------------------------------------
    # sub-plans
    # ------------------------------------------------------------------

    def _build_prompt_actions(
        self,
        classification: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        failure_tags = [self._safe_text(x).lower() for x in self._safe_list(classification.get("failure_tags", []))]
        severity = self._safe_text(classification.get("severity", "low")).lower()

        extra_prompt_lines: List[str] = []
        extra_negative_terms: List[str] = []

        extra_prompt_lines.append(
            "Repair mode: preserve already-correct content and only strengthen inconsistent continuity elements."
        )

        if "identity_drift" in failure_tags:
            extra_prompt_lines.append(
                "Preserve the exact same character identity, same face structure, same hairstyle, same clothing, and same accessories."
            )
            extra_negative_terms.extend(["different face", "changed identity", "wrong hairstyle"])

        if "location_drift" in failure_tags:
            extra_prompt_lines.append(
                "Preserve the same location layout, same background anchors, same lighting continuity, and same environment appearance."
            )
            extra_negative_terms.extend(["wrong background", "different location", "changed environment"])

        if "prop_loss" in failure_tags or "props_drift" in failure_tags:
            extra_prompt_lines.append(
                "Keep the important props present, visually consistent, and correctly related to the character."
            )
            extra_negative_terms.extend(["missing prop", "wrong object", "prop inconsistency"])

        if "style_drift" in failure_tags:
            extra_prompt_lines.append(
                "Preserve the same cinematic style, same color tone, same shot feeling, and same mood."
            )
            extra_negative_terms.extend(["style mismatch", "different cinematic tone"])

        if "transition_break" in failure_tags or "transition_drift" in failure_tags:
            extra_prompt_lines.append(
                "Make the opening of this scene feel like a natural continuation of the ending of the previous scene."
            )
            extra_negative_terms.extend(["hard scene reset", "abrupt transition", "transition break"])

        if "motion_drift" in failure_tags:
            extra_prompt_lines.append(
                "Use more controlled and stable motion to reduce drift and preserve continuity."
            )
            extra_negative_terms.extend(["unstable motion", "camera jitter", "excessive motion"])

        if severity in {"medium", "high"}:
            extra_prompt_lines.append(
                "Be extra strict about continuity and reduce creative drift."
            )
            extra_negative_terms.extend(["continuity drift", "scene mismatch", "subject mismatch"])

        return {
            "extra_prompt_lines": self._dedupe_keep_order(extra_prompt_lines),
            "extra_negative_terms": self._dedupe_keep_order(extra_negative_terms),
        }

    def _build_control_actions(
        self,
        classification: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        failure_tags = [self._safe_text(x).lower() for x in self._safe_list(classification.get("failure_tags", []))]
        weight_deltas: Dict[str, float] = {}

        # baseline continuity strengthening
        weight_deltas["continuity_strength"] = 0.10

        if "identity_drift" in failure_tags:
            weight_deltas["identity_strength"] = 0.18

        if "location_drift" in failure_tags:
            weight_deltas["location_strength"] = 0.18

        if "prop_loss" in failure_tags or "props_drift" in failure_tags:
            weight_deltas["prop_strength"] = 0.18

        if "style_drift" in failure_tags:
            weight_deltas["style_strength"] = 0.15

        if "transition_break" in failure_tags or "transition_drift" in failure_tags:
            weight_deltas["transition_strength"] = 0.20

        if "motion_drift" in failure_tags:
            weight_deltas["motion_strength"] = -0.12

        return {
            "weight_deltas": weight_deltas
        }

    def _build_reference_actions(
        self,
        classification: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        primary_failure = self._safe_text(classification.get("primary_failure", "")).lower()
        needs_reprioritization = bool(classification.get("needs_reference_reprioritization", False))

        reprioritize_to = ""

        if needs_reprioritization:
            if primary_failure == "identity_drift":
                reprioritize_to = "identity"
            elif primary_failure == "location_drift":
                reprioritize_to = "location"
            elif primary_failure in {"transition_break", "transition_drift"}:
                reprioritize_to = "transition"
            elif primary_failure in {"prop_loss", "props_drift"}:
                reprioritize_to = "prop"

        return {
            "reprioritize_to": reprioritize_to
        }

    def _build_generation_actions(
        self,
        classification: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        severity = self._safe_text(classification.get("severity", "low")).lower()
        suppress_motion = bool(classification.get("needs_motion_suppression", False))

        guidance_boost = self.default_guidance_boost
        step_boost = self.default_step_boost
        reference_boost = self.default_reference_boost

        if severity == "medium":
            guidance_boost += 0.2
            step_boost += 1
        elif severity == "high":
            guidance_boost += 0.4
            step_boost += 3
            reference_boost += 0.08

        return {
            "guidance_scale_delta": guidance_boost,
            "num_inference_steps_delta": step_boost,
            "reference_strength_delta": reference_boost,
            "suppress_motion": suppress_motion,
        }

    def _build_downstream_actions(
        self,
        classification: Dict[str, Any],
        scene_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        downstream_risk = bool(classification.get("downstream_risk", False))
        dependent_on_previous = bool(scene_packet.get("dependent_on_previous", False))

        return {
            "invalidate_downstream_if_repaired": downstream_risk,
            "dependent_on_previous": dependent_on_previous,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _extract_role_reference(self, reference_bundle: Dict[str, Any], key: str) -> str:
        items = self._safe_list(reference_bundle.get(key, []))
        for item in items:
            item = self._safe_dict(item)
            path = self._safe_text(item.get("path", ""))
            if path:
                return path
        return ""

    def _append_sentences(self, base: str, lines: List[str]) -> str:
        base = self._safe_text(base)
        lines = [self._safe_text(x) for x in lines if self._safe_text(x)]
        if not lines:
            return base
        if not base:
            return " ".join(lines).strip()
        return (base + " " + " ".join(lines)).strip()

    def _append_csv_terms(self, base: str, terms: List[str]) -> str:
        base_items = [x.strip() for x in self._safe_text(base).split(",") if x.strip()]
        all_items = base_items + [self._safe_text(x) for x in terms if self._safe_text(x)]
        return ", ".join(self._dedupe_keep_order(all_items)).strip()

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

    def _clamp01(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)

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