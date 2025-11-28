# src/continuity/consistency_scorer.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

try:
    from PIL import Image
    import numpy as np
except Exception:
    Image = None
    np = None


class ConsistencyScorer:
    """
    Conservative but practical consistency scorer.

    Important behavior:
    - fallback/debug outputs are rejected
    - placeholder-only outputs are rejected
    - failed generations are rejected
    - degenerate frames are rejected
    - real successful generated video with usable keyframes can pass even if
      semantic extraction is only partial/weak

    Reason:
    In early stage, generated_summary may be weak because we do not yet have
    CLIP/GroundingDINO/face detector. But if CogVideoX genuinely generated
    real frames, scene 1 should be allowed to commit reference frames so scene 2
    can continue properly.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        scoring_cfg = self._safe_dict(
            continuity_cfg.get("scoring", self.config.get("scoring", {}))
        )

        self.threshold_accept = float(scoring_cfg.get("threshold_accept", 0.60))
        self.threshold_hard_fail = float(scoring_cfg.get("threshold_hard_fail", 0.35))

        self.weight_identity = float(scoring_cfg.get("weight_identity", 0.28))
        self.weight_location = float(scoring_cfg.get("weight_location", 0.22))
        self.weight_props = float(scoring_cfg.get("weight_props", 0.15))
        self.weight_style = float(scoring_cfg.get("weight_style", 0.15))
        self.weight_transition = float(scoring_cfg.get("weight_transition", 0.20))

        self.hard_fail_on_identity = bool(scoring_cfg.get("hard_fail_on_identity", False))
        self.hard_fail_on_location = bool(scoring_cfg.get("hard_fail_on_location", False))

        # New practical option:
        # Allow real successful video/keyframes to pass with partial visual evidence.
        self.allow_real_video_bootstrap = bool(scoring_cfg.get("allow_real_video_bootstrap", True))
        self.real_video_bootstrap_score = float(scoring_cfg.get("real_video_bootstrap_score", 0.68))
        self.real_video_partial_cap = float(scoring_cfg.get("real_video_partial_cap", 0.72))
        self.real_video_missing_cap = float(scoring_cfg.get("real_video_missing_cap", 0.62))

        visual_cfg = self._safe_dict(scoring_cfg.get("visual_sanity", {}))
        self.enable_visual_sanity = bool(visual_cfg.get("enabled", True))
        self.min_std_threshold = float(visual_cfg.get("min_std_threshold", 6.0))
        self.max_dominant_ratio = float(visual_cfg.get("max_dominant_ratio", 0.95))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def score(
        self,
        scene_packet: Dict[str, Any] = None,
        generation_result: Dict[str, Any] = None,
        prompt_bundle: Dict[str, Any] = None,
        continuity_package: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)
        prompt_bundle = self._safe_dict(prompt_bundle)
        continuity_package = self._safe_dict(continuity_package)

        metadata = self._safe_dict(generation_result.get("metadata", {}))
        prompt_contract = self._safe_dict(prompt_bundle.get("prompt_contract", {}))

        invalid_tags = self._invalid_generation_tags(generation_result, metadata)
        visual_tags = self._visual_sanity_tags(generation_result)

        forced_fail_tags = self._dedupe_keep_order(invalid_tags + visual_tags)

        # Hard reject only real invalid output types.
        if forced_fail_tags:
            return {
                "ok": False,
                "score": 0.05,
                "overall_score": 0.05,
                "accepted": False,
                "hard_fail": True,
                "failure_tags": forced_fail_tags,
                "semantic_evidence_status": self._safe_text(metadata.get("semantic_evidence_status", "missing")) or "missing",
                "has_semantic_evidence": bool(metadata.get("has_semantic_evidence", False)),
                "placeholder_conditioning_only": bool(metadata.get("placeholder_conditioning_only", False)),
                "drift_report": {
                    "drift_flags": forced_fail_tags,
                    "severity": "high",
                    "semantic_evidence_status": self._safe_text(metadata.get("semantic_evidence_status", "missing")) or "missing",
                    "has_semantic_evidence": bool(metadata.get("has_semantic_evidence", False)),
                    "placeholder_conditioning_only": bool(metadata.get("placeholder_conditioning_only", False)),
                    "component_scores": {
                        "identity": 0.0,
                        "location": 0.0,
                        "props": 0.0,
                        "style": 0.0,
                        "transition": 0.0,
                    },
                    "component_details": {
                        "identity": {"reason": "forced_fail_output"},
                        "location": {"reason": "forced_fail_output"},
                        "props": {"reason": "forced_fail_output"},
                        "style": {"reason": "forced_fail_output"},
                        "transition": {"reason": "forced_fail_output"},
                    },
                },
            }

        expected = self._build_expected_state(
            scene_packet=scene_packet,
            prompt_contract=prompt_contract,
            continuity_package=continuity_package,
        )
        observed = self._build_observed_state(
            generation_result=generation_result,
            metadata=metadata,
            scene_packet=scene_packet,
            prompt_bundle=prompt_bundle,
        )

        evidence_status = self._safe_text(
            metadata.get("semantic_evidence_status", "")
            or observed.get("semantic_evidence_status", "")
            or "missing"
        ).lower()

        has_semantic_evidence = bool(
            metadata.get(
                "has_semantic_evidence",
                observed.get("has_semantic_evidence", False),
            )
        )

        placeholder_conditioning_only = bool(
            metadata.get("placeholder_conditioning_only", False)
            or observed.get("placeholder_conditioning_only", False)
        )

        real_video_ok = self._is_real_video_output(generation_result, metadata)

        identity_score, identity_info = self._score_identity(expected, observed)
        location_score, location_info = self._score_location(expected, observed)
        props_score, props_info = self._score_props(expected, observed)
        style_score, style_info = self._score_style(expected, observed)
        transition_score, transition_info = self._score_transition(expected, observed)

        weighted_score = (
            identity_score * self.weight_identity
            + location_score * self.weight_location
            + props_score * self.weight_props
            + style_score * self.weight_style
            + transition_score * self.weight_transition
        )

        # --------------------------------------------------------------
        # Evidence policy
        # --------------------------------------------------------------
        if placeholder_conditioning_only:
            weighted_score = min(weighted_score, 0.10)

        elif evidence_status == "missing" or not has_semantic_evidence:
            if self.allow_real_video_bootstrap and real_video_ok:
                # This is the key fix:
                # real generated frames can be accepted initially, but only with
                # moderate confidence. Later CLIP/GroundingDINO can replace this.
                weighted_score = max(weighted_score, self.real_video_bootstrap_score)
                weighted_score = min(weighted_score, self.real_video_missing_cap)
            else:
                weighted_score = min(weighted_score, 0.30)

        elif evidence_status == "partial":
            if self.allow_real_video_bootstrap and real_video_ok:
                weighted_score = max(weighted_score, self.real_video_bootstrap_score)
                weighted_score = min(weighted_score, self.real_video_partial_cap)
            else:
                weighted_score = min(weighted_score, 0.50)

        failure_tags = []
        drift_flags = []

        failure_tags.extend(identity_info.get("failure_tags", []))
        failure_tags.extend(location_info.get("failure_tags", []))
        failure_tags.extend(props_info.get("failure_tags", []))
        failure_tags.extend(style_info.get("failure_tags", []))
        failure_tags.extend(transition_info.get("failure_tags", []))

        drift_flags.extend(identity_info.get("drift_flags", []))
        drift_flags.extend(location_info.get("drift_flags", []))
        drift_flags.extend(props_info.get("drift_flags", []))
        drift_flags.extend(style_info.get("drift_flags", []))
        drift_flags.extend(transition_info.get("drift_flags", []))

        # If real video passed by bootstrap, do not mark missing evidence as fatal.
        if evidence_status == "missing" and real_video_ok:
            drift_flags = [x for x in drift_flags if x != "missing_semantic_evidence"]
            failure_tags = [x for x in failure_tags if x != "missing_semantic_evidence"]

        failure_tags = self._dedupe_keep_order(failure_tags)
        drift_flags = self._dedupe_keep_order(drift_flags)

        severity = self._severity_from_component_scores(
            identity_score=identity_score,
            location_score=location_score,
            props_score=props_score,
            style_score=style_score,
            transition_score=transition_score,
            drift_flags=drift_flags,
            real_video_ok=real_video_ok,
        )

        hard_fail = False
        if weighted_score < self.threshold_hard_fail:
            hard_fail = True
        if self.hard_fail_on_identity and "identity_drift" in drift_flags:
            hard_fail = True
        if self.hard_fail_on_location and "location_drift" in drift_flags:
            hard_fail = True

        if not bool(generation_result.get("ok", False)):
            hard_fail = True
            if "generation_failure" not in drift_flags:
                drift_flags.append("generation_failure")
            if "generation_failure" not in failure_tags:
                failure_tags.append("generation_failure")

        accepted = (weighted_score >= self.threshold_accept) and (not hard_fail)

        drift_report = {
            "drift_flags": drift_flags,
            "severity": severity,
            "semantic_evidence_status": evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "placeholder_conditioning_only": placeholder_conditioning_only,
            "real_video_ok": real_video_ok,
            "component_scores": {
                "identity": round(identity_score, 6),
                "location": round(location_score, 6),
                "props": round(props_score, 6),
                "style": round(style_score, 6),
                "transition": round(transition_score, 6),
            },
            "component_details": {
                "identity": identity_info,
                "location": location_info,
                "props": props_info,
                "style": style_info,
                "transition": transition_info,
            },
        }

        return {
            "ok": True,
            "score": round(weighted_score, 6),
            "overall_score": round(weighted_score, 6),
            "accepted": accepted,
            "hard_fail": hard_fail,
            "failure_tags": failure_tags,
            "semantic_evidence_status": evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "placeholder_conditioning_only": placeholder_conditioning_only,
            "real_video_ok": real_video_ok,
            "drift_report": drift_report,
        }

    # ------------------------------------------------------------------
    # invalid generation checks
    # ------------------------------------------------------------------

    def _invalid_generation_tags(self, generation_result: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
        tags = []

        if bool(metadata.get("is_fallback_output", False)):
            tags.extend(["generation_failure", "placeholder_output", "fallback_output"])

        if bool(metadata.get("generation_failed", False)):
            tags.append("generation_failure")

        if bool(metadata.get("placeholder_conditioning_only", False)):
            tags.append("placeholder_conditioning")

        quality_score = metadata.get("quality_score", 1.0)
        try:
            quality_score = float(quality_score)
        except Exception:
            quality_score = 1.0

        if quality_score <= 0.10:
            tags.append("placeholder_output")

        if not bool(generation_result.get("ok", False)):
            tags.append("generation_failure")

        return self._dedupe_keep_order(tags)

    def _visual_sanity_tags(self, generation_result: Dict[str, Any]) -> List[str]:
        if not self.enable_visual_sanity:
            return []

        keyframe_path = self._first_nonempty([
            generation_result.get("best_keyframe_path", ""),
            generation_result.get("transition_frame_path", ""),
            generation_result.get("identity_frame_path", ""),
            generation_result.get("selected_keyframe_path", ""),
            generation_result.get("keyframe_path", ""),
            generation_result.get("middle_frame_path", ""),
            generation_result.get("first_frame_path", ""),
            generation_result.get("last_frame_path", ""),
        ])

        if not keyframe_path:
            return []

        if Image is None or np is None:
            return []

        if not os.path.isfile(keyframe_path):
            return []

        try:
            img = Image.open(keyframe_path).convert("RGB")
            arr = np.array(img)

            std_val = float(arr.std())
            flat_like = std_val < self.min_std_threshold

            pixels = arr.reshape(-1, 3)
            if len(pixels) == 0:
                return ["low_detail_output"]

            sample = pixels[::max(1, len(pixels) // 5000)]
            unique, counts = np.unique(sample, axis=0, return_counts=True)
            dominant_ratio = float(counts.max()) / float(counts.sum()) if len(counts) > 0 else 1.0
            dominant_like = dominant_ratio >= self.max_dominant_ratio

            tags = []
            if flat_like:
                tags.append("low_detail_output")
            if dominant_like:
                tags.append("degenerate_frame")

            return self._dedupe_keep_order(tags)
        except Exception:
            return []

    def _is_real_video_output(self, generation_result: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        if not bool(generation_result.get("ok", False)):
            return False

        if bool(metadata.get("generation_failed", False)):
            return False
        if bool(metadata.get("is_fallback_output", False)):
            return False
        if bool(metadata.get("placeholder_conditioning_only", False)):
            return False

        video_path = self._first_nonempty([
            generation_result.get("video_path", ""),
            generation_result.get("output_video_path", ""),
            metadata.get("video_path", ""),
        ])

        keyframe_path = self._first_nonempty([
            generation_result.get("best_keyframe_path", ""),
            generation_result.get("transition_frame_path", ""),
            generation_result.get("identity_frame_path", ""),
            generation_result.get("selected_keyframe_path", ""),
            generation_result.get("keyframe_path", ""),
            generation_result.get("middle_frame_path", ""),
            generation_result.get("first_frame_path", ""),
            generation_result.get("last_frame_path", ""),
            metadata.get("best_keyframe_path", ""),
            metadata.get("transition_frame_path", ""),
            metadata.get("identity_frame_path", ""),
            metadata.get("middle_frame_path", ""),
            metadata.get("first_frame_path", ""),
            metadata.get("last_frame_path", ""),
        ])

        if not video_path or not keyframe_path:
            return False

        if not os.path.isfile(video_path):
            return False
        if not os.path.isfile(keyframe_path):
            return False

        quality_score = self._safe_float(metadata.get("quality_score", 0.0), 0.0)
        if quality_score < 0.50:
            return False

        return True

    # ------------------------------------------------------------------
    # expected / observed state
    # ------------------------------------------------------------------

    def _build_expected_state(
        self,
        scene_packet: Dict[str, Any],
        prompt_contract: Dict[str, Any],
        continuity_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)
        prompt_contract = self._safe_dict(prompt_contract)

        identity_lock = self._safe_dict(prompt_contract.get("identity_lock", {}))
        location_lock = self._safe_dict(prompt_contract.get("location_lock", {}))
        prop_lock = self._safe_dict(prompt_contract.get("prop_lock", {}))
        style_lock = self._safe_dict(prompt_contract.get("style_lock", {}))
        allowed_changes = self._safe_dict(prompt_contract.get("allowed_changes", {}))
        same_as_previous = self._safe_dict(prompt_contract.get("same_as_previous", {}))
        story_core = self._safe_dict(prompt_contract.get("story_core", {}))

        return {
            "scene_id": self._safe_text(prompt_contract.get("scene_id", "") or scene_packet.get("scene_id", "")),
            "dependent_on_previous": bool(
                story_core.get("dependent_on_previous", scene_packet.get("dependent_on_previous", False))
            ),
            "same_as_previous": same_as_previous,
            "characters": self._safe_list(identity_lock.get("characters", [])),
            "locked_names": self._safe_list(identity_lock.get("locked_names", [])),
            "location": location_lock,
            "props": self._safe_list(prop_lock.get("props", [])),
            "prop_names": self._safe_list(prop_lock.get("names", [])),
            "style": style_lock,
            "allowed_changes": allowed_changes,
            "scene_text": self._safe_text(prompt_contract.get("scene_text", "")),
        }

    def _build_observed_state(
        self,
        generation_result: Dict[str, Any],
        metadata: Dict[str, Any],
        scene_packet: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(metadata)

        generated_summary = self._safe_dict(metadata.get("generated_summary", {}))
        reference_plan = self._safe_dict(metadata.get("reference_plan", {}))
        scene_policy = self._safe_dict(metadata.get("scene_policy", {}))

        characters = self._safe_list(generated_summary.get("characters", []))
        location = self._safe_dict(generated_summary.get("location", {}))
        props = self._safe_list(generated_summary.get("props", []))
        style = self._safe_dict(generated_summary.get("style", {}))
        summary_source = self._safe_text(generated_summary.get("source", "missing") or "missing").lower()

        has_semantic_evidence = bool(characters or location or props or style)
        semantic_evidence_status = self._safe_text(metadata.get("semantic_evidence_status", "") or "").lower()

        if not semantic_evidence_status:
            semantic_evidence_status = "present" if has_semantic_evidence else "missing"

        if not has_semantic_evidence and semantic_evidence_status == "present":
            semantic_evidence_status = "partial"

        return {
            "ok": bool(generation_result.get("ok", False)),
            "scene_id": self._safe_text(generation_result.get("scene_id", "")),
            "characters": characters,
            "location": location,
            "props": props,
            "style": style,
            "generated_summary_source": summary_source,
            "semantic_evidence_status": semantic_evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "placeholder_conditioning_only": bool(metadata.get("placeholder_conditioning_only", False)),
            "has_video": bool(
                self._safe_text(generation_result.get("video_path", ""))
                or self._safe_text(generation_result.get("output_video_path", ""))
            ),
            "has_keyframe": bool(
                self._safe_text(generation_result.get("keyframe_path", ""))
                or self._safe_text(generation_result.get("best_keyframe_path", ""))
                or self._safe_text(generation_result.get("selected_keyframe_path", ""))
                or self._safe_text(generation_result.get("transition_frame_path", ""))
                or self._safe_text(generation_result.get("identity_frame_path", ""))
            ),
            "reference_plan": reference_plan,
            "scene_policy": scene_policy,
        }

    # ------------------------------------------------------------------
    # component scoring
    # ------------------------------------------------------------------

    def _score_identity(self, expected: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        same_as_previous = self._safe_dict(expected.get("same_as_previous", {}))
        expected_chars = self._safe_list(expected.get("characters", []))
        observed_chars = self._safe_list(observed.get("characters", []))

        if not bool(same_as_previous.get("character_identity", False)):
            return 0.70, {"reason": "identity_not_strictly_locked", "failure_tags": [], "drift_flags": []}

        expected_names = self._extract_names(expected_chars)
        observed_names = self._extract_names(observed_chars)

        if not expected_names:
            return 0.55, {"reason": "no_expected_identity_names", "failure_tags": [], "drift_flags": []}

        if not observed_names:
            return 0.50, {"reason": "missing_observed_identity_names", "failure_tags": [], "drift_flags": []}

        overlap = self._list_overlap_ratio(expected_names, observed_names)
        score = overlap

        failure_tags = []
        drift_flags = []

        if overlap < 0.55:
            failure_tags.append("identity_drift")
            drift_flags.append("identity_drift")

        return self._clamp(score), {
            "score": round(self._clamp(score), 6),
            "failure_tags": failure_tags,
            "drift_flags": drift_flags,
            "expected_names": expected_names,
            "observed_names": observed_names,
        }

    def _score_location(self, expected: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        same_as_previous = self._safe_dict(expected.get("same_as_previous", {}))
        expected_loc = self._safe_dict(expected.get("location", {}))
        observed_loc = self._safe_dict(observed.get("location", {}))

        if not bool(same_as_previous.get("location", False)):
            return 0.70, {"reason": "location_not_strictly_locked", "failure_tags": [], "drift_flags": []}

        score_parts = []

        exp_name = self._safe_text(expected_loc.get("name", ""))
        obs_name = self._safe_text(observed_loc.get("name", "") or observed_loc.get("name_hint", ""))
        if exp_name:
            if obs_name:
                score_parts.append(self._soft_text_match(exp_name, obs_name))
            else:
                score_parts.append(0.55)

        exp_anchors = self._dedupe_keep_order(
            [self._safe_text(x) for x in self._safe_list(expected_loc.get("anchors", []))]
        )
        obs_anchors = self._dedupe_keep_order(
            [self._safe_text(x) for x in self._safe_list(observed_loc.get("anchors", observed_loc.get("anchors_hint", [])))]
        )
        if exp_anchors:
            if obs_anchors:
                score_parts.append(self._list_overlap_ratio(exp_anchors, obs_anchors))
            else:
                score_parts.append(0.55)

        score = sum(score_parts) / len(score_parts) if score_parts else 0.55

        failure_tags = []
        drift_flags = []
        if score < 0.45:
            failure_tags.append("location_drift")
            drift_flags.append("location_drift")

        return self._clamp(score), {
            "score": round(self._clamp(score), 6),
            "failure_tags": failure_tags,
            "drift_flags": drift_flags,
        }

    def _score_props(self, expected: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        same_as_previous = self._safe_dict(expected.get("same_as_previous", {}))
        expected_props = self._safe_list(expected.get("props", []))
        observed_props = self._safe_list(observed.get("props", []))

        if not bool(same_as_previous.get("props", False)):
            return 0.70, {"reason": "props_not_strictly_locked", "failure_tags": [], "drift_flags": []}

        exp_names = self._extract_names(expected_props)
        obs_names = self._extract_names(observed_props)

        if not exp_names:
            return 0.60, {"reason": "no_expected_props", "failure_tags": [], "drift_flags": []}

        if not obs_names:
            return 0.50, {"reason": "missing_observed_props", "failure_tags": [], "drift_flags": []}

        score = self._list_overlap_ratio(exp_names, obs_names)

        failure_tags = []
        drift_flags = []
        if score < 0.50:
            failure_tags.append("prop_loss")
            drift_flags.append("props_drift")

        return self._clamp(score), {
            "score": round(self._clamp(score), 6),
            "failure_tags": failure_tags,
            "drift_flags": drift_flags,
        }

    def _score_style(self, expected: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        same_as_previous = self._safe_dict(expected.get("same_as_previous", {}))
        expected_style = self._safe_dict(expected.get("style", {}))
        observed_style = self._safe_dict(observed.get("style", {}))

        if not bool(same_as_previous.get("style", False)):
            return 0.70, {"reason": "style_not_strictly_locked", "failure_tags": [], "drift_flags": []}

        score_parts = []

        for field in ["visual_style", "color_tone", "shot_type", "camera_angle", "camera_motion", "mood", "lighting"]:
            exp_val = self._safe_text(expected_style.get(field, ""))
            if exp_val:
                obs_val = self._safe_text(
                    observed_style.get(field, "")
                    or observed_style.get(f"{field}_hint", "")
                )
                if obs_val:
                    score_parts.append(self._soft_text_match(exp_val, obs_val))
                else:
                    score_parts.append(0.55)

        score = sum(score_parts) / len(score_parts) if score_parts else 0.60

        failure_tags = []
        drift_flags = []
        if score < 0.45:
            failure_tags.append("style_drift")
            drift_flags.append("style_drift")

        return self._clamp(score), {
            "score": round(self._clamp(score), 6),
            "failure_tags": failure_tags,
            "drift_flags": drift_flags,
        }

    def _score_transition(self, expected: Dict[str, Any], observed: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        dependent = bool(expected.get("dependent_on_previous", False))
        if not dependent:
            return 0.70, {"reason": "scene_not_dependent", "failure_tags": [], "drift_flags": []}

        reference_plan = self._safe_dict(observed.get("reference_plan", {}))
        scene_policy = self._safe_dict(observed.get("scene_policy", {}))

        score_parts = []

        if bool(reference_plan.get("transition_reference_path", "")) or bool(reference_plan.get("primary_reference_type", "") == "transition_reference"):
            score_parts.append(1.0)
        else:
            score_parts.append(0.60)

        if bool(
            scene_policy.get("prefer_transition_guidance", False)
            or scene_policy.get("prefer_transition_anchor", False)
            or scene_policy.get("dependent_scene", False)
        ):
            score_parts.append(0.90)
        else:
            score_parts.append(0.65)

        if bool(observed.get("has_keyframe", False)):
            score_parts.append(0.90)
        else:
            score_parts.append(0.50)

        score = sum(score_parts) / len(score_parts)

        failure_tags = []
        drift_flags = []
        if score < 0.50:
            failure_tags.append("transition_break")
            drift_flags.append("transition_drift")

        return self._clamp(score), {
            "score": round(self._clamp(score), 6),
            "failure_tags": failure_tags,
            "drift_flags": drift_flags,
        }

    # ------------------------------------------------------------------
    # scoring utilities
    # ------------------------------------------------------------------

    def _severity_from_component_scores(
        self,
        identity_score: float,
        location_score: float,
        props_score: float,
        style_score: float,
        transition_score: float,
        drift_flags: List[str],
        real_video_ok: bool = False,
    ) -> str:
        min_score = min(identity_score, location_score, props_score, style_score, transition_score)

        if "generation_failure" in drift_flags or "placeholder_output" in drift_flags or "degenerate_frame" in drift_flags:
            return "high"
        if real_video_ok and min_score >= 0.45:
            return "low"
        if min_score < 0.35:
            return "high"
        if min_score < 0.60:
            return "medium"
        if len(drift_flags) > 0:
            return "low"
        return "none"

    def _extract_names(self, items: List[Any]) -> List[str]:
        names = []
        for item in items or []:
            if isinstance(item, dict):
                name = (
                    self._safe_text(item.get("name", ""))
                    or self._safe_text(item.get("name_hint", ""))
                    or self._safe_text(item.get("label", ""))
                )
                if name:
                    names.append(name)
            else:
                text = self._safe_text(item)
                if text:
                    names.append(text)
        return self._dedupe_keep_order(names)

    def _soft_text_match(self, a: str, b: str) -> float:
        a = self._safe_text(a).lower()
        b = self._safe_text(b).lower()

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            return 0.85

        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0

        overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
        return self._clamp(overlap)

    def _list_overlap_ratio(self, expected: List[str], observed: List[str]) -> float:
        expected = [self._safe_text(x).lower() for x in expected if self._safe_text(x)]
        observed = [self._safe_text(x).lower() for x in observed if self._safe_text(x)]

        if not expected and not observed:
            return 1.0
        if not expected:
            return 1.0
        if not observed:
            return 0.0

        e = set(expected)
        o = set(observed)
        return self._clamp(len(e & o) / max(1, len(e)))

    def _first_nonempty(self, values: List[Any]) -> str:
        for v in values or []:
            t = self._safe_text(v)
            if t:
                return t
        return ""

    def _clamp(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)

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


ContinuityConsistencyScorer = ConsistencyScorer