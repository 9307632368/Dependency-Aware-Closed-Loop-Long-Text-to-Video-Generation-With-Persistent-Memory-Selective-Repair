# src/continuity/manager.py
from __future__ import annotations

import copy
import os
from typing import Dict, Any, List


class ContinuityManager:
    """
    Continuity orchestration layer.

    Responsibilities:
    - prepare continuity package before generation
    - build same/change constraints
    - build dependency-aware reference bundle
    - update memory and reference bank after accepted generation
    - use role-based keyframes:
        identity_frame_path
        location_frame_path
        prop_frame_path
        transition_frame_path
        style_frame_path

    Important idea:
    Scene 2 should not always use one random/middle keyframe.
    It should use reference frames according to dependency type:
    - same character  -> identity_frame
    - same location   -> location_frame
    - same prop       -> prop_frame
    - direct continuation -> transition_frame
    - same style      -> style_frame
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        self.control_defaults = self._safe_dict(
            continuity_cfg.get("control_defaults", {})
        )

        self.memory = self._build_memory(self.config)
        self.reference_bank = self._build_reference_bank(self.config)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def prepare_scene(self, scene_packet: Dict[str, Any]) -> Dict[str, Any]:
        scene_packet = copy.deepcopy(self._safe_dict(scene_packet))
        scene_id = self._scene_id(scene_packet)

        scene_text = self._extract_scene_text(scene_packet)
        same_constraints = self._build_same_constraints(scene_packet)
        change_constraints = self._build_change_constraints(scene_packet)

        generation_contract = self._build_generation_contract(
            scene_packet=scene_packet,
            same_constraints=same_constraints,
            change_constraints=change_constraints,
        )

        reference_bundle = self._build_reference_bundle(
            scene_packet=scene_packet,
            same_constraints=same_constraints,
        )

        control_weights = self._build_control_weights(
            scene_packet=scene_packet,
            same_constraints=same_constraints,
            change_constraints=change_constraints,
        )

        return {
            "scene_id": scene_id,
            "text_prompt": scene_text,
            "scene_packet": scene_packet,
            "same_constraints": same_constraints,
            "change_constraints": change_constraints,
            "generation_contract": generation_contract,
            "reference_bundle": reference_bundle,
            "control_weights": control_weights,
        }

    def update_after_generation(
        self,
        scene_packet: Dict[str, Any],
        generation_result: Dict[str, Any],
        score_report: Dict[str, Any] = None,
        drift_report: Dict[str, Any] = None,
        prompt_bundle: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Update memory and reference bank after accepted generation.

        This should ideally be called only for accepted scenes.
        """
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)
        score_report = self._safe_dict(score_report)
        drift_report = self._safe_dict(drift_report)
        prompt_bundle = self._safe_dict(prompt_bundle)

        scene_id = self._scene_id(scene_packet)

        generation_result = self._ensure_generated_summary(
            generation_result=generation_result,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
        )

        memory_update = {}
        reference_update = {}

        if self.memory is not None:
            memory_update = self._update_memory(
                scene_packet=scene_packet,
                generation_result=generation_result,
                score_report=score_report,
                drift_report=drift_report,
                prompt_bundle=prompt_bundle,
            )

        if self.reference_bank is not None:
            reference_update = self._update_reference_bank(
                scene_packet=scene_packet,
                generation_result=generation_result,
                score_report=score_report,
                prompt_bundle=prompt_bundle,
            )

        selected_keyframe = self._select_keyframe_from_generation_result(generation_result)

        return {
            "ok": True,
            "scene_id": scene_id,
            "selected_keyframe": selected_keyframe,
            "memory_update": memory_update,
            "reference_update": reference_update,
            "generation_result": generation_result,
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "memory": self._get_memory_state(),
            "reference_bank": self._extract_reference_bank_state(),
        }

    # ------------------------------------------------------------------
    # prepare helpers
    # ------------------------------------------------------------------

    def _build_same_constraints(self, scene_packet: Dict[str, Any]) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)

        direct_same = self._safe_dict(scene_packet.get("same_constraints", {}))
        if direct_same:
            return copy.deepcopy(direct_same)

        same_as_previous = self._safe_dict(scene_packet.get("same_as_previous", {}))
        memory_state = self._get_memory_state()

        characters = []
        location = {}
        props = []
        style = {}

        if same_as_previous.get("character_identity", False):
            characters = copy.deepcopy(self._safe_list(memory_state.get("characters", [])))

        if same_as_previous.get("location", False):
            location = copy.deepcopy(self._safe_dict(memory_state.get("location", {})))

        if same_as_previous.get("props", False):
            props = copy.deepcopy(self._safe_list(memory_state.get("props", [])))

        if same_as_previous.get("style", False):
            style = copy.deepcopy(self._safe_dict(memory_state.get("style", {})))

        return {
            "characters": characters,
            "location": location,
            "props": props,
            "style": style,
        }

    def _build_change_constraints(self, scene_packet: Dict[str, Any]) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)

        direct_change = self._safe_dict(scene_packet.get("change_constraints", {}))
        if direct_change:
            return copy.deepcopy(direct_change)

        return {
            "target_actions": self._safe_list(scene_packet.get("target_actions", [])),
            "target_emotions": self._safe_list(scene_packet.get("target_emotions", [])),
            "target_poses": self._safe_list(scene_packet.get("target_poses", [])),
            "target_time_of_day": self._safe_text(scene_packet.get("target_time_of_day", "")),
            "target_weather": self._safe_text(scene_packet.get("target_weather", "")),
            "target_shot_type": self._safe_text(scene_packet.get("target_shot_type", "")),
            "target_camera_motion": self._safe_text(scene_packet.get("target_camera_motion", "")),
            "target_camera_angle": self._safe_text(scene_packet.get("target_camera_angle", "")),
            "pose_change": bool(scene_packet.get("pose_change", False)),
            "expression_change": bool(scene_packet.get("expression_change", False)),
            "camera_angle_change": bool(scene_packet.get("camera_angle_change", False)),
            "camera_motion_change": bool(scene_packet.get("camera_motion_change", False)),
            "action_change": bool(scene_packet.get("action_change", False)),
            "location_change": bool(scene_packet.get("location_change", False)),
            "outfit_change": bool(scene_packet.get("outfit_change", False)),
            "time_shift": bool(scene_packet.get("time_shift", False)),
            "new_character_entry": bool(scene_packet.get("new_character_entry", False)),
        }

    def _build_generation_contract(
        self,
        scene_packet: Dict[str, Any],
        same_constraints: Dict[str, Any],
        change_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)
        same_as_previous = self._safe_dict(scene_packet.get("same_as_previous", {}))

        return {
            "same_as_previous": same_as_previous,
            "must_keep": {
                "characters": copy.deepcopy(self._safe_list(same_constraints.get("characters", []))),
                "location": copy.deepcopy(self._safe_dict(same_constraints.get("location", {}))),
                "props": copy.deepcopy(self._safe_list(same_constraints.get("props", []))),
                "style": copy.deepcopy(self._safe_dict(same_constraints.get("style", {}))),
            },
            "can_change": copy.deepcopy(change_constraints),
        }

    def _build_reference_bundle(
        self,
        scene_packet: Dict[str, Any],
        same_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build dependency-aware reference bundle.

        The key change:
        Instead of one generic keyframe, this now uses role-specific references:
        - transition_reference: last/transition frame
        - identity_reference: best character frame
        - location_reference: best background frame
        - prop_reference: best object frame
        - style_reference: style representative frame
        """
        scene_packet = self._safe_dict(scene_packet)

        direct_ref = self._safe_dict(scene_packet.get("reference_bundle", {}))
        if direct_ref:
            return copy.deepcopy(direct_ref)

        ref_bank_state = self._extract_reference_bank_state()
        scene_id = self._scene_id(scene_packet)
        same_as_previous = self._safe_dict(scene_packet.get("same_as_previous", {}))
        dependent = bool(scene_packet.get("dependent_on_previous", False))

        role_refs = self._safe_dict(ref_bank_state.get("role_refs", {}))

        transition_frame = self._first_existing([
            role_refs.get("transition_frame_path", ""),
            ref_bank_state.get("transition_frame_path", ""),
            ref_bank_state.get("selected_keyframe", ""),
            ref_bank_state.get("last_keyframe", ""),
        ])

        identity_frame = self._first_existing([
            role_refs.get("identity_frame_path", ""),
            ref_bank_state.get("identity_frame_path", ""),
            ref_bank_state.get("selected_keyframe", ""),
        ])

        location_frame = self._first_existing([
            role_refs.get("location_frame_path", ""),
            ref_bank_state.get("location_frame_path", ""),
            ref_bank_state.get("selected_keyframe", ""),
        ])

        prop_frame = self._first_existing([
            role_refs.get("prop_frame_path", ""),
            ref_bank_state.get("prop_frame_path", ""),
            ref_bank_state.get("selected_keyframe", ""),
        ])

        style_frame = self._first_existing([
            role_refs.get("style_frame_path", ""),
            ref_bank_state.get("style_frame_path", ""),
            ref_bank_state.get("selected_keyframe", ""),
        ])

        character_refs = []
        if same_as_previous.get("character_identity", False):
            if identity_frame:
                character_refs.append({
                    "path": identity_frame,
                    "role": "identity_reference",
                    "reason": "same character dependency",
                    "scene_id": ref_bank_state.get("last_scene_id", ""),
                })
            for item in self._safe_list(ref_bank_state.get("character_refs", [])):
                item = self._safe_dict(item)
                path = self._safe_text(item.get("path", ""))
                if path and self._path_exists(path):
                    character_refs.append(copy.deepcopy(item))

        location_ref = {}
        if same_as_previous.get("location", False) and location_frame:
            location_ref = {
                "path": location_frame,
                "role": "location_reference",
                "reason": "same location dependency",
                "scene_id": ref_bank_state.get("last_scene_id", ""),
            }

        prop_refs = []
        if same_as_previous.get("props", False):
            if prop_frame:
                prop_refs.append({
                    "path": prop_frame,
                    "role": "prop_reference",
                    "reason": "same prop dependency",
                    "scene_id": ref_bank_state.get("last_scene_id", ""),
                })
            for item in self._safe_list(ref_bank_state.get("prop_refs", [])):
                item = self._safe_dict(item)
                path = self._safe_text(item.get("path", ""))
                if path and self._path_exists(path):
                    prop_refs.append(copy.deepcopy(item))

        style_ref = {}
        if same_as_previous.get("style", False) and style_frame:
            style_ref = {
                "path": style_frame,
                "role": "style_reference",
                "reason": "same style dependency",
                "scene_id": ref_bank_state.get("last_scene_id", ""),
            }

        # Dependency-aware primary reference selection.
        primary_reference_path = ""
        primary_reference_type = ""

        if dependent and transition_frame:
            primary_reference_path = transition_frame
            primary_reference_type = "transition_reference"
        elif same_as_previous.get("character_identity", False) and identity_frame:
            primary_reference_path = identity_frame
            primary_reference_type = "identity_reference"
        elif same_as_previous.get("location", False) and location_frame:
            primary_reference_path = location_frame
            primary_reference_type = "location_reference"
        elif same_as_previous.get("props", False) and prop_frame:
            primary_reference_path = prop_frame
            primary_reference_type = "prop_reference"
        elif same_as_previous.get("style", False) and style_frame:
            primary_reference_path = style_frame
            primary_reference_type = "style_reference"

        secondary_references = []
        for p in [
            identity_frame,
            location_frame,
            prop_frame,
            transition_frame,
            style_frame,
        ]:
            if p and p != primary_reference_path:
                secondary_references.append(p)

        secondary_references = self._dedupe_keep_order(secondary_references)

        return {
            "scene_id": scene_id,

            # legacy-compatible fields
            "previous_scene_keyframe": transition_frame or primary_reference_path,
            "character_refs": character_refs,
            "location_ref": location_ref,
            "prop_refs": prop_refs,
            "primary_reference_path": primary_reference_path,
            "primary_reference_type": primary_reference_type,
            "secondary_references": secondary_references,

            # new role-specific fields
            "identity_frame_path": identity_frame,
            "location_frame_path": location_frame,
            "prop_frame_path": prop_frame,
            "transition_frame_path": transition_frame,
            "style_frame_path": style_frame,

            "style_ref": style_ref,

            "reference_decision": {
                "dependent_on_previous": dependent,
                "same_as_previous": same_as_previous,
                "primary_reference_type": primary_reference_type,
                "primary_reference_path": primary_reference_path,
                "used_role_refs": {
                    "identity": bool(identity_frame),
                    "location": bool(location_frame),
                    "prop": bool(prop_frame),
                    "transition": bool(transition_frame),
                    "style": bool(style_frame),
                },
            },
        }

    def _build_control_weights(
        self,
        scene_packet: Dict[str, Any],
        same_constraints: Dict[str, Any],
        change_constraints: Dict[str, Any],
    ) -> Dict[str, float]:
        scene_packet = self._safe_dict(scene_packet)

        direct_weights = self._safe_dict(scene_packet.get("control_weights", {}))
        if direct_weights:
            return self._normalized_control_weights(direct_weights)

        same_as_previous = self._safe_dict(scene_packet.get("same_as_previous", {}))

        weights = {
            "identity_strength": self._safe_float(self.control_defaults.get("identity_strength", 0.85), 0.85),
            "location_strength": self._safe_float(self.control_defaults.get("location_strength", 0.80), 0.80),
            "style_strength": self._safe_float(self.control_defaults.get("style_strength", 0.72), 0.72),
            "prop_strength": self._safe_float(self.control_defaults.get("prop_strength", 0.70), 0.70),
            "transition_strength": self._safe_float(self.control_defaults.get("transition_strength", 0.82), 0.82),
            "continuity_strength": self._safe_float(self.control_defaults.get("continuity_strength", 0.84), 0.84),
            "motion_strength": self._safe_float(self.control_defaults.get("motion_strength", 0.60), 0.60),
        }

        if not same_as_previous.get("character_identity", False):
            weights["identity_strength"] = min(weights["identity_strength"], 0.25)

        if not same_as_previous.get("location", False):
            weights["location_strength"] = min(weights["location_strength"], 0.25)

        if not same_as_previous.get("props", False):
            weights["prop_strength"] = min(weights["prop_strength"], 0.25)

        if not same_as_previous.get("style", False):
            weights["style_strength"] = min(weights["style_strength"], 0.30)

        if not bool(scene_packet.get("dependent_on_previous", False)):
            weights["transition_strength"] = min(weights["transition_strength"], 0.20)
            weights["continuity_strength"] = min(weights["continuity_strength"], 0.35)

        if change_constraints.get("location_change", False):
            weights["location_strength"] = min(weights["location_strength"], 0.20)

        if change_constraints.get("outfit_change", False):
            weights["identity_strength"] = max(weights["identity_strength"], 0.60)

        if change_constraints.get("camera_motion_change", False):
            weights["motion_strength"] = max(weights["motion_strength"], 0.70)

        return self._normalized_control_weights(weights)

    # ------------------------------------------------------------------
    # observed-state extraction
    # ------------------------------------------------------------------

    def _ensure_generated_summary(
        self,
        generation_result: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        scene_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        already_has = bool(self._safe_dict(metadata.get("generated_summary", {})))
        if already_has:
            return generation_result

        try:
            from src.continuity.extract import attach_generated_summary
            return attach_generated_summary(
                generation_result=generation_result,
                prompt_bundle=prompt_bundle,
                scene_packet=scene_packet,
                config=self.config,
            )
        except Exception:
            metadata.setdefault(
                "generated_summary",
                {
                    "characters": [],
                    "location": {},
                    "props": [],
                    "style": {},
                    "source": "missing",
                },
            )
            metadata.setdefault("semantic_evidence_status", "missing")
            metadata.setdefault("has_semantic_evidence", False)
            generation_result["metadata"] = metadata
            return generation_result

    def _build_observed_memory_payload(
        self,
        generation_result: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        scene_packet: Dict[str, Any],
        score_report: Dict[str, Any],
        drift_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        generation_result = self._ensure_generated_summary(
            generation_result=generation_result,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
        )

        metadata = self._safe_dict(generation_result.get("metadata", {}))
        generated_summary = self._safe_dict(metadata.get("generated_summary", {}))
        semantic_evidence_status = self._safe_text(
            metadata.get("semantic_evidence_status", "")
            or self._safe_dict(drift_report).get("semantic_evidence_status", "")
            or self._safe_dict(score_report).get("semantic_evidence_status", "")
            or "missing"
        ).lower()

        has_semantic_evidence = bool(
            metadata.get(
                "has_semantic_evidence",
                bool(
                    self._safe_list(generated_summary.get("characters", []))
                    or self._safe_dict(generated_summary.get("location", {}))
                    or self._safe_list(generated_summary.get("props", []))
                    or self._safe_dict(generated_summary.get("style", {}))
                ),
            )
        )

        role_refs = self._extract_role_keyframes_from_generation_result(generation_result)

        return {
            "characters": copy.deepcopy(self._safe_list(generated_summary.get("characters", []))),
            "location": copy.deepcopy(self._safe_dict(generated_summary.get("location", {}))),
            "props": copy.deepcopy(self._safe_list(generated_summary.get("props", []))),
            "style": copy.deepcopy(self._safe_dict(generated_summary.get("style", {}))),
            "source": self._safe_text(generated_summary.get("source", "missing")) or "missing",
            "semantic_evidence_status": semantic_evidence_status or "missing",
            "has_semantic_evidence": has_semantic_evidence,
            "selected_keyframe": self._select_keyframe_from_generation_result(generation_result),
            "role_refs": role_refs,
        }

    # ------------------------------------------------------------------
    # update helpers
    # ------------------------------------------------------------------

    def _update_memory(
        self,
        scene_packet: Dict[str, Any],
        generation_result: Dict[str, Any],
        score_report: Dict[str, Any],
        drift_report: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)
        score_report = self._safe_dict(score_report)
        drift_report = self._safe_dict(drift_report)
        prompt_bundle = self._safe_dict(prompt_bundle)

        prompt_contract = self._safe_dict(prompt_bundle.get("prompt_contract", {}))
        observed = self._build_observed_memory_payload(
            generation_result=generation_result,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
            score_report=score_report,
            drift_report=drift_report,
        )

        semantic_evidence_status = self._safe_text(observed.get("semantic_evidence_status", "missing")).lower()
        has_semantic_evidence = bool(observed.get("has_semantic_evidence", False))

        identity_lock = self._safe_dict(prompt_contract.get("identity_lock", {}))
        location_lock = self._safe_dict(prompt_contract.get("location_lock", {}))
        prop_lock = self._safe_dict(prompt_contract.get("prop_lock", {}))
        style_lock = self._safe_dict(prompt_contract.get("style_lock", {}))

        final_characters = copy.deepcopy(self._safe_list(observed.get("characters", [])))
        final_location = copy.deepcopy(self._safe_dict(observed.get("location", {})))
        final_props = copy.deepcopy(self._safe_list(observed.get("props", [])))
        final_style = copy.deepcopy(self._safe_dict(observed.get("style", {})))

        if semantic_evidence_status == "partial" and has_semantic_evidence:
            if not final_characters:
                for item in self._safe_list(identity_lock.get("characters", [])):
                    item = self._safe_dict(item)
                    name = self._safe_text(item.get("name", ""))
                    if name:
                        final_characters.append({"name_hint": name, "evidence": "prompt_fallback"})

            if not final_location:
                name = self._safe_text(location_lock.get("name", ""))
                anchors = self._safe_list(location_lock.get("anchors", []))
                if name or anchors:
                    final_location = {
                        "name_hint": name,
                        "anchors_hint": copy.deepcopy(anchors),
                        "evidence": "prompt_fallback",
                    }

            if not final_props:
                for item in self._safe_list(prop_lock.get("props", [])):
                    item = self._safe_dict(item)
                    name = self._safe_text(item.get("name", ""))
                    if name:
                        final_props.append({"name_hint": name, "evidence": "prompt_fallback"})

            if not final_style:
                for field in ["visual_style", "color_tone", "shot_type", "camera_angle", "camera_motion", "mood"]:
                    value = self._safe_text(style_lock.get(field, ""))
                    if value:
                        final_style[f"{field}_hint"] = value

        role_refs = self._safe_dict(observed.get("role_refs", {}))

        memory_payload = {
            "scene_id": self._scene_id(scene_packet),
            "last_scene_id": self._scene_id(scene_packet),

            "characters": final_characters if semantic_evidence_status != "missing" else [],
            "location": final_location if semantic_evidence_status != "missing" else {},
            "props": final_props if semantic_evidence_status != "missing" else [],
            "style": final_style if semantic_evidence_status != "missing" else {},

            "selected_keyframe": self._safe_text(observed.get("selected_keyframe", "")),
            "semantic_evidence_status": semantic_evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "source": self._safe_text(observed.get("source", "missing")) or "missing",

            # important: role refs are visual assets, so store them even if semantic text evidence is weak
            "role_refs": role_refs,
            "identity_frame_path": self._safe_text(role_refs.get("identity_frame_path", "")),
            "location_frame_path": self._safe_text(role_refs.get("location_frame_path", "")),
            "prop_frame_path": self._safe_text(role_refs.get("prop_frame_path", "")),
            "transition_frame_path": self._safe_text(role_refs.get("transition_frame_path", "")),
            "style_frame_path": self._safe_text(role_refs.get("style_frame_path", "")),
        }

        if hasattr(self.memory, "update"):
            try:
                self.memory.update(memory_payload)
                return {"ok": True, "payload": memory_payload}
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"memory.update failed: {e}",
                    "payload": memory_payload,
                }

        if isinstance(self.memory, dict):
            if semantic_evidence_status != "missing":
                self.memory["characters"] = copy.deepcopy(memory_payload["characters"])
                self.memory["location"] = copy.deepcopy(memory_payload["location"])
                self.memory["props"] = copy.deepcopy(memory_payload["props"])
                self.memory["style"] = copy.deepcopy(memory_payload["style"])

            self.memory["selected_keyframe"] = memory_payload["selected_keyframe"]
            self.memory["last_keyframe"] = memory_payload["selected_keyframe"]
            self.memory["last_scene_id"] = memory_payload["scene_id"]
            self.memory["semantic_evidence_status"] = semantic_evidence_status
            self.memory["has_semantic_evidence"] = has_semantic_evidence
            self.memory["source"] = memory_payload["source"]

            self.memory["role_refs"] = copy.deepcopy(role_refs)
            self.memory["identity_frame_path"] = memory_payload["identity_frame_path"]
            self.memory["location_frame_path"] = memory_payload["location_frame_path"]
            self.memory["prop_frame_path"] = memory_payload["prop_frame_path"]
            self.memory["transition_frame_path"] = memory_payload["transition_frame_path"]
            self.memory["style_frame_path"] = memory_payload["style_frame_path"]
            self.memory["last_update"] = memory_payload

            return {"ok": True, "payload": memory_payload}

        return {
            "ok": False,
            "error": "memory object does not support update",
            "payload": memory_payload,
        }

    def _update_reference_bank(
        self,
        scene_packet: Dict[str, Any],
        generation_result: Dict[str, Any],
        score_report: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_packet = self._safe_dict(scene_packet)
        generation_result = self._safe_dict(generation_result)
        score_report = self._safe_dict(score_report)
        prompt_bundle = self._safe_dict(prompt_bundle)

        generation_result = self._ensure_generated_summary(
            generation_result=generation_result,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
        )

        metadata = self._safe_dict(generation_result.get("metadata", {}))
        generated_summary = self._safe_dict(metadata.get("generated_summary", {}))
        prompt_contract = self._safe_dict(prompt_bundle.get("prompt_contract", {}))

        selected_keyframe = self._select_keyframe_from_generation_result(generation_result)
        role_refs = self._extract_role_keyframes_from_generation_result(generation_result)
        score = self._extract_score(score_report)

        identity_frame = self._safe_text(role_refs.get("identity_frame_path", "")) or selected_keyframe
        location_frame = self._safe_text(role_refs.get("location_frame_path", "")) or selected_keyframe
        prop_frame = self._safe_text(role_refs.get("prop_frame_path", "")) or selected_keyframe
        transition_frame = self._safe_text(role_refs.get("transition_frame_path", "")) or selected_keyframe
        style_frame = self._safe_text(role_refs.get("style_frame_path", "")) or selected_keyframe

        character_refs = []
        observed_characters = self._safe_list(generated_summary.get("characters", []))
        if observed_characters:
            for ch in observed_characters:
                ch = self._safe_dict(ch)
                name = self._safe_text(ch.get("name", "")) or self._safe_text(ch.get("name_hint", ""))
                if name and identity_frame:
                    character_refs.append({
                        "name": name,
                        "path": identity_frame,
                        "score": score,
                        "reason": "role_identity_frame_observed",
                        "scene_id": self._scene_id(scene_packet),
                    })
        else:
            for ch in self._safe_list(self._safe_dict(prompt_contract.get("identity_lock", {})).get("characters", [])):
                ch = self._safe_dict(ch)
                name = self._safe_text(ch.get("name", ""))
                if name and identity_frame:
                    character_refs.append({
                        "name": name,
                        "path": identity_frame,
                        "score": score,
                        "reason": "role_identity_frame_prompt_fallback",
                        "scene_id": self._scene_id(scene_packet),
                    })

        location_ref = {}
        observed_loc = self._safe_dict(generated_summary.get("location", {}))
        if location_frame:
            loc_name = (
                self._safe_text(observed_loc.get("name", ""))
                or self._safe_text(observed_loc.get("name_hint", ""))
            )
            if not loc_name:
                loc = self._safe_dict(prompt_contract.get("location_lock", {}))
                loc_name = self._safe_text(loc.get("name", ""))
            location_ref = {
                "name": loc_name,
                "path": location_frame,
                "score": score,
                "reason": "role_location_frame",
                "scene_id": self._scene_id(scene_packet),
            }

        prop_refs = []
        observed_props = self._safe_list(generated_summary.get("props", []))
        if observed_props:
            for prop in observed_props:
                prop = self._safe_dict(prop)
                name = self._safe_text(prop.get("name", "")) or self._safe_text(prop.get("name_hint", ""))
                if name and prop_frame:
                    prop_refs.append({
                        "name": name,
                        "path": prop_frame,
                        "score": score,
                        "reason": "role_prop_frame_observed",
                        "scene_id": self._scene_id(scene_packet),
                    })
        else:
            for prop in self._safe_list(self._safe_dict(prompt_contract.get("prop_lock", {})).get("props", [])):
                prop = self._safe_dict(prop)
                name = self._safe_text(prop.get("name", ""))
                if name and prop_frame:
                    prop_refs.append({
                        "name": name,
                        "path": prop_frame,
                        "score": score,
                        "reason": "role_prop_frame_prompt_fallback",
                        "scene_id": self._scene_id(scene_packet),
                    })

        update_payload = {
            "scene_id": self._scene_id(scene_packet),
            "last_scene_id": self._scene_id(scene_packet),
            "selected_keyframe": selected_keyframe,
            "last_keyframe": transition_frame or selected_keyframe,

            "role_refs": role_refs,
            "identity_frame_path": identity_frame,
            "location_frame_path": location_frame,
            "prop_frame_path": prop_frame,
            "transition_frame_path": transition_frame,
            "style_frame_path": style_frame,

            "character_refs": character_refs,
            "location_ref": location_ref,
            "prop_refs": prop_refs,

            "semantic_evidence_status": self._safe_text(metadata.get("semantic_evidence_status", "missing")) or "missing",
        }

        if hasattr(self.reference_bank, "update"):
            try:
                self.reference_bank.update(update_payload)
                return {"ok": True, "payload": update_payload}
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"reference_bank.update failed: {e}",
                    "payload": update_payload,
                }

        if isinstance(self.reference_bank, dict):
            self.reference_bank["last_update"] = update_payload
            self.reference_bank["last_scene_id"] = update_payload["last_scene_id"]
            self.reference_bank["selected_keyframe"] = selected_keyframe
            self.reference_bank["last_keyframe"] = update_payload["last_keyframe"]

            self.reference_bank["role_refs"] = copy.deepcopy(role_refs)
            self.reference_bank["identity_frame_path"] = identity_frame
            self.reference_bank["location_frame_path"] = location_frame
            self.reference_bank["prop_frame_path"] = prop_frame
            self.reference_bank["transition_frame_path"] = transition_frame
            self.reference_bank["style_frame_path"] = style_frame

            self.reference_bank["character_refs"] = character_refs
            self.reference_bank["location_ref"] = location_ref
            self.reference_bank["prop_refs"] = prop_refs

            return {"ok": True, "payload": update_payload}

        return {
            "ok": False,
            "error": "reference_bank object does not support update",
            "payload": update_payload,
        }

    # ------------------------------------------------------------------
    # role keyframe helpers
    # ------------------------------------------------------------------

    def _extract_role_keyframes_from_generation_result(self, generation_result: Dict[str, Any]) -> Dict[str, str]:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        keys = [
            "identity_frame_path",
            "location_frame_path",
            "prop_frame_path",
            "transition_frame_path",
            "style_frame_path",
            "first_frame_path",
            "middle_frame_path",
            "last_frame_path",
            "best_keyframe_path",
        ]

        out = {}
        for key in keys:
            val = (
                self._safe_text(generation_result.get(key, ""))
                or self._safe_text(metadata.get(key, ""))
            )
            out[key] = val

        return out

    # ------------------------------------------------------------------
    # state / builder helpers
    # ------------------------------------------------------------------

    def _get_memory_state(self) -> Dict[str, Any]:
        if self.memory is None:
            return {}

        if hasattr(self.memory, "to_dict"):
            try:
                return self._safe_dict(self.memory.to_dict())
            except Exception:
                return {}

        if isinstance(self.memory, dict):
            return copy.deepcopy(self.memory)

        if hasattr(self.memory, "__dict__"):
            try:
                return copy.deepcopy(vars(self.memory))
            except Exception:
                return {}

        return {}

    def _extract_reference_bank_state(self) -> Dict[str, Any]:
        if self.reference_bank is None:
            return {}

        if hasattr(self.reference_bank, "to_dict"):
            try:
                return self._safe_dict(self.reference_bank.to_dict())
            except Exception:
                return {}

        if isinstance(self.reference_bank, dict):
            return copy.deepcopy(self.reference_bank)

        if hasattr(self.reference_bank, "__dict__"):
            try:
                return copy.deepcopy(vars(self.reference_bank))
            except Exception:
                return {}

        return {}

    def _build_memory(self, config: Dict[str, Any]):
        try:
            from src.continuity.memory import ContinuityMemory  # type: ignore
            return ContinuityMemory(config)
        except Exception:
            pass

        try:
            from src.continuity.memory import Memory  # type: ignore
            return Memory(config)
        except Exception:
            pass

        return {
            "characters": [],
            "location": {},
            "props": [],
            "style": {},
            "role_refs": {},
        }

    def _build_reference_bank(self, config: Dict[str, Any]):
        try:
            from src.continuity.reference_bank import ReferenceBank  # type: ignore
            return ReferenceBank(config)
        except Exception:
            pass

        return {
            "character_refs": [],
            "location_ref": {},
            "prop_refs": [],
            "role_refs": {},
        }

    def _select_keyframe_from_generation_result(self, generation_result: Dict[str, Any]) -> str:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        candidates = [
            generation_result.get("best_keyframe_path", ""),
            generation_result.get("transition_frame_path", ""),
            generation_result.get("identity_frame_path", ""),
            generation_result.get("selected_keyframe_path", ""),
            generation_result.get("keyframe_path", ""),
            generation_result.get("middle_frame_path", ""),
            generation_result.get("last_frame_path", ""),
            generation_result.get("first_frame_path", ""),

            metadata.get("best_keyframe_path", ""),
            metadata.get("transition_frame_path", ""),
            metadata.get("identity_frame_path", ""),
            metadata.get("selected_keyframe_path", ""),
            metadata.get("keyframe_path", ""),
            metadata.get("middle_frame_path", ""),
            metadata.get("last_frame_path", ""),
            metadata.get("first_frame_path", ""),
        ]

        return self._first_existing(candidates, allow_non_existing=True)

    def _extract_scene_text(self, scene_packet: Dict[str, Any]) -> str:
        scene_packet = self._safe_dict(scene_packet)

        for key in ["scene_text", "text", "prompt", "description"]:
            val = self._safe_text(scene_packet.get(key, ""))
            if val:
                return val

        return ""

    def _scene_id(self, scene_packet: Dict[str, Any]) -> str:
        scene_packet = self._safe_dict(scene_packet)
        sid = self._safe_text(scene_packet.get("scene_id", ""))
        if sid:
            return sid
        sid = self._safe_text(scene_packet.get("id", ""))
        if sid:
            return sid
        return "scene_unknown"

    def _extract_score(self, score_report: Dict[str, Any]) -> float:
        score_report = self._safe_dict(score_report)
        if "overall_score" in score_report:
            return self._safe_float(score_report.get("overall_score", 0.0), 0.0)
        if "score" in score_report:
            return self._safe_float(score_report.get("score", 0.0), 0.0)
        return 0.0

    def _normalized_control_weights(self, weights: Dict[str, Any]) -> Dict[str, float]:
        weights = self._safe_dict(weights)

        out = {
            "identity_strength": self._clamp01(self._safe_float(weights.get("identity_strength", weights.get("identity_weight", 0.0)), 0.0)),
            "location_strength": self._clamp01(self._safe_float(weights.get("location_strength", weights.get("location_weight", 0.0)), 0.0)),
            "style_strength": self._clamp01(self._safe_float(weights.get("style_strength", weights.get("style_weight", 0.0)), 0.0)),
            "prop_strength": self._clamp01(self._safe_float(weights.get("prop_strength", weights.get("props_weight", 0.0)), 0.0)),
            "transition_strength": self._clamp01(self._safe_float(weights.get("transition_strength", weights.get("transition_weight", 0.0)), 0.0)),
            "continuity_strength": self._clamp01(self._safe_float(weights.get("continuity_strength", 0.0), 0.0)),
            "motion_strength": self._clamp01(self._safe_float(weights.get("motion_strength", 0.0), 0.0)),
        }

        out["identity_weight"] = out["identity_strength"]
        out["location_weight"] = out["location_strength"]
        out["style_weight"] = out["style_strength"]
        out["props_weight"] = out["prop_strength"]
        out["transition_weight"] = out["transition_strength"]

        return out

    def _first_existing(self, candidates: List[Any], allow_non_existing: bool = False) -> str:
        first_non_empty = ""

        for item in candidates or []:
            path = self._safe_text(item)
            if not path:
                continue

            if not first_non_empty:
                first_non_empty = path

            if self._path_exists(path):
                return path

        if allow_non_existing:
            return first_non_empty

        return ""

    def _path_exists(self, path: str) -> bool:
        path = self._safe_text(path)
        return bool(path) and os.path.isfile(path)

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