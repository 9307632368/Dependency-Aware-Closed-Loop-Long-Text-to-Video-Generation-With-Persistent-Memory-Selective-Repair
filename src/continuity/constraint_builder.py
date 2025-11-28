from typing import Dict, Any, List


def _text_or_empty(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _build_character_keep_block(characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ch in characters:
        out.append({
            "char_id": ch.get("char_id", ""),
            "name": ch.get("name", ""),
            "aliases": ch.get("aliases", []),
            "face_desc": ch.get("face_desc", ""),
            "hair": ch.get("hair", ""),
            "clothing": ch.get("clothing", []),
            "accessories": ch.get("accessories", []),
            "pose": ch.get("pose", ""),
            "emotion": ch.get("emotion", ""),
            "action": ch.get("action", ""),
        })
    return out


def _build_location_keep_block(location: Dict[str, Any]) -> Dict[str, Any]:
    if not location:
        return {}

    return {
        "location_id": location.get("location_id", ""),
        "name": location.get("name", ""),
        "category": location.get("category", ""),
        "anchors": location.get("anchors", []),
        "lighting": location.get("lighting", ""),
        "weather": location.get("weather", ""),
        "time_of_day": location.get("time_of_day", ""),
        "atmosphere": location.get("atmosphere", ""),
    }


def _build_prop_keep_block(props: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in props:
        out.append({
            "prop_id": p.get("prop_id", p.get("name", "")),
            "name": p.get("name", ""),
            "holder": p.get("holder", ""),
            "status": p.get("status", ""),
            "attributes": p.get("attributes", []),
        })
    return out


def _build_style_keep_block(style: Dict[str, Any]) -> Dict[str, Any]:
    if not style:
        return {}

    return {
        "visual_style": style.get("visual_style", ""),
        "color_tone": style.get("color_tone", ""),
        "shot_type": style.get("shot_type", ""),
        "camera_angle": style.get("camera_angle", ""),
        "camera_motion": style.get("camera_motion", ""),
        "mood": style.get("mood", ""),
    }


def _score_reference_entry(ref: Dict[str, Any], bonus: float = 0.0) -> float:
    if not ref:
        return -1.0
    return _safe_float(ref.get("score", 0.0), 0.0) + bonus


def _pick_primary_reference(
    scene_packet: Dict[str, Any],
    same_constraints: Dict[str, Any],
    memory,
    reference_bank
) -> Dict[str, Any]:
    """
    Choose the strongest primary reference source for the current scene.

    Priority logic:
    1. Previous scene last/selected keyframe for strong transition continuity
    2. Best same-character reference when identity continuity matters
    3. Best same-location reference when environment continuity matters
    4. Best prop reference
    """
    scene_id = scene_packet.get("scene_id", "")
    same_flags = scene_packet.get("same_as_previous", {})
    continuity_signals = scene_packet.get("continuity_signals", {})
    priorities = scene_packet.get("continuity_priority", {})

    continuity_strength = _safe_float(continuity_signals.get("continuity_strength", 0.5), 0.5)
    identity_priority = _safe_float(priorities.get("identity", 1.0), 1.0)
    location_priority = _safe_float(priorities.get("location", 1.0), 1.0)
    props_priority = _safe_float(priorities.get("props", 1.0), 1.0)

    candidates = []

    # previous scene selected/last frame
    last_scene = memory.get_last_scene()
    if last_scene:
        prev_selected = last_scene.get("metadata", {}).get("selected_keyframe", "")
        prev_last = last_scene.get("metadata", {}).get("last_frame", "")

        if prev_last:
            candidates.append({
                "type": "previous_last_frame",
                "frame_path": prev_last,
                "score": 0.70 + 0.25 * continuity_strength,
                "scene_id": last_scene.get("scene_id", ""),
                "reason": "strong temporal continuity",
            })

        if prev_selected:
            candidates.append({
                "type": "previous_selected_keyframe",
                "frame_path": prev_selected,
                "score": 0.68 + 0.22 * continuity_strength,
                "scene_id": last_scene.get("scene_id", ""),
                "reason": "previous scene representative frame",
            })

    # character refs
    if same_flags.get("character_identity"):
        for ch in same_constraints.get("characters", []):
            char_id = ch.get("char_id", "")
            ref = reference_bank.get_character_reference(char_id)
            if ref and ref.get("frame_path"):
                candidates.append({
                    "type": "character_reference",
                    "frame_path": ref.get("frame_path", ""),
                    "score": _score_reference_entry(ref, bonus=0.20 * identity_priority),
                    "scene_id": ref.get("scene_id", ""),
                    "reason": f"character identity continuity:{char_id}",
                    "char_id": char_id,
                })

    # location refs
    if same_flags.get("location"):
        loc = same_constraints.get("location", {})
        loc_id = loc.get("location_id", "")
        ref = reference_bank.get_location_reference(loc_id)
        if ref and ref.get("frame_path"):
            candidates.append({
                "type": "location_reference",
                "frame_path": ref.get("frame_path", ""),
                "score": _score_reference_entry(ref, bonus=0.18 * location_priority),
                "scene_id": ref.get("scene_id", ""),
                "reason": f"location continuity:{loc_id}",
                "location_id": loc_id,
            })

    # prop refs
    for p in same_constraints.get("props", []):
        prop_name = p.get("name", "")
        ref = reference_bank.get_prop_reference(prop_name)
        if ref and ref.get("frame_path"):
            candidates.append({
                "type": "prop_reference",
                "frame_path": ref.get("frame_path", ""),
                "score": _score_reference_entry(ref, bonus=0.10 * props_priority),
                "scene_id": ref.get("scene_id", ""),
                "reason": f"prop continuity:{prop_name}",
                "prop_id": prop_name,
            })

    candidates = [c for c in candidates if c.get("frame_path")]
    if not candidates:
        return {}

    candidates.sort(key=lambda x: _safe_float(x.get("score", 0.0), 0.0), reverse=True)
    best = candidates[0]

    return {
        "scene_id": scene_id,
        "primary_reference_type": best.get("type", ""),
        "primary_reference_path": best.get("frame_path", ""),
        "primary_reference_score": best.get("score", 0.0),
        "primary_reference_scene_id": best.get("scene_id", ""),
        "primary_reference_reason": best.get("reason", ""),
    }


def _build_secondary_references(
    same_constraints: Dict[str, Any],
    reference_bank,
    primary_reference_path: str,
    limit: int = 4
) -> List[Dict[str, Any]]:
    refs = []

    # character refs
    for ch in same_constraints.get("characters", []):
        char_id = ch.get("char_id", "")
        ref = reference_bank.get_character_reference(char_id)
        if ref and ref.get("frame_path") and ref.get("frame_path") != primary_reference_path:
            refs.append({
                "type": "character_reference",
                "frame_path": ref.get("frame_path", ""),
                "score": ref.get("score", 0.0),
                "scene_id": ref.get("scene_id", ""),
                "char_id": char_id,
            })

    # location ref
    loc = same_constraints.get("location", {})
    loc_id = loc.get("location_id", "")
    if loc_id:
        ref = reference_bank.get_location_reference(loc_id)
        if ref and ref.get("frame_path") and ref.get("frame_path") != primary_reference_path:
            refs.append({
                "type": "location_reference",
                "frame_path": ref.get("frame_path", ""),
                "score": ref.get("score", 0.0),
                "scene_id": ref.get("scene_id", ""),
                "location_id": loc_id,
            })

    # prop refs
    for p in same_constraints.get("props", []):
        prop_name = p.get("name", "")
        ref = reference_bank.get_prop_reference(prop_name)
        if ref and ref.get("frame_path") and ref.get("frame_path") != primary_reference_path:
            refs.append({
                "type": "prop_reference",
                "frame_path": ref.get("frame_path", ""),
                "score": ref.get("score", 0.0),
                "scene_id": ref.get("scene_id", ""),
                "prop_id": prop_name,
            })

    unique = []
    seen = set()
    for r in refs:
        fp = r.get("frame_path", "")
        if fp and fp not in seen:
            seen.add(fp)
            unique.append(r)

    unique.sort(key=lambda x: _safe_float(x.get("score", 0.0), 0.0), reverse=True)
    return unique[:limit]


def _build_reference_bundle(scene_packet: Dict[str, Any], memory, reference_bank) -> Dict[str, Any]:
    same_constraints = memory.build_same_constraints(scene_packet)

    primary = _pick_primary_reference(scene_packet, same_constraints, memory, reference_bank)
    primary_path = primary.get("primary_reference_path", "")

    secondary = _build_secondary_references(
        same_constraints=same_constraints,
        reference_bank=reference_bank,
        primary_reference_path=primary_path,
        limit=4
    )

    bundle = {
        "primary_reference_type": primary.get("primary_reference_type", ""),
        "primary_reference_path": primary_path,
        "primary_reference_score": primary.get("primary_reference_score", 0.0),
        "primary_reference_scene_id": primary.get("primary_reference_scene_id", ""),
        "primary_reference_reason": primary.get("primary_reference_reason", ""),
        "secondary_references": secondary,
        "character_refs": [],
        "location_ref": None,
        "prop_refs": [],
        "previous_scene_keyframe": None,
    }

    # keep legacy compatibility
    for ch in same_constraints.get("characters", []):
        char_id = ch.get("char_id", "")
        ref = reference_bank.get_character_reference(char_id)
        if ref:
            bundle["character_refs"].append({
                "char_id": char_id,
                "frame_path": ref.get("frame_path", ""),
                "score": ref.get("score", 0.0),
                "scene_id": ref.get("scene_id", ""),
            })

    loc = same_constraints.get("location", {})
    loc_id = loc.get("location_id", "")
    if loc_id:
        loc_ref = reference_bank.get_location_reference(loc_id)
        if loc_ref:
            bundle["location_ref"] = {
                "location_id": loc_id,
                "frame_path": loc_ref.get("frame_path", ""),
                "score": loc_ref.get("score", 0.0),
                "scene_id": loc_ref.get("scene_id", ""),
            }

    for p in same_constraints.get("props", []):
        prop_name = p.get("name", "")
        if not prop_name:
            continue
        pref = reference_bank.get_prop_reference(prop_name)
        if pref:
            bundle["prop_refs"].append({
                "prop_id": prop_name,
                "frame_path": pref.get("frame_path", ""),
                "score": pref.get("score", 0.0),
                "scene_id": pref.get("scene_id", ""),
            })

    last_scene = memory.get_last_scene()
    previous_keyframe = last_scene.get("metadata", {}).get("selected_keyframe")
    if previous_keyframe:
        bundle["previous_scene_keyframe"] = previous_keyframe

    return bundle


def _build_control_weights(scene_packet: Dict[str, Any]) -> Dict[str, float]:
    priorities = scene_packet.get("continuity_priority", {})
    continuity_strength = scene_packet.get("continuity_signals", {}).get("continuity_strength", 0.5)

    identity_weight = priorities.get("identity", 1.0)
    outfit_weight = priorities.get("outfit", 0.95)
    location_weight = priorities.get("location", 0.90)
    props_weight = priorities.get("props", 0.80)
    style_weight = priorities.get("style", 0.75)
    motion_weight = priorities.get("motion", 0.60)

    if continuity_strength >= 0.75:
        identity_weight += 0.10
        outfit_weight += 0.06
        location_weight += 0.06
        style_weight += 0.04
    elif continuity_strength <= 0.25:
        motion_weight += 0.06

    return {
        "text_weight": 1.0,
        "identity_weight": min(identity_weight, 1.30),
        "outfit_weight": min(outfit_weight, 1.20),
        "location_weight": min(location_weight, 1.20),
        "props_weight": min(props_weight, 1.10),
        "style_weight": min(style_weight, 1.05),
        "motion_weight": min(motion_weight, 1.00),
        "continuity_strength": continuity_strength,
    }


def _build_scene_change_summary(scene_packet: Dict[str, Any], change_constraints: Dict[str, Any]) -> Dict[str, Any]:
    characters = scene_packet.get("entities", {}).get("characters", [])
    location = scene_packet.get("location", {})
    style = scene_packet.get("style", {})
    props = scene_packet.get("props", {}).get("items", [])

    return {
        "scene_action_targets": [c.get("action", "") for c in characters if c.get("action")],
        "scene_emotion_targets": [c.get("emotion", "") for c in characters if c.get("emotion")],
        "scene_pose_targets": [c.get("pose", "") for c in characters if c.get("pose")],
        "scene_props_present": props,
        "scene_location_time": location.get("time_of_day", ""),
        "scene_location_weather": location.get("weather", ""),
        "scene_shot_type": style.get("shot_type", ""),
        "scene_camera_motion": style.get("camera_motion", ""),
        "scene_camera_angle": style.get("camera_angle", ""),
        "change_constraints": change_constraints,
    }


def _build_generation_contract(
    scene_packet: Dict[str, Any],
    same_constraints: Dict[str, Any],
    change_constraints: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "must_keep": {
            "characters": _build_character_keep_block(same_constraints.get("characters", [])),
            "location": _build_location_keep_block(same_constraints.get("location", {})),
            "props": _build_prop_keep_block(same_constraints.get("props", [])),
            "style": _build_style_keep_block(same_constraints.get("style", {})),
        },
        "can_change": _build_scene_change_summary(scene_packet, change_constraints),
        "same_as_previous": scene_packet.get("same_as_previous", {}),
        "continuity_signals": scene_packet.get("continuity_signals", {}),
    }


def _build_text_prompt(scene_packet: Dict[str, Any]) -> str:
    return _text_or_empty(scene_packet.get("scene_text", ""))


def build_continuity_package(
    scene_packet: Dict[str, Any],
    memory,
    reference_bank
) -> Dict[str, Any]:
    same_constraints = memory.build_same_constraints(scene_packet)
    change_constraints = memory.build_change_constraints(scene_packet)
    reference_bundle = _build_reference_bundle(scene_packet, memory, reference_bank)
    control_weights = _build_control_weights(scene_packet)
    generation_contract = _build_generation_contract(scene_packet, same_constraints, change_constraints)

    return {
        "scene_id": scene_packet.get("scene_id", ""),
        "text_prompt": _build_text_prompt(scene_packet),
        "scene_packet": scene_packet,
        "same_constraints": same_constraints,
        "change_constraints": change_constraints,
        "generation_contract": generation_contract,
        "reference_bundle": reference_bundle,
        "control_weights": control_weights,
        "metadata": {
            "builder": "constraint_builder",
            "version": "consistency_v3"
        }
    }