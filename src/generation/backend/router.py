# src/generation/backend/router.py
from __future__ import annotations

from typing import Dict, Any, List


def choose_backend_route(
    prompt_bundle: Dict[str, Any],
    available_backends: List[str],
    repair_mode: bool = False,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Final routing policy:

    - repair -> svd
    - first scene -> cogvideox only
    - dependent continuation scenes -> svd
    - independent later scenes -> cogvideox
    """
    config = config or {}
    prompt_bundle = prompt_bundle or {}

    scene_packet = (prompt_bundle.get("scene_packet", {}) or {})
    prompt_metadata = (prompt_bundle.get("prompt_metadata", {}) or {})
    reference_bundle = (prompt_bundle.get("reference_bundle", {}) or {})
    control_weights = (prompt_bundle.get("control_weights", {}) or {})

    scene_id = str(prompt_bundle.get("scene_id", "") or scene_packet.get("scene_id", "")).lower()
    retry_index = int(prompt_metadata.get("retry_index", 0) or 0)

    first_scene = _is_first_scene(scene_id)
    dependent = _is_dependent_scene(scene_packet)
    has_references = _has_meaningful_references(reference_bundle)
    continuity_strength = float(control_weights.get("continuity_strength", 0.0) or 0.0)

    scores = {
        "svd": 0.0,
        "cogvideox": 0.0,
    }

    if repair_mode:
        scores["svd"] += 100.0
        scores["cogvideox"] += 0.0

    elif first_scene:
        # force scene 1 to be text-first cogvideox
        scores["cogvideox"] += 100.0
        scores["svd"] += 0.0

    elif dependent:
        scores["svd"] += 100.0
        scores["cogvideox"] += 5.0

    else:
        scores["cogvideox"] += 100.0
        scores["svd"] += 5.0

    if has_references and (not first_scene):
        scores["svd"] += 10.0

    if retry_index >= 1 and (not first_scene):
        scores["svd"] += 5.0

    if continuity_strength >= 0.65 and (not first_scene):
        scores["svd"] += 5.0

    primary = _argmax(scores, available_backends)
    fallback = _second_best(scores, primary, available_backends)

    reason = _build_reason(
        repair_mode=repair_mode,
        first_scene=first_scene,
        dependent=dependent,
        retry_index=retry_index,
        has_references=has_references,
        continuity_strength=continuity_strength,
        primary=primary,
    )

    return {
        "primary_backend": primary,
        "fallback_backend": fallback,
        "route_reason": reason,
        "route_scores": scores,
    }


def _is_first_scene(scene_id: str) -> bool:
    scene_id = (scene_id or "").lower()
    return scene_id.endswith("001") or scene_id in {"scene1", "scene_1", "1"}


def _is_dependent_scene(scene_packet: Dict[str, Any]) -> bool:
    scene_packet = scene_packet or {}

    if "dependent_on_previous" in scene_packet and scene_packet.get("dependent_on_previous") is not None:
        return bool(scene_packet.get("dependent_on_previous"))

    same_as_previous = scene_packet.get("same_as_previous", {}) or {}

    if bool(same_as_previous.get("character_identity", False)):
        return True
    if bool(same_as_previous.get("outfit", False)):
        return True
    if bool(same_as_previous.get("location", False)):
        return True
    if bool(same_as_previous.get("style", False)):
        return True

    props = same_as_previous.get("props", [])
    if isinstance(props, list) and len(props) > 0:
        return True

    return False


def _has_meaningful_references(reference_bundle: Dict[str, Any]) -> bool:
    reference_bundle = reference_bundle or {}

    if reference_bundle.get("primary_reference_path"):
        return True

    if reference_bundle.get("previous_scene_keyframe"):
        return True

    char_refs = reference_bundle.get("character_refs", [])
    if isinstance(char_refs, list) and len(char_refs) > 0:
        return True

    location_ref = reference_bundle.get("location_ref", {})
    if isinstance(location_ref, dict) and location_ref.get("path"):
        return True

    prop_refs = reference_bundle.get("prop_refs", [])
    if isinstance(prop_refs, list) and len(prop_refs) > 0:
        return True

    secondary = reference_bundle.get("secondary_references", [])
    if isinstance(secondary, list) and len(secondary) > 0:
        return True

    return False


def _argmax(scores: Dict[str, float], available: List[str]) -> str:
    best_name = ""
    best_score = float("-inf")

    for name, score in scores.items():
        if name not in available:
            continue
        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def _second_best(scores: Dict[str, float], primary: str, available: List[str]) -> str:
    candidates = []

    for name, score in scores.items():
        if name == primary:
            continue
        if name not in available:
            continue
        candidates.append((score, name))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    return candidates[0][1]


def _build_reason(
    repair_mode: bool,
    first_scene: bool,
    dependent: bool,
    retry_index: int,
    has_references: bool,
    continuity_strength: float,
    primary: str,
) -> str:
    reasons = []

    if repair_mode:
        reasons.append("repair_mode")
    if first_scene:
        reasons.append("first_scene_text_only")
    if dependent:
        reasons.append("dependent_scene")
    if retry_index >= 1:
        reasons.append("retry")
    if has_references:
        reasons.append("has_references")
    if continuity_strength >= 0.65:
        reasons.append("high_continuity")

    if not reasons:
        reasons.append("default")

    return f"{primary}|" + ",".join(reasons)