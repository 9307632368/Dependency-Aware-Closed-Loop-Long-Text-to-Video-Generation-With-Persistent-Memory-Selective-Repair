# src/generation/prompt_builder.py
from __future__ import annotations

from typing import Dict, Any, List


# ----------------------------------------------------------------------
# public api
# ----------------------------------------------------------------------

def build_prompt_bundle(continuity_package: Dict[str, Any]) -> Dict[str, Any]:
    continuity_package = _safe_dict(continuity_package)

    scene_id = _safe_text(continuity_package.get("scene_id", ""))
    scene_packet = _safe_dict(continuity_package.get("scene_packet", {}))
    scene_text = _safe_text(
        continuity_package.get("text_prompt", "")
        or scene_packet.get("scene_text", "")
        or scene_packet.get("text", "")
        or scene_packet.get("prompt", "")
    )

    same_constraints = _safe_dict(continuity_package.get("same_constraints", {}))
    change_constraints = _safe_dict(continuity_package.get("change_constraints", {}))
    generation_contract = _safe_dict(continuity_package.get("generation_contract", {}))
    reference_bundle = _safe_dict(continuity_package.get("reference_bundle", {}))
    control_weights = _safe_dict(continuity_package.get("control_weights", {}))
    generation_params = _safe_dict(continuity_package.get("generation_params", {}))

    prompt_contract = _build_prompt_contract(
        scene_id=scene_id,
        scene_packet=scene_packet,
        scene_text=scene_text,
        same_constraints=same_constraints,
        change_constraints=change_constraints,
        generation_contract=generation_contract,
    )

    positive_prompt = _build_compact_positive_prompt(
        scene_text=scene_text,
        prompt_contract=prompt_contract,
        control_weights=control_weights,
    )

    model_prompt = _build_model_prompt(
        scene_text=scene_text,
        prompt_contract=prompt_contract,
        control_weights=control_weights,
    )

    continuity_prompt = _build_continuity_prompt(prompt_contract)
    repair_prompt = _build_repair_prompt(prompt_contract)
    negative_prompt = _build_compact_negative_prompt(prompt_contract)

    prompt_metadata = {
        "scene_id": scene_id,
        "retry_index": 0,
        "repair_mode": False,
        "prompt_compaction_mode": True,
        "prompt_word_budget": 90,
        "negative_prompt_word_budget": 40,
        "continuity_strength": _safe_float(control_weights.get("continuity_strength", 0.0), 0.0),
        "dependent_on_previous": bool(scene_packet.get("dependent_on_previous", False)),
    }

    return {
        "scene_id": scene_id,
        "prompt": model_prompt,
        "analysis_prompt": continuity_prompt,
        "story_prompt": positive_prompt,
        "continuity_prompt": continuity_prompt,
        "model_prompt": model_prompt,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "repair_prompt": repair_prompt,
        "prompt_contract": prompt_contract,
        "prompt_metadata": prompt_metadata,
        "reference_bundle": reference_bundle,
        "control_weights": control_weights,
        "generation_params": generation_params,
        "scene_packet": scene_packet,
        "same_constraints": same_constraints,
        "change_constraints": change_constraints,
        "generation_contract": generation_contract,
        "retry_context": {
            "retry_index": 0,
            "failure_tags": [],
            "drift_report": {},
        },
    }


def build_repair_prompt_bundle(
    continuity_package: Dict[str, Any],
    drift_report: Dict[str, Any],
    retry_index: int,
    failure_tags: List[str],
) -> Dict[str, Any]:
    base = build_prompt_bundle(continuity_package)

    drift_report = _safe_dict(drift_report)
    failure_tags = _dedupe_keep_order([_safe_text(x) for x in (failure_tags or []) if _safe_text(x)])

    compact_repair_lines = _build_compact_repair_lines(
        prompt_contract=_safe_dict(base.get("prompt_contract", {})),
        failure_tags=failure_tags,
    )

    base["positive_prompt"] = _append_sentences(
        _safe_text(base.get("positive_prompt", "")),
        compact_repair_lines,
        max_words=95,
    )
    base["model_prompt"] = _append_sentences(
        _safe_text(base.get("model_prompt", "")),
        compact_repair_lines,
        max_words=95,
    )
    base["continuity_prompt"] = _append_sentences(
        _safe_text(base.get("continuity_prompt", "")),
        compact_repair_lines[:2],
        max_words=70,
    )
    base["repair_prompt"] = _append_sentences(
        _safe_text(base.get("repair_prompt", "")),
        compact_repair_lines,
        max_words=95,
    )

    neg_terms = _build_repair_negative_terms(failure_tags)
    base["negative_prompt"] = _append_csv_terms(
        _safe_text(base.get("negative_prompt", "")),
        neg_terms,
        max_terms=22,
    )

    prompt_metadata = _safe_dict(base.get("prompt_metadata", {}))
    prompt_metadata["retry_index"] = int(retry_index or 0)
    prompt_metadata["repair_mode"] = True
    prompt_metadata["failure_tags"] = failure_tags
    prompt_metadata["drift_severity"] = _safe_text(drift_report.get("severity", ""))
    base["prompt_metadata"] = prompt_metadata

    retry_context = _safe_dict(base.get("retry_context", {}))
    retry_context["retry_index"] = int(retry_index or 0)
    retry_context["failure_tags"] = failure_tags
    retry_context["drift_report"] = drift_report
    base["retry_context"] = retry_context

    base["prompt"] = base["model_prompt"]
    return base


# ----------------------------------------------------------------------
# contract building
# ----------------------------------------------------------------------

def _build_prompt_contract(
    scene_id: str,
    scene_packet: Dict[str, Any],
    scene_text: str,
    same_constraints: Dict[str, Any],
    change_constraints: Dict[str, Any],
    generation_contract: Dict[str, Any],
) -> Dict[str, Any]:
    same_as_previous = _safe_dict(scene_packet.get("same_as_previous", {}))
    must_keep = _safe_dict(generation_contract.get("must_keep", {}))
    can_change = _safe_dict(generation_contract.get("can_change", {}))

    identity_lock = {
        "characters": _safe_list(must_keep.get("characters", same_constraints.get("characters", []))),
        "locked_names": _extract_names(_safe_list(must_keep.get("characters", same_constraints.get("characters", [])))),
    }

    location_lock = _safe_dict(must_keep.get("location", same_constraints.get("location", {})))
    prop_lock = {
        "props": _safe_list(must_keep.get("props", same_constraints.get("props", []))),
        "names": _extract_names(_safe_list(must_keep.get("props", same_constraints.get("props", [])))),
    }
    style_lock = _safe_dict(must_keep.get("style", same_constraints.get("style", {})))

    return {
        "scene_id": scene_id,
        "scene_text": scene_text,
        "same_as_previous": same_as_previous,
        "identity_lock": identity_lock,
        "location_lock": location_lock,
        "prop_lock": prop_lock,
        "style_lock": style_lock,
        "allowed_changes": can_change if can_change else copy_like(change_constraints),
        "story_core": {
            "dependent_on_previous": bool(scene_packet.get("dependent_on_previous", False)),
        },
    }


# ----------------------------------------------------------------------
# compact prompt building
# ----------------------------------------------------------------------

def _build_compact_positive_prompt(
    scene_text: str,
    prompt_contract: Dict[str, Any],
    control_weights: Dict[str, Any],
) -> str:
    chunks: List[str] = []

    core = _compress_sentence(scene_text)
    if core:
        chunks.append(core)

    keep_chunks = _build_priority_keep_chunks(prompt_contract, control_weights)
    change_chunks = _build_priority_change_chunks(prompt_contract)

    chunks.extend(keep_chunks[:4])
    chunks.extend(change_chunks[:2])

    text = ". ".join([c for c in chunks if c]).strip()
    return _truncate_words(text, 90)


def _build_model_prompt(
    scene_text: str,
    prompt_contract: Dict[str, Any],
    control_weights: Dict[str, Any],
) -> str:
    chunks: List[str] = []

    core = _compress_sentence(scene_text)
    if core:
        chunks.append(core)

    same_as_previous = _safe_dict(prompt_contract.get("same_as_previous", {}))

    if bool(same_as_previous.get("character_identity", False)):
        chunks.append(_build_identity_chunk(prompt_contract))
    if bool(same_as_previous.get("location", False)):
        chunks.append(_build_location_chunk(prompt_contract))
    if bool(same_as_previous.get("props", False)):
        chunks.append(_build_prop_chunk(prompt_contract))
    if bool(same_as_previous.get("style", False)):
        chunks.append(_build_style_chunk(prompt_contract))

    chunks.extend(_build_priority_change_chunks(prompt_contract)[:2])

    text = ". ".join([c for c in chunks if c]).strip()
    return _truncate_words(text, 90)


def _build_continuity_prompt(prompt_contract: Dict[str, Any]) -> str:
    same_as_previous = _safe_dict(prompt_contract.get("same_as_previous", {}))
    chunks = []

    if bool(same_as_previous.get("character_identity", False)):
        chunks.append(_build_identity_chunk(prompt_contract))
    if bool(same_as_previous.get("location", False)):
        chunks.append(_build_location_chunk(prompt_contract))
    if bool(same_as_previous.get("props", False)):
        chunks.append(_build_prop_chunk(prompt_contract))
    if bool(same_as_previous.get("style", False)):
        chunks.append(_build_style_chunk(prompt_contract))

    return _truncate_words(". ".join([c for c in chunks if c]), 65)


def _build_repair_prompt(prompt_contract: Dict[str, Any]) -> str:
    chunks = ["Fix continuity errors while preserving the intended scene"]
    chunks.extend(_build_priority_keep_chunks(prompt_contract, {}))
    return _truncate_words(". ".join([c for c in chunks if c]), 85)


def _build_compact_negative_prompt(prompt_contract: Dict[str, Any]) -> str:
    terms = [
        "blurry",
        "distorted face",
        "extra limbs",
        "duplicate person",
        "broken anatomy",
        "low detail",
        "text watermark",
        "random background",
        "wrong character",
        "wrong outfit",
        "wrong location",
        "missing prop",
        "style mismatch",
    ]
    return ", ".join(terms[:18])


# ----------------------------------------------------------------------
# repair prompt compaction
# ----------------------------------------------------------------------

def _build_compact_repair_lines(
    prompt_contract: Dict[str, Any],
    failure_tags: List[str],
) -> List[str]:
    lines: List[str] = []

    if "identity_drift" in failure_tags:
        lines.append(_build_identity_chunk(prompt_contract))

    if "location_drift" in failure_tags:
        lines.append(_build_location_chunk(prompt_contract))

    if "prop_loss" in failure_tags or "props_drift" in failure_tags:
        lines.append(_build_prop_chunk(prompt_contract))

    if "style_drift" in failure_tags:
        lines.append(_build_style_chunk(prompt_contract))

    if "transition_drift" in failure_tags or "transition_break" in failure_tags:
        lines.append("Make the opening feel like a direct continuation of the previous scene")

    if not lines:
        lines.append("Reduce drift and preserve continuity")

    return [_truncate_words(x, 18) for x in _dedupe_keep_order(lines)]


def _build_repair_negative_terms(failure_tags: List[str]) -> List[str]:
    terms = ["random image", "placeholder output", "wrong scene"]

    if "identity_drift" in failure_tags:
        terms.extend(["wrong character", "changed face", "changed hairstyle"])

    if "location_drift" in failure_tags:
        terms.extend(["wrong background", "different location"])

    if "prop_loss" in failure_tags or "props_drift" in failure_tags:
        terms.extend(["missing prop", "wrong object"])

    if "style_drift" in failure_tags:
        terms.extend(["style mismatch", "wrong cinematic tone"])

    if "transition_drift" in failure_tags or "transition_break" in failure_tags:
        terms.extend(["hard reset", "abrupt transition"])

    return _dedupe_keep_order(terms)


# ----------------------------------------------------------------------
# chunk builders
# ----------------------------------------------------------------------

def _build_priority_keep_chunks(prompt_contract: Dict[str, Any], control_weights: Dict[str, Any]) -> List[str]:
    weights = {
        "identity": _safe_float(_safe_dict(control_weights).get("identity_strength", 0.0), 0.0),
        "location": _safe_float(_safe_dict(control_weights).get("location_strength", 0.0), 0.0),
        "props": _safe_float(_safe_dict(control_weights).get("prop_strength", 0.0), 0.0),
        "style": _safe_float(_safe_dict(control_weights).get("style_strength", 0.0), 0.0),
    }

    candidates = [
        (weights["identity"], _build_identity_chunk(prompt_contract)),
        (weights["location"], _build_location_chunk(prompt_contract)),
        (weights["props"], _build_prop_chunk(prompt_contract)),
        (weights["style"], _build_style_chunk(prompt_contract)),
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates if c]


def _build_priority_change_chunks(prompt_contract: Dict[str, Any]) -> List[str]:
    allowed = _safe_dict(prompt_contract.get("allowed_changes", {}))
    chunks = []

    action = _first_nonempty_list_item(allowed.get("target_actions", []))
    emotion = _first_nonempty_list_item(allowed.get("target_emotions", []))
    shot = _safe_text(allowed.get("target_shot_type", ""))
    camera = _safe_text(allowed.get("target_camera_motion", ""))

    if action:
        chunks.append(f"Action: {action}")
    if emotion:
        chunks.append(f"Emotion: {emotion}")
    if shot:
        chunks.append(f"Shot: {shot}")
    if camera:
        chunks.append(f"Camera: {camera}")

    return chunks


def _build_identity_chunk(prompt_contract: Dict[str, Any]) -> str:
    identity_lock = _safe_dict(prompt_contract.get("identity_lock", {}))
    chars = _safe_list(identity_lock.get("characters", []))
    if not chars:
        return ""

    ch = _safe_dict(chars[0])
    parts = []

    name = _safe_text(ch.get("name", ""))
    if name:
        parts.append(name)

    hair = _safe_text(ch.get("hair", ""))
    if hair:
        parts.append(hair)

    clothing = _first_nonempty_list_item(ch.get("clothing", []))
    if clothing:
        parts.append(clothing)

    accessories = _first_nonempty_list_item(ch.get("accessories", []))
    if accessories:
        parts.append(accessories)

    if not parts:
        return "Keep the same character identity"

    return "Keep same character: " + ", ".join(parts[:4])


def _build_location_chunk(prompt_contract: Dict[str, Any]) -> str:
    loc = _safe_dict(prompt_contract.get("location_lock", {}))
    parts = []

    name = _safe_text(loc.get("name", ""))
    if name:
        parts.append(name)

    anchors = _safe_list(loc.get("anchors", []))
    first_anchor = _first_nonempty_list_item(anchors)
    if first_anchor:
        parts.append(first_anchor)

    lighting = _safe_text(loc.get("lighting", ""))
    if lighting:
        parts.append(lighting)

    if not parts:
        return "Keep the same location"

    return "Keep same location: " + ", ".join(parts[:3])


def _build_prop_chunk(prompt_contract: Dict[str, Any]) -> str:
    prop_lock = _safe_dict(prompt_contract.get("prop_lock", {}))
    props = _safe_list(prop_lock.get("props", []))
    if not props:
        return ""

    names = []
    for p in props[:3]:
        p = _safe_dict(p)
        name = _safe_text(p.get("name", ""))
        if name:
            names.append(name)

    if not names:
        return "Keep the same important props"

    return "Keep props: " + ", ".join(names)


def _build_style_chunk(prompt_contract: Dict[str, Any]) -> str:
    style = _safe_dict(prompt_contract.get("style", prompt_contract.get("style_lock", {})))
    parts = []

    visual_style = _safe_text(style.get("visual_style", ""))
    color_tone = _safe_text(style.get("color_tone", ""))
    mood = _safe_text(style.get("mood", ""))
    shot = _safe_text(style.get("shot_type", ""))

    if visual_style:
        parts.append(visual_style)
    if color_tone:
        parts.append(color_tone)
    if mood:
        parts.append(mood)
    if shot:
        parts.append(shot)

    if not parts:
        return ""

    return "Keep style: " + ", ".join(parts[:3])


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _append_sentences(base: str, lines: List[str], max_words: int) -> str:
    parts = []
    if _safe_text(base):
        parts.append(_safe_text(base))
    for line in lines or []:
        line = _safe_text(line)
        if line:
            parts.append(line)
    return _truncate_words(". ".join(parts), max_words)


def _append_csv_terms(base: str, terms: List[str], max_terms: int) -> str:
    items = []
    if _safe_text(base):
        items.extend([x.strip() for x in _safe_text(base).split(",") if x.strip()])
    items.extend([_safe_text(x) for x in (terms or []) if _safe_text(x)])
    items = _dedupe_keep_order(items)
    return ", ".join(items[:max_terms])


def _compress_sentence(text: str) -> str:
    text = _safe_text(text)
    if not text:
        return ""
    return _truncate_words(" ".join(text.split()), 40)


def _truncate_words(text: str, max_words: int) -> str:
    words = _safe_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _extract_names(items: List[Any]) -> List[str]:
    names = []
    for item in items or []:
        if isinstance(item, dict):
            name = _safe_text(item.get("name", ""))
            if name:
                names.append(name)
        else:
            val = _safe_text(item)
            if val:
                names.append(val)
    return _dedupe_keep_order(names)


def _first_nonempty_list_item(values: Any) -> str:
    vals = _safe_list(values)
    for v in vals:
        t = _safe_text(v)
        if t:
            return t
    return ""


def copy_like(x: Any):
    if isinstance(x, dict):
        return {k: copy_like(v) for k, v in x.items()}
    if isinstance(x, list):
        return [copy_like(v) for v in x]
    return x


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items or []:
        text = _safe_text(item)
        if not text:
            continue
        low = text.lower()
        if low not in seen:
            seen.add(low)
            out.append(text)
    return out


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _safe_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {}


def _safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return []


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default