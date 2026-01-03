# src/generation/backend/common.py
from __future__ import annotations

import copy
import os
from typing import Dict, Any, List, Tuple, Optional


# ---------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------

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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _to_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        val = x.strip().lower()
        if val in {"true", "1", "yes", "y"}:
            return True
        if val in {"false", "0", "no", "n"}:
            return False
    return default


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items or []:
        text = _safe_text(item)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)

    return out


def ensure_dir(path: str) -> str:
    path = _safe_text(path)
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _file_exists(path: str) -> bool:
    path = _safe_text(path)
    return bool(path) and os.path.isfile(path)


def _first_existing_path(paths: List[str]) -> str:
    for p in paths or []:
        p = _safe_text(p)
        if _file_exists(p):
            return p
    return ""


def _append_text(base: str, extra: str) -> str:
    base = _safe_text(base)
    extra = _safe_text(extra)
    if not base:
        return extra
    if not extra:
        return base
    return (base + " " + extra).strip()


def _merge_csv_text(base: str, extras: List[str]) -> str:
    base_items = [x.strip() for x in _safe_text(base).split(",") if x.strip()]
    extra_items = [_safe_text(x) for x in extras if _safe_text(x)]
    return ", ".join(_dedupe_keep_order(base_items + extra_items)).strip()


# ---------------------------------------------------------------------
# prompt selection
# ---------------------------------------------------------------------

def get_prompt_variants(prompt_bundle: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize the multi-prompt bundle into explicit prompt roles.

    The redesigned prompt builder now provides:
    - analysis_prompt
    - story_prompt
    - continuity_prompt
    - model_prompt
    - positive_prompt
    - negative_prompt
    - repair_prompt

    This helper keeps backends simple and backward-compatible.
    """
    prompt_bundle = _safe_dict(prompt_bundle)

    analysis_prompt = _safe_text(prompt_bundle.get("analysis_prompt", ""))
    story_prompt = _safe_text(prompt_bundle.get("story_prompt", ""))
    continuity_prompt = _safe_text(prompt_bundle.get("continuity_prompt", ""))
    model_prompt = _safe_text(prompt_bundle.get("model_prompt", ""))
    positive_prompt = _safe_text(prompt_bundle.get("positive_prompt", ""))
    negative_prompt = _safe_text(prompt_bundle.get("negative_prompt", ""))
    repair_prompt = _safe_text(prompt_bundle.get("repair_prompt", ""))
    legacy_prompt = _safe_text(prompt_bundle.get("prompt", ""))

    if not model_prompt:
        model_prompt = positive_prompt or story_prompt or legacy_prompt or analysis_prompt
    if not positive_prompt:
        positive_prompt = model_prompt or story_prompt or legacy_prompt or analysis_prompt
    if not analysis_prompt:
        analysis_prompt = legacy_prompt or positive_prompt
    if not story_prompt:
        story_prompt = model_prompt or positive_prompt or legacy_prompt

    return {
        "analysis_prompt": analysis_prompt,
        "story_prompt": story_prompt,
        "continuity_prompt": continuity_prompt,
        "model_prompt": model_prompt,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "repair_prompt": repair_prompt,
        "legacy_prompt": legacy_prompt,
    }


def choose_generation_prompt(
    prompt_bundle: Dict[str, Any],
    prefer_repair: bool = False,
    prefer_positive: bool = True,
    max_len: int = 900,
) -> str:
    """
    Choose the actual prompt text to send to the model.

    Policy:
    - repair mode -> repair_prompt if available
    - else prefer positive_prompt
    - else model_prompt
    - else story_prompt
    - else legacy prompt
    - else analysis prompt

    Also trims overlong prompts conservatively.
    """
    variants = get_prompt_variants(prompt_bundle)

    candidates: List[str] = []
    if prefer_repair:
        candidates.append(variants["repair_prompt"])
    if prefer_positive:
        candidates.append(variants["positive_prompt"])
    candidates.extend([
        variants["model_prompt"],
        variants["story_prompt"],
        variants["legacy_prompt"],
        variants["analysis_prompt"],
    ])

    prompt = ""
    for item in candidates:
        item = _safe_text(item)
        if item:
            prompt = item
            break

    prompt = prompt.strip()
    if len(prompt) <= max_len:
        return prompt

    # Conservative trimming: preserve beginning because most scene content
    # and continuity instructions are front-loaded.
    return prompt[: max_len - 3].rstrip() + "..."


def choose_prompt_variant(
    prompt_bundle: Dict[str, Any],
    backend_kind: str = "",
    repair_mode: bool = False,
    max_len: Optional[int] = None,
) -> str:
    """Backward-compatible prompt selector used by older backend code.

    This keeps the current prompt-bundle design while preserving the older
    choose_prompt_variant(...) API that some backends still import.
    """
    backend_kind = _safe_text(backend_kind).lower()
    if max_len is None:
        max_len = 1100 if backend_kind == "cogvideox" else 850

    return choose_generation_prompt(
        prompt_bundle=prompt_bundle,
        prefer_repair=repair_mode,
        prefer_positive=True,
        max_len=max_len,
    )


def build_negative_prompt(
    prompt_bundle: Dict[str, Any],
    backend_kind: str = "",
    extra_negative_terms: Optional[List[str]] = None,
    repair_mode: bool = False,
) -> str:
    """
    Build the final negative prompt sent to the backend.
    """
    prompt_bundle = _safe_dict(prompt_bundle)
    variants = get_prompt_variants(prompt_bundle)
    negative_prompt = variants["negative_prompt"]

    extras = list(extra_negative_terms or [])

    if repair_mode:
        extras.extend([
            "continuity break",
            "identity mismatch",
            "location mismatch",
        ])

    backend_kind = _safe_text(backend_kind).lower()
    if backend_kind in {"svd", "stablevideodiffusion", "stable_video_diffusion"}:
        extras.extend([
            "watermark",
            "text overlay",
            "frame corruption",
            "washed out frame",
        ])

    if backend_kind in {"cogvideox", "cogvideo", "cogvideo_x"}:
        extras.extend([
            "subject inconsistency",
            "background inconsistency",
            "hard identity drift",
        ])

    return _merge_csv_text(negative_prompt, extras)


# ---------------------------------------------------------------------
# control policy helpers
# ---------------------------------------------------------------------

def normalize_control_weights(control_weights: Dict[str, Any]) -> Dict[str, float]:
    control_weights = _safe_dict(control_weights)

    out = {
        "identity_strength": safe_float(
            control_weights.get("identity_strength", control_weights.get("identity_weight", 0.0)),
            0.0,
        ),
        "location_strength": safe_float(
            control_weights.get("location_strength", control_weights.get("location_weight", 0.0)),
            0.0,
        ),
        "style_strength": safe_float(
            control_weights.get("style_strength", control_weights.get("style_weight", 0.0)),
            0.0,
        ),
        "prop_strength": safe_float(
            control_weights.get("prop_strength", control_weights.get("props_weight", 0.0)),
            0.0,
        ),
        "transition_strength": safe_float(
            control_weights.get("transition_strength", control_weights.get("transition_weight", 0.0)),
            0.0,
        ),
        "continuity_strength": safe_float(control_weights.get("continuity_strength", 0.5), 0.5),
        "motion_strength": safe_float(control_weights.get("motion_strength", 0.5), 0.5),
    }

    # Preserve aliases for older code.
    out["identity_weight"] = out["identity_strength"]
    out["location_weight"] = out["location_strength"]
    out["style_weight"] = out["style_strength"]
    out["props_weight"] = out["prop_strength"]
    out["transition_weight"] = out["transition_strength"]

    return out


def strengthen_params_for_retry(
    generation_params: Dict[str, Any],
    control_weights: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
    retry_index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Backend-facing strengthening logic for retries.

    This is intentionally stronger than plain prompt changes:
    - guidance / steps / reference strength can increase
    - motion freedom can be reduced
    - control locks become tighter
    """
    params = copy.deepcopy(_safe_dict(generation_params))
    weights = normalize_control_weights(control_weights)
    prompt_bundle = _safe_dict(prompt_bundle)

    metadata = _safe_dict(prompt_bundle.get("prompt_metadata", {}))
    retry_context = _safe_dict(prompt_bundle.get("retry_context", {}))
    failure_tags = _dedupe_keep_order(
        [_safe_text(x) for x in _safe_list(metadata.get("failure_tags", []))]
        + [_safe_text(x) for x in _safe_list(retry_context.get("failure_tags", []))]
    )
    failure_tags_low = [x.lower() for x in failure_tags]

    guidance = safe_float(params.get("guidance_scale", 5.0), 5.0)
    steps = safe_int(params.get("num_inference_steps", 20), 20)
    strength = safe_float(params.get("strength", 0.75), 0.75)
    reference_strength = safe_float(params.get("reference_strength", 0.70), 0.70)

    if retry_index >= 1:
        guidance += 0.40
        steps += 2
        reference_strength = min(1.00, reference_strength + 0.08)

        weights["continuity_strength"] = min(1.00, weights["continuity_strength"] + 0.12)
        weights["identity_strength"] = min(1.00, weights["identity_strength"] + 0.08)
        weights["location_strength"] = min(1.00, weights["location_strength"] + 0.08)
        weights["style_strength"] = min(1.00, weights["style_strength"] + 0.05)
        weights["prop_strength"] = min(1.00, weights["prop_strength"] + 0.05)

    if retry_index >= 2:
        guidance += 0.40
        steps += 3
        reference_strength = min(1.00, reference_strength + 0.10)
        strength = min(1.00, strength + 0.05)
        weights["transition_strength"] = min(1.00, weights["transition_strength"] + 0.08)
        weights["motion_strength"] = max(0.20, weights["motion_strength"] - 0.10)

    if "identity_drift" in failure_tags_low:
        weights["identity_strength"] = min(1.00, weights["identity_strength"] + 0.18)
        weights["continuity_strength"] = min(1.00, weights["continuity_strength"] + 0.06)
        reference_strength = min(1.00, reference_strength + 0.10)
        guidance += 0.20

    if "location_drift" in failure_tags_low:
        weights["location_strength"] = min(1.00, weights["location_strength"] + 0.18)
        weights["continuity_strength"] = min(1.00, weights["continuity_strength"] + 0.05)
        reference_strength = min(1.00, reference_strength + 0.10)
        guidance += 0.20

    if "prop_loss" in failure_tags_low or "props_drift" in failure_tags_low:
        weights["prop_strength"] = min(1.00, weights["prop_strength"] + 0.18)

    if "style_drift" in failure_tags_low:
        weights["style_strength"] = min(1.00, weights["style_strength"] + 0.15)
        guidance += 0.20

    if "transition_break" in failure_tags_low or "transition_drift" in failure_tags_low:
        weights["transition_strength"] = min(1.00, weights["transition_strength"] + 0.20)
        weights["continuity_strength"] = min(1.00, weights["continuity_strength"] + 0.06)
        reference_strength = min(1.00, reference_strength + 0.12)

    if "motion_drift" in failure_tags_low:
        weights["motion_strength"] = max(0.15, weights["motion_strength"] - 0.12)
        params["motion_scheduler_bias"] = "stable"

    params["guidance_scale"] = round(guidance, 4)
    params["num_inference_steps"] = int(steps)
    params["strength"] = round(min(1.00, strength), 4)
    params["reference_strength"] = round(min(1.00, reference_strength), 4)
    params["retry_index"] = int(retry_index)

    weights["identity_weight"] = weights["identity_strength"]
    weights["location_weight"] = weights["location_strength"]
    weights["style_weight"] = weights["style_strength"]
    weights["props_weight"] = weights["prop_strength"]
    weights["transition_weight"] = weights["transition_strength"]

    return params, weights


def strengthen_controls_for_retry(
    control_weights: Dict[str, Any],
    retry_index: int = 0,
    repair_mode: bool = False,
) -> Dict[str, float]:
    """Backward-compatible control-only retry strengthening for older code."""
    weights = normalize_control_weights(control_weights)

    if retry_index >= 1:
        weights["continuity_strength"] = min(1.0, weights["continuity_strength"] + 0.10)
        weights["identity_strength"] = min(1.0, weights["identity_strength"] + 0.08)
        weights["location_strength"] = min(1.0, weights["location_strength"] + 0.08)
        weights["style_strength"] = min(1.0, weights["style_strength"] + 0.05)
        weights["prop_strength"] = min(1.0, weights["prop_strength"] + 0.05)

    if retry_index >= 2:
        weights["transition_strength"] = min(1.0, weights["transition_strength"] + 0.08)
        weights["motion_strength"] = max(0.20, weights["motion_strength"] - 0.08)

    if repair_mode:
        weights["continuity_strength"] = min(1.0, weights["continuity_strength"] + 0.12)
        weights["identity_strength"] = min(1.0, weights["identity_strength"] + 0.10)
        weights["location_strength"] = min(1.0, weights["location_strength"] + 0.10)
        weights["prop_strength"] = min(1.0, weights["prop_strength"] + 0.08)
        weights["style_strength"] = min(1.0, weights["style_strength"] + 0.08)
        weights["transition_strength"] = min(1.0, weights["transition_strength"] + 0.10)

    weights["identity_weight"] = weights["identity_strength"]
    weights["location_weight"] = weights["location_strength"]
    weights["style_weight"] = weights["style_strength"]
    weights["props_weight"] = weights["prop_strength"]
    weights["transition_weight"] = weights["transition_strength"]

    return weights


# ---------------------------------------------------------------------
# reference policy / reference ranking
# ---------------------------------------------------------------------

def _score_reference_candidate(
    candidate_role: str,
    candidate_path: str,
    control_weights: Dict[str, float],
    same_as_previous: Dict[str, Any],
    prompt_bundle: Dict[str, Any],
) -> float:
    """
    Assign a routing score to each candidate reference.
    """
    if not _file_exists(candidate_path):
        return -1.0

    role = _safe_text(candidate_role).lower()
    weights = normalize_control_weights(control_weights)
    same_as_previous = _safe_dict(same_as_previous)
    prompt_bundle = _safe_dict(prompt_bundle)

    score = 0.0
    if role == "identity_reference":
        score += 1.20 + weights["identity_strength"]
        if same_as_previous.get("character_identity", False):
            score += 0.40
        if same_as_previous.get("outfit", False):
            score += 0.15

    elif role == "location_reference":
        score += 1.10 + weights["location_strength"]
        if same_as_previous.get("location", False):
            score += 0.35

    elif role == "transition_reference":
        score += 1.00 + weights["transition_strength"]
        if _to_bool(_safe_dict(prompt_bundle.get("scene_packet", {})).get("dependent_on_previous", False), False):
            score += 0.50

    elif role == "prop_reference":
        score += 0.95 + weights["prop_strength"]
        if same_as_previous.get("props", False):
            score += 0.25

    else:
        score += 0.20

    retry_context = _safe_dict(prompt_bundle.get("retry_context", {}))
    failure_tags = [x.lower() for x in _safe_list(retry_context.get("failure_tags", []))]

    if "identity_drift" in failure_tags and role == "identity_reference":
        score += 0.60
    if "location_drift" in failure_tags and role == "location_reference":
        score += 0.60
    if ("transition_break" in failure_tags or "transition_drift" in failure_tags) and role == "transition_reference":
        score += 0.65
    if ("prop_loss" in failure_tags or "props_drift" in failure_tags) and role == "prop_reference":
        score += 0.55

    return score


def choose_reference_plan(
    prompt_bundle: Dict[str, Any],
    backend_kind: str = "",
) -> Dict[str, Any]:
    """
    Build a strong, explicit reference plan.
    """
    prompt_bundle = _safe_dict(prompt_bundle)
    reference_bundle = _safe_dict(prompt_bundle.get("reference_bundle", {}))
    control_weights = normalize_control_weights(prompt_bundle.get("control_weights", {}))
    contract = _safe_dict(prompt_bundle.get("prompt_contract", {}))
    same_as_previous = _safe_dict(contract.get("same_as_previous", {}))

    candidates: List[Dict[str, Any]] = []

    previous_keyframe = _safe_text(reference_bundle.get("previous_scene_keyframe", ""))
    if previous_keyframe:
        candidates.append({
            "role": "transition_reference",
            "path": previous_keyframe,
            "reason": "previous scene keyframe",
        })

    for item in _safe_list(reference_bundle.get("character_refs", [])):
        item = _safe_dict(item)
        path = _safe_text(item.get("path", ""))
        if path:
            candidates.append({
                "role": "identity_reference",
                "path": path,
                "reason": _safe_text(item.get("reason", "")) or "character reference",
            })

    location_ref = _safe_dict(reference_bundle.get("location_ref", {}))
    loc_path = _safe_text(location_ref.get("path", ""))
    if loc_path:
        candidates.append({
            "role": "location_reference",
            "path": loc_path,
            "reason": _safe_text(location_ref.get("reason", "")) or "location reference",
        })

    for item in _safe_list(reference_bundle.get("prop_refs", [])):
        item = _safe_dict(item)
        path = _safe_text(item.get("path", ""))
        if path:
            candidates.append({
                "role": "prop_reference",
                "path": path,
                "reason": _safe_text(item.get("reason", "")) or "prop reference",
            })

    primary_reference_path = _safe_text(reference_bundle.get("primary_reference_path", ""))
    primary_reference_type = _safe_text(reference_bundle.get("primary_reference_type", ""))
    if primary_reference_path:
        candidates.append({
            "role": primary_reference_type or "primary_reference",
            "path": primary_reference_path,
            "reason": _safe_text(reference_bundle.get("primary_reference_reason", "")) or "legacy primary reference",
        })

    for path in _safe_list(reference_bundle.get("secondary_references", [])):
        path = _safe_text(path)
        if path:
            candidates.append({
                "role": "secondary_reference",
                "path": path,
                "reason": "secondary reference",
            })

    ranked = []
    for cand in candidates:
        score = _score_reference_candidate(
            candidate_role=cand["role"],
            candidate_path=cand["path"],
            control_weights=control_weights,
            same_as_previous=same_as_previous,
            prompt_bundle=prompt_bundle,
        )
        if score >= 0.0:
            item = dict(cand)
            item["score"] = round(score, 6)
            ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    primary = ranked[0] if ranked else {}
    secondary_paths = _dedupe_keep_order([_safe_text(x.get("path", "")) for x in ranked[1:]])

    identity_reference_path = ""
    location_reference_path = ""
    transition_reference_path = ""
    prop_reference_path = ""

    for item in ranked:
        role = _safe_text(item.get("role", "")).lower()
        path = _safe_text(item.get("path", ""))
        if role == "identity_reference" and not identity_reference_path:
            identity_reference_path = path
        elif role == "location_reference" and not location_reference_path:
            location_reference_path = path
        elif role == "transition_reference" and not transition_reference_path:
            transition_reference_path = path
        elif role == "prop_reference" and not prop_reference_path:
            prop_reference_path = path

    backend_kind = _safe_text(backend_kind).lower()
    if backend_kind in {"svd", "stablevideodiffusion", "stable_video_diffusion"}:
        ordered = [
            transition_reference_path,
            identity_reference_path,
            location_reference_path,
            prop_reference_path,
            _safe_text(primary.get("path", "")),
        ]
        svd_primary = _first_existing_path(ordered)
        if svd_primary:
            for item in ranked:
                if _safe_text(item.get("path", "")) == svd_primary:
                    primary = item
                    break

    return {
        "primary_role": _safe_text(primary.get("role", "")),
        "primary_path": _safe_text(primary.get("path", "")),
        "primary_reason": _safe_text(primary.get("reason", "")),
        "secondary_paths": secondary_paths,
        "identity_reference_path": identity_reference_path,
        "location_reference_path": location_reference_path,
        "transition_reference_path": transition_reference_path,
        "prop_reference_path": prop_reference_path,
        "all_ranked": ranked,
        "has_any_reference": bool(ranked),
    }


def rank_reference_paths(reference_plan: Dict[str, Any]) -> List[str]:
    """Backward-compatible helper returning ranked reference paths only."""
    reference_plan = _safe_dict(reference_plan)

    ranked_items = _safe_list(reference_plan.get("all_ranked", []))
    if ranked_items:
        return [
            _safe_text(_safe_dict(item).get("path", ""))
            for item in ranked_items
            if _safe_text(_safe_dict(item).get("path", ""))
        ]

    ordered = [
        _safe_text(reference_plan.get("primary_path", "")),
        _safe_text(reference_plan.get("transition_reference_path", "")),
        _safe_text(reference_plan.get("identity_reference_path", "")),
        _safe_text(reference_plan.get("location_reference_path", "")),
        _safe_text(reference_plan.get("prop_reference_path", "")),
    ]

    legacy_refs = _safe_list(reference_plan.get("reference_paths", []))
    ordered.extend([_safe_text(x) for x in legacy_refs])
    return _dedupe_keep_order([x for x in ordered if x])


# ---------------------------------------------------------------------
# prompt construction for backend calls
# ---------------------------------------------------------------------

def build_clean_scene_prompt(
    prompt_bundle: Dict[str, Any],
    backend_kind: str = "",
    prefer_repair: bool = False,
) -> str:
    """
    Build the final model-facing prompt.
    """
    prompt_bundle = _safe_dict(prompt_bundle)
    backend_kind = _safe_text(backend_kind).lower()

    prompt = choose_generation_prompt(
        prompt_bundle=prompt_bundle,
        prefer_repair=prefer_repair,
        prefer_positive=True,
        max_len=1100 if backend_kind == "cogvideox" else 850,
    )

    if backend_kind in {"svd", "stablevideodiffusion", "stable_video_diffusion"}:
        prompt = prompt.replace(
            "Apply very strong continuity locking across identity, location, style, and props.",
            ""
        )
        prompt = " ".join(prompt.split()).strip()

    return prompt


# ---------------------------------------------------------------------
# generation parameter extraction
# ---------------------------------------------------------------------

def extract_generation_params(
    prompt_bundle: Dict[str, Any],
    backend_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend_defaults = _safe_dict(backend_defaults)
    prompt_bundle = _safe_dict(prompt_bundle)

    params = copy.deepcopy(_safe_dict(prompt_bundle.get("generation_params", {})))
    if not params:
        params = copy.deepcopy(backend_defaults)

    _setdefault_num(params, "guidance_scale", backend_defaults.get("guidance_scale", 5.0))
    _setdefault_num(params, "num_inference_steps", backend_defaults.get("num_inference_steps", 20))
    _setdefault_num(params, "strength", backend_defaults.get("strength", 0.75))
    _setdefault_num(params, "reference_strength", backend_defaults.get("reference_strength", 0.70))

    return params


def _setdefault_num(d: Dict[str, Any], key: str, value: Any) -> None:
    if key not in d or d.get(key) in ["", None]:
        d[key] = value


# ---------------------------------------------------------------------
# output metadata
# ---------------------------------------------------------------------

def infer_semantic_evidence_status(metadata: Dict[str, Any]) -> str:
    metadata = _safe_dict(metadata)

    if "generated_summary" not in metadata and "generated_metadata" in metadata:
        old_summary = _safe_dict(metadata.get("generated_metadata", {}))
        if old_summary:
            metadata["generated_summary"] = old_summary

    summary = _safe_dict(metadata.get("generated_summary", {}))
    characters = _safe_list(summary.get("characters", []))
    location = _safe_dict(summary.get("location", {}))
    props = _safe_list(summary.get("props", []))
    style = _safe_dict(summary.get("style", {}))

    has_summary_content = bool(characters or location or props or style)
    if has_summary_content:
        return "present"

    if _to_bool(metadata.get("placeholder_conditioning_only", False), False):
        return "missing"

    return "missing"


def build_generated_metadata_from_constraints(
    prompt_bundle: Dict[str, Any],
    reference_plan: Dict[str, Any],
    generation_params: Dict[str, Any],
    backend_kind: str,
    video_path: str = "",
    keyframe_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach rich metadata for:
    - continuity manager
    - scorer
    - later evaluation
    - debugging / experiment logs
    """
    prompt_bundle = _safe_dict(prompt_bundle)
    reference_plan = _safe_dict(reference_plan)
    generation_params = _safe_dict(generation_params)
    keyframe_paths = _safe_dict(keyframe_paths)

    variants = get_prompt_variants(prompt_bundle)
    metadata = {
        "backend_used": _safe_text(backend_kind),
        "video_path": _safe_text(video_path),
        "prompt_used": variants["positive_prompt"] or variants["model_prompt"],
        "negative_prompt_used": variants["negative_prompt"],
        "analysis_prompt": variants["analysis_prompt"],
        "story_prompt": variants["story_prompt"],
        "continuity_prompt": variants["continuity_prompt"],
        "repair_prompt": variants["repair_prompt"],
        "prompt_metadata": copy.deepcopy(_safe_dict(prompt_bundle.get("prompt_metadata", {}))),
        "retry_context": copy.deepcopy(_safe_dict(prompt_bundle.get("retry_context", {}))),
        "generation_params": copy.deepcopy(generation_params),
        "reference_plan": copy.deepcopy(reference_plan),
        "primary_reference_role": _safe_text(reference_plan.get("primary_role", "")),
        "primary_reference_path": _safe_text(reference_plan.get("primary_path", "")),
        "primary_reference_reason": _safe_text(reference_plan.get("primary_reason", "")),
        "secondary_reference_paths": _safe_list(reference_plan.get("secondary_paths", [])),
        "identity_reference_path": _safe_text(reference_plan.get("identity_reference_path", "")),
        "location_reference_path": _safe_text(reference_plan.get("location_reference_path", "")),
        "transition_reference_path": _safe_text(reference_plan.get("transition_reference_path", "")),
        "prop_reference_path": _safe_text(reference_plan.get("prop_reference_path", "")),
        "first_frame_path": _safe_text(keyframe_paths.get("first_frame_path", "")),
        "middle_frame_path": _safe_text(keyframe_paths.get("middle_frame_path", "")),
        "last_frame_path": _safe_text(keyframe_paths.get("last_frame_path", "")),
        "best_keyframe_path": _safe_text(
            keyframe_paths.get("best_keyframe_path", keyframe_paths.get("keyframe_path", ""))
        ),
        "generated_summary": {
            "characters": [],
            "location": {},
            "props": [],
            "style": {},
            "source": "missing",
        },
        "semantic_evidence_status": "missing",
        "placeholder_conditioning_only": False,
        "has_real_init_image": False,
        "has_real_reference_assets": False,
        "reference_source_type": "missing",
    }

    if "generated_summary" not in metadata and "generated_metadata" in metadata:
        old_meta = _safe_dict(metadata.get("generated_metadata", {}))
        if old_meta:
            metadata["generated_summary"] = copy.deepcopy(old_meta)

    generated_summary = _safe_dict(metadata.get("generated_summary", {}))
    generated_summary["characters"] = _safe_list(generated_summary.get("characters", []))
    generated_summary["location"] = _safe_dict(generated_summary.get("location", {}))
    generated_summary["props"] = _safe_list(generated_summary.get("props", []))
    generated_summary["style"] = _safe_dict(generated_summary.get("style", {}))
    generated_summary["source"] = _safe_text(generated_summary.get("source", "missing")) or "missing"
    metadata["generated_summary"] = generated_summary

    metadata["placeholder_conditioning_only"] = _to_bool(metadata.get("placeholder_conditioning_only", False), False)
    metadata["has_real_init_image"] = _to_bool(metadata.get("has_real_init_image", False), False)
    metadata["has_real_reference_assets"] = _to_bool(metadata.get("has_real_reference_assets", False), False)
    metadata["reference_source_type"] = _safe_text(metadata.get("reference_source_type", "missing")) or "missing"
    metadata["semantic_evidence_status"] = infer_semantic_evidence_status(metadata)

    return metadata


# ---------------------------------------------------------------------
# scene / role detection helpers
# ---------------------------------------------------------------------

def is_first_scene(prompt_bundle: Dict[str, Any]) -> bool:
    prompt_bundle = _safe_dict(prompt_bundle)
    scene_id = _safe_text(prompt_bundle.get("scene_id", ""))
    if not scene_id:
        scene_id = _safe_text(_safe_dict(prompt_bundle.get("scene_packet", {})).get("scene_id", ""))

    low = scene_id.lower()
    return low.endswith("001") or low in {"scene1", "scene_1", "1"}


def is_dependent_scene(prompt_bundle: Dict[str, Any]) -> bool:
    prompt_bundle = _safe_dict(prompt_bundle)
    scene_packet = _safe_dict(prompt_bundle.get("scene_packet", {}))
    if "dependent_on_previous" in scene_packet:
        return _to_bool(scene_packet.get("dependent_on_previous", False), False)

    contract = _safe_dict(prompt_bundle.get("prompt_contract", {}))
    story_core = _safe_dict(contract.get("story_core", {}))
    return _to_bool(story_core.get("dependent_on_previous", False), False)