# src/generation/retry.py
from __future__ import annotations

import copy
from typing import Dict, Any, List


def build_retry_prompt_bundle(
    original_prompt_bundle: Dict[str, Any],
    scene_packet: Dict[str, Any],
    continuity_payload: Dict[str, Any],
    retry_index: int,
) -> Dict[str, Any]:
    """
    Build a compact retry prompt bundle.

    Important rule:
    - do NOT keep appending long retry paragraphs
    - keep only short, high-priority corrections
    """
    bundle = copy.deepcopy(original_prompt_bundle or {})
    scene_packet = scene_packet or {}
    continuity_payload = continuity_payload or {}

    failure_tags = _dedupe_keep_order([
        _safe_text(x) for x in continuity_payload.get("failure_tags", []) if _safe_text(x)
    ])
    drift_report = continuity_payload.get("drift_report", {}) or {}

    short_lines = _build_retry_lines(
        scene_packet=scene_packet,
        failure_tags=failure_tags,
        drift_report=drift_report,
    )

    bundle["positive_prompt"] = _truncate_words(
        _replace_tail_with_compact_retry(bundle.get("positive_prompt", ""), short_lines),
        95,
    )
    bundle["model_prompt"] = _truncate_words(
        _replace_tail_with_compact_retry(bundle.get("model_prompt", ""), short_lines),
        95,
    )
    bundle["continuity_prompt"] = _truncate_words(
        _replace_tail_with_compact_retry(bundle.get("continuity_prompt", ""), short_lines[:2]),
        70,
    )
    bundle["repair_prompt"] = _truncate_words(
        _replace_tail_with_compact_retry(bundle.get("repair_prompt", ""), short_lines),
        95,
    )
    bundle["prompt"] = bundle["model_prompt"]

    neg_terms = _build_retry_negative_terms(failure_tags)
    bundle["negative_prompt"] = _append_csv_terms(
        bundle.get("negative_prompt", ""),
        neg_terms,
        max_terms=24,
    )

    prompt_metadata = dict(bundle.get("prompt_metadata", {}) or {})
    prompt_metadata["retry_index"] = int(retry_index or 0)
    prompt_metadata["repair_mode"] = False
    prompt_metadata["failure_tags"] = failure_tags
    prompt_metadata["drift_severity"] = _safe_text(drift_report.get("severity", ""))
    bundle["prompt_metadata"] = prompt_metadata

    retry_context = dict(bundle.get("retry_context", {}) or {})
    retry_context["retry_index"] = int(retry_index or 0)
    retry_context["failure_tags"] = failure_tags
    retry_context["drift_report"] = drift_report
    bundle["retry_context"] = retry_context

    generation_params = dict(bundle.get("generation_params", {}) or {})
    generation_params["retry_index"] = int(retry_index or 0)
    bundle["generation_params"] = generation_params

    return bundle


def _build_retry_lines(
    scene_packet: Dict[str, Any],
    failure_tags: List[str],
    drift_report: Dict[str, Any],
) -> List[str]:
    same_as_previous = scene_packet.get("same_as_previous", {}) or {}

    lines: List[str] = ["Reduce drift and keep the intended scene"]

    if "identity_drift" in failure_tags or bool(same_as_previous.get("character_identity", False)):
        lines.append("Keep the same character identity")

    if "location_drift" in failure_tags or bool(same_as_previous.get("location", False)):
        lines.append("Keep the same location layout")

    props = same_as_previous.get("props", [])
    if "prop_loss" in failure_tags or "props_drift" in failure_tags or (isinstance(props, list) and len(props) > 0):
        lines.append("Keep the same important props")

    if "style_drift" in failure_tags or bool(same_as_previous.get("style", False)):
        lines.append("Keep the same cinematic style")

    if "transition_drift" in failure_tags or "transition_break" in failure_tags:
        lines.append("Make the opening continue the previous scene naturally")

    severity = _safe_text(drift_report.get("severity", "")).lower()
    if severity in {"medium", "high"}:
        lines.append("Prefer continuity over creativity")

    return _dedupe_keep_order(lines[:5])


def _build_retry_negative_terms(failure_tags: List[str]) -> List[str]:
    terms = ["random image", "placeholder output", "wrong scene"]

    if "identity_drift" in failure_tags:
        terms.extend(["wrong character", "changed face"])

    if "location_drift" in failure_tags:
        terms.extend(["wrong background", "different location"])

    if "prop_loss" in failure_tags or "props_drift" in failure_tags:
        terms.extend(["missing prop", "wrong object"])

    if "style_drift" in failure_tags:
        terms.extend(["style mismatch"])

    if "transition_drift" in failure_tags or "transition_break" in failure_tags:
        terms.extend(["abrupt transition", "hard reset"])

    return _dedupe_keep_order(terms)


def _replace_tail_with_compact_retry(base_text: str, short_lines: List[str]) -> str:
    base = _safe_text(base_text)
    lines = [_safe_text(x) for x in short_lines if _safe_text(x)]
    if not lines:
        return base

    if not base:
        return ". ".join(lines)

    # keep only the first main sentence block from base, then add compact retry instructions
    first_part = _truncate_words(base, 55)
    return first_part + ". " + ". ".join(lines)


def _append_csv_terms(base: str, terms: List[str], max_terms: int) -> str:
    items = []
    if _safe_text(base):
        items.extend([x.strip() for x in _safe_text(base).split(",") if x.strip()])
    items.extend([_safe_text(x) for x in (terms or []) if _safe_text(x)])
    items = _dedupe_keep_order(items)
    return ", ".join(items[:max_terms])


def _truncate_words(text: str, max_words: int) -> str:
    words = _safe_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


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