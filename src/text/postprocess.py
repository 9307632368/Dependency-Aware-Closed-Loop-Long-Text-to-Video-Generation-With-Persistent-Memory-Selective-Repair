# src/text/postprocess.py
from typing import Dict, Any, List, Tuple


def normalize_text(s: str) -> str:
    """
    Light normalization for matching.
    """
    return " ".join(s.strip().split())


def assign_scene_ids(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reassign scene ids sequentially from 1..N
    """
    for i, scene in enumerate(scenes, start=1):
        scene["scene_id"] = i
    return scenes


def compute_char_indices(original_text: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute deterministic start_char_index and end_char_index
    by locating scene_text inside original_text in order.
    """
    search_start = 0
    original = original_text

    for scene in scenes:
        scene_text = scene.get("scene_text", "").strip()

        if not scene_text:
            scene["start_char_index"] = -1
            scene["end_char_index"] = -1
            continue

        idx = original.find(scene_text, search_start)

        # fallback: try normalized search
        if idx == -1:
            norm_original = normalize_text(original[search_start:])
            norm_scene = normalize_text(scene_text)

            norm_idx = norm_original.find(norm_scene)
            if norm_idx != -1:
                # approximate mapping back to original range
                idx = search_start + original[search_start:].find(scene_text[: min(len(scene_text), 30)])
            else:
                idx = -1

        if idx == -1:
            scene["start_char_index"] = -1
            scene["end_char_index"] = -1
        else:
            scene["start_char_index"] = idx
            scene["end_char_index"] = idx + len(scene_text)
            search_start = idx + len(scene_text)

    return scenes


def check_coverage(original_text: str, scenes: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Check whether scenes cover the text in order and without major missing parts.
    This is a practical check, not mathematically perfect.
    """
    total_text_len = len(original_text.strip())
    matched_len = 0
    ordered = True
    missing_indices = []

    prev_end = -1

    for i, scene in enumerate(scenes):
        start = scene.get("start_char_index", -1)
        end = scene.get("end_char_index", -1)

        if start == -1 or end == -1:
            missing_indices.append(i + 1)
            continue

        if start < prev_end:
            ordered = False

        if end > start:
            matched_len += (end - start)

        prev_end = end

    coverage_ratio = 0.0
    if total_text_len > 0:
        coverage_ratio = matched_len / total_text_len

    info = {
        "ordered": ordered,
        "coverage_ratio": round(coverage_ratio, 4),
        "missing_scene_matches": missing_indices,
        "total_scenes": len(scenes),
    }

    # relaxed threshold because LLM may paraphrase slightly
    ok = ordered and (coverage_ratio >= 0.6) and (len(missing_indices) <= max(1, len(scenes) // 3))

    return ok, info


def finalize_scenes(original_text: str, scenes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    End-to-end scene cleanup.
    """
    scenes = assign_scene_ids(scenes)

    for scene in scenes:
        scene["scene_text"] = scene.get("scene_text", "").strip()
        scene["location"] = scene.get("location", "").strip()
        scene["time_hint"] = scene.get("time_hint", "").strip()
        scene["core_action"] = scene.get("core_action", "").strip()
        scene["camera_style"] = scene.get("camera_style", "").strip()
        scene["new_scene_reason"] = scene.get("new_scene_reason", "").strip()

        if "characters" not in scene or not isinstance(scene["characters"], list):
            scene["characters"] = []

    scenes = compute_char_indices(original_text, scenes)
    ok, coverage_info = check_coverage(original_text, scenes)

    return scenes, {
        "coverage_ok": ok,
        "coverage_info": coverage_info
    }