# src/pipeline/run_text_pipeline.py
from typing import Dict, Any, List

from src.main import run_pipeline
from src.text.packet_builder import build_scene_packets


def _normalize_scene_texts(scenes_raw: List[Any]) -> List[str]:
    """
    Supports multiple possible scene formats:
    - ["scene 1 text", "scene 2 text"]
    - [{"scene_text": "..."}, ...]
    - [{"text": "..."}, ...]
    """
    scene_texts = []

    for item in scenes_raw or []:
        if isinstance(item, str):
            txt = item.strip()
            if txt:
                scene_texts.append(txt)
            continue

        if isinstance(item, dict):
            txt = (
                item.get("scene_text")
                or item.get("text")
                or item.get("content")
                or ""
            )
            txt = str(txt).strip()
            if txt:
                scene_texts.append(txt)

    return scene_texts


def run_text_planning_pipeline(
    prompt_text: str,
    config_path: str = "configs/settings.yaml",
    logger=None,
    raw_save_dir: str = None
) -> Dict[str, Any]:
    """
    Runs:
    1. segmentation
    2. dependency detection
    3. scene packet building

    Returns:
    {
        "ok": True,
        "scenes": {"scenes":[...], "global_notes": {...}},
        "dependencies": {"dependencies":[...]},
        "scene_packets": [ ... ]
    }
    """
    out = run_pipeline(
        prompt_text=prompt_text,
        config_path=config_path,
        logger=logger,
        raw_save_dir=raw_save_dir
    )

    if not out.get("ok", False):
        return out

    result = out.get("result", {})
    scenes_raw = result.get("scenes", [])
    dependencies = result.get("dependencies", [])
    global_notes = result.get("global_notes", {})

    scene_texts = _normalize_scene_texts(scenes_raw)

    if logger:
        logger.info(f"Building scene packets for {len(scene_texts)} scenes")

    scene_packets = build_scene_packets(
        scenes=scene_texts,
        dependencies=dependencies
    )

    return {
        "ok": True,
        "scenes": {
            "scenes": scene_texts,
            "global_notes": global_notes
        },
        "dependencies": {
            "dependencies": dependencies
        },
        "scene_packets": scene_packets,
        "raw_llm_result": result
    }