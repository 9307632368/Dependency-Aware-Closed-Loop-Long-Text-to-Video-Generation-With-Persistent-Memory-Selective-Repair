# src/pipeline/run_full_pipeline.py
from __future__ import annotations

import json
import os
import time
import copy
from pathlib import Path
from typing import Dict, Any, List

from src.pipeline.run_generation_pipeline import run_generation_pipeline
from src.video.stitch import stitch_scene_clips


def _safe_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {}


def _safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return []


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_dir(path: str) -> str:
    path = _safe_text(path)
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _inject_run_specific_backend_dirs(cfg: dict, run_root: Path) -> dict:
    cfg = copy.deepcopy(cfg or {})

    generated_root = run_root / "generated"
    cog_dir = generated_root / "cogvideox"
    svd_dir = generated_root / "svd"

    _ensure_dir(str(generated_root))
    _ensure_dir(str(cog_dir))
    _ensure_dir(str(svd_dir))

    cfg["output_dir"] = str(generated_root)

    if "backends" not in cfg or not isinstance(cfg["backends"], dict):
        cfg["backends"] = {}

    if "cogvideox" not in cfg["backends"] or not isinstance(cfg["backends"]["cogvideox"], dict):
        cfg["backends"]["cogvideox"] = {}
    if "svd" not in cfg["backends"] or not isinstance(cfg["backends"]["svd"], dict):
        cfg["backends"]["svd"] = {}

    cfg["backends"]["cogvideox"]["output_dir"] = str(cog_dir)
    cfg["backends"]["svd"]["output_dir"] = str(svd_dir)

    backend_name = str(cfg.get("backend", "") or "").strip().lower()
    if backend_name == "cogvideox":
        cfg["output_dir"] = str(cog_dir)
    elif backend_name == "svd":
        cfg["output_dir"] = str(svd_dir)
    else:
        cfg["output_dir"] = str(generated_root)

    return cfg


def run_full_pipeline(
    prompt_text: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full long-video pipeline:
    1. Text planning pipeline
    2. Generation pipeline
    3. Stitching
    4. Summary + artifacts

    This version keeps generated media inside the current full-pipeline run folder.
    """
    start_time = time.time()

    config = dict(config or {})
    output_cfg = _safe_dict(config.get("output", {}))
    full_output_dir = Path(
        _ensure_dir(
            output_cfg.get(
                "full_pipeline_output_dir",
                output_cfg.get("output_dir", "outputs/full_pipeline"),
            )
        )
    )
    artifacts_dir = Path(_ensure_dir(str(full_output_dir / "artifacts")))
    generation_dir = Path(_ensure_dir(str(full_output_dir / "generation")))
    generated_media_dir = Path(_ensure_dir(str(full_output_dir / "generated")))

    config = _inject_run_specific_backend_dirs(config, full_output_dir)
    config.setdefault("output", {})
    config["output"]["generation_output_dir"] = str(generation_dir)

    # ------------------------------------------------------------
    # Stage 1: Text pipeline
    # ------------------------------------------------------------
    text_pipeline_output = _run_text_stage(
        prompt_text=prompt_text,
        config=config,
    )
    _write_json(
        str(full_output_dir / "text_pipeline_output.json"),
        text_pipeline_output,
    )

    scene_packets = _extract_scene_packets(text_pipeline_output)

    # ------------------------------------------------------------
    # Stage 2: Generation pipeline
    # ------------------------------------------------------------
    generation_pipeline_output = run_generation_pipeline(
        scene_packets=scene_packets,
        config=config,
    )
    _write_json(
        str(full_output_dir / "generation_pipeline_output.json"),
        generation_pipeline_output,
    )

    # ------------------------------------------------------------
    # Stage 3: Stitching
    # ------------------------------------------------------------
    stitching_output = stitch_scene_clips(
        generation_output=generation_pipeline_output,
        output_dir=str(full_output_dir),
        config=config,
    )
    _write_json(
        str(full_output_dir / "stitching_output.json"),
        stitching_output,
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    elapsed_seconds = time.time() - start_time

    full_summary = _build_full_pipeline_summary(
        prompt_text=prompt_text,
        text_pipeline_output=text_pipeline_output,
        generation_pipeline_output=generation_pipeline_output,
        stitching_output=stitching_output,
        elapsed_seconds=elapsed_seconds,
    )

    full_output = {
        "ok": bool(
            text_pipeline_output.get("ok", False)
            and generation_pipeline_output.get("scene_results") is not None
        ),
        "prompt_text": prompt_text,
        "text_pipeline_output": text_pipeline_output,
        "generation_pipeline_output": generation_pipeline_output,
        "stitching_output": stitching_output,
        "full_pipeline_summary": full_summary,
        "generated_media_root": str(generated_media_dir),
    }

    _write_json(
        str(full_output_dir / "full_pipeline_summary.json"),
        full_summary,
    )
    _write_json(
        str(full_output_dir / "full_pipeline_output.json"),
        full_output,
    )

    _write_research_artifacts(
        artifacts_dir=str(artifacts_dir),
        prompt_text=prompt_text,
        text_pipeline_output=text_pipeline_output,
        generation_pipeline_output=generation_pipeline_output,
        stitching_output=stitching_output,
        full_summary=full_summary,
    )

    return full_output


# ----------------------------------------------------------------------
# Text stage compatibility
# ----------------------------------------------------------------------

def _run_text_stage(prompt_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.pipeline.run_text_pipeline import run_text_pipeline
        out = run_text_pipeline(prompt_text=prompt_text, config=config)
        return _safe_dict(out)
    except Exception:
        pass

    try:
        from src.pipeline.run_text_pipeline import run_text_planning_pipeline
        out = run_text_planning_pipeline(prompt_text=prompt_text, config=config)
        return _safe_dict(out)
    except Exception as e:
        return {
            "ok": False,
            "error": f"text pipeline import/execution failed: {e}",
            "scene_packets": [],
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _extract_scene_packets(text_pipeline_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    text_pipeline_output = _safe_dict(text_pipeline_output)

    for key in ["scene_packets", "packets", "scenes_with_packets"]:
        items = _safe_list(text_pipeline_output.get(key, []))
        if items:
            return items

    scenes = _safe_list(text_pipeline_output.get("scenes", []))
    packets: List[Dict[str, Any]] = []

    for idx, scene in enumerate(scenes):
        scene = _safe_dict(scene)
        packets.append({
            "scene_id": _safe_text(scene.get("scene_id", "")) or f"scene_{idx + 1:03d}",
            "text": _safe_text(scene.get("text", "")) or _safe_text(scene.get("scene_text", "")),
            "scene_text": _safe_text(scene.get("scene_text", "")) or _safe_text(scene.get("text", "")),
            "dependent_on_previous": bool(scene.get("dependent_on_previous", False)),
            "same_as_previous": _safe_dict(scene.get("same_as_previous", {})),
            "metadata": scene,
        })

    return packets


def _build_full_pipeline_summary(
    prompt_text: str,
    text_pipeline_output: Dict[str, Any],
    generation_pipeline_output: Dict[str, Any],
    stitching_output: Dict[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    scene_results = _safe_list(generation_pipeline_output.get("scene_results", []))
    accepted_scene_results = _safe_list(generation_pipeline_output.get("accepted_scene_results", []))
    failed_scene_results = _safe_list(generation_pipeline_output.get("failed_scene_results", []))
    generation_summary = _safe_dict(generation_pipeline_output.get("pipeline_summary", {}))

    backend_counts = _safe_dict(generation_summary.get("backend_counts", {}))
    avg_score = _safe_float(generation_summary.get("avg_score", 0.0), 0.0)

    scene_table = []
    for item in scene_results:
        item = _safe_dict(item)
        score_report = _safe_dict(item.get("score_report", {}))
        generation_result = _safe_dict(item.get("generation_result", {}))
        scene_table.append({
            "scene_id": _safe_text(item.get("scene_id", "")),
            "accepted": bool(item.get("accepted", False)),
            "backend_used": _safe_text(item.get("backend_used", "")),
            "score": _extract_score(score_report),
            "retry_count": int(item.get("retry_count", 0) or 0),
            "repair_used": bool(item.get("repair_used", False)),
            "video_path": _safe_text(
                generation_result.get("video_path", "")
                or generation_result.get("output_video_path", "")
            ),
        })

    return {
        "num_scenes": len(scene_results),
        "num_accepted_scenes": len(accepted_scene_results),
        "num_failed_scenes": len(failed_scene_results),
        "accept_rate": (len(accepted_scene_results) / len(scene_results)) if scene_results else 0.0,
        "avg_scene_score": avg_score,
        "backend_counts": backend_counts,
        "final_video_available": bool(_safe_text(stitching_output.get("video_path", ""))),
        "final_video_path": _safe_text(stitching_output.get("video_path", "")),
        "elapsed_seconds": elapsed_seconds,
        "scene_table": scene_table,
        "prompt_preview": prompt_text[:500],
    }


def _write_research_artifacts(
    artifacts_dir: str,
    prompt_text: str,
    text_pipeline_output: Dict[str, Any],
    generation_pipeline_output: Dict[str, Any],
    stitching_output: Dict[str, Any],
    full_summary: Dict[str, Any],
) -> None:
    _ensure_dir(artifacts_dir)

    _write_json(
        os.path.join(artifacts_dir, "input_prompt.json"),
        {"prompt_text": prompt_text},
    )

    _write_json(
        os.path.join(artifacts_dir, "text_stage_summary.json"),
        {
            "ok": bool(text_pipeline_output.get("ok", False)),
            "num_scene_packets": len(_extract_scene_packets(text_pipeline_output)),
            "keys": list(_safe_dict(text_pipeline_output).keys()),
        },
    )

    _write_json(
        os.path.join(artifacts_dir, "generation_stage_summary.json"),
        {
            "pipeline_summary": _safe_dict(generation_pipeline_output.get("pipeline_summary", {})),
            "num_scene_results": len(_safe_list(generation_pipeline_output.get("scene_results", []))),
            "num_accepted_scene_results": len(_safe_list(generation_pipeline_output.get("accepted_scene_results", []))),
            "num_failed_scene_results": len(_safe_list(generation_pipeline_output.get("failed_scene_results", []))),
        },
    )

    _write_json(
        os.path.join(artifacts_dir, "stitching_stage_summary.json"),
        {
            "ok": bool(stitching_output.get("ok", False)),
            "video_path": _safe_text(stitching_output.get("video_path", "")),
            "clips_used": _safe_list(stitching_output.get("clips_used", [])),
            "reason": _safe_text(stitching_output.get("reason", "")),
        },
    )

    _write_json(
        os.path.join(artifacts_dir, "full_summary.json"),
        full_summary,
    )


def _extract_score(score_report: Dict[str, Any]) -> float:
    score_report = _safe_dict(score_report)
    if "overall_score" in score_report:
        return _safe_float(score_report.get("overall_score", 0.0), 0.0)
    if "score" in score_report:
        return _safe_float(score_report.get("score", 0.0), 0.0)
    return 0.0