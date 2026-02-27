# src/pipeline/run_generation_pipeline.py
from __future__ import annotations

import copy
import json
import os
from typing import Dict, Any, List, Optional

from src.generation.scene_generator import SceneGenerator
from src.continuity.manager import ContinuityManager
from src.generation.backend.factory import build_backend
from src.repair.scene_repair import SceneRepairEngine


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


def _json_safe(obj: Any):
    """
    Convert complex runtime objects into JSON-safe structures.
    Important for:
    - PIL Images
    - numpy arrays
    - custom objects
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]

    # PIL Image
    cls_name = obj.__class__.__name__
    mod_name = getattr(obj.__class__, "__module__", "")

    if cls_name == "Image" or mod_name.startswith("PIL."):
        return f"<nonserializable:{cls_name}>"

    # numpy arrays
    if mod_name.startswith("numpy"):
        return f"<nonserializable:{cls_name}>"

    # fallback for anything else
    return f"<nonserializable:{cls_name}>"


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)


def run_generation_pipeline(
    scene_packets: List[Dict[str, Any]],
    config: Dict[str, Any],
    continuity_manager: Optional[ContinuityManager] = None,
    scene_generator: Optional[SceneGenerator] = None,
    repair_engine: Optional[SceneRepairEngine] = None,
) -> Dict[str, Any]:
    """
    Run scene-by-scene generation pipeline.

    Responsibilities:
    - prepare continuity package for each scene
    - run scene generation with retry / repair through SceneGenerator
    - optionally run explicit SceneRepairEngine closed-loop repair
    - update continuity memory only after accepted generation
    - keep detailed per-scene records
    - support dependency-aware downstream regeneration marks
    - produce pipeline-level summary for later stitching / evaluation
    """
    config = dict(config or {})
    scene_packets = [dict(x or {}) for x in (scene_packets or [])]

    output_cfg = _safe_dict(config.get("output", {}))
    generation_cfg = _safe_dict(config.get("generation", {}))
    continuity_cfg = _safe_dict(config.get("continuity", {}))
    repair_cfg = _safe_dict(config.get("repair", {}))

    output_dir = _ensure_dir(
        output_cfg.get(
            "generation_output_dir",
            output_cfg.get(
                "output_dir",
                "outputs/generation_pipeline",
            ),
        )
    )

    pipeline_debug_dir = _ensure_dir(os.path.join(output_dir, "debug"))
    scenes_debug_dir = _ensure_dir(os.path.join(pipeline_debug_dir, "scenes"))

    if continuity_manager is None:
        continuity_manager = ContinuityManager(
            config=continuity_cfg if continuity_cfg else config
        )

    if scene_generator is None:
        backend = build_backend(config)
        scorer = _build_scorer_from_config(config)
        repair_backend = _build_repair_backend_from_config(config)

        scene_generator = SceneGenerator(
            backend=backend,
            scorer=scorer,
            repair_backend=repair_backend,
            config=config,
        )

    if repair_engine is None:
        repair_engine = SceneRepairEngine(config)

    scene_results: List[Dict[str, Any]] = []
    accepted_scene_results: List[Dict[str, Any]] = []
    failed_scene_results: List[Dict[str, Any]] = []

    invalidated_scene_ids: List[str] = []
    downstream_regeneration_queue: List[str] = []

    hard_fail_stop = bool(generation_cfg.get("stop_on_hard_failure", False))
    dependency_aware_repair = bool(
        repair_cfg.get("dependency_aware_repair", True)
    )
    score_commit_threshold = float(
        generation_cfg.get("commit_score_threshold", 0.60)
    )
    enable_explicit_repair_engine = bool(
        repair_cfg.get("enable_explicit_repair_engine", True)
    )

    for scene_index, scene_packet in enumerate(scene_packets):
        scene_packet = dict(scene_packet or {})
        scene_id = _scene_id(scene_packet, scene_index)

        scene_packet["_invalidated_by_upstream_repair"] = scene_id in invalidated_scene_ids

        try:
            continuity_package = continuity_manager.prepare_scene(scene_packet)
            continuity_package = dict(continuity_package or {})
        except Exception as e:
            failed = _build_prepare_failure_result(scene_packet, scene_id, str(e))
            scene_results.append(failed)
            failed_scene_results.append(failed)
            _write_scene_debug(scenes_debug_dir, scene_id, failed)
            if hard_fail_stop:
                break
            continue

        continuity_package["scene_id"] = continuity_package.get("scene_id", scene_id) or scene_id
        continuity_package["scene_packet"] = continuity_package.get("scene_packet", scene_packet) or scene_packet

        scene_output = scene_generator.generate_scene(continuity_package)
        scene_output = dict(scene_output or {})
        scene_output["scene_id"] = scene_output.get("scene_id", scene_id) or scene_id

        repair_debug = {
            "repair_engine_enabled": enable_explicit_repair_engine,
            "repair_attempted": False,
            "repair_result_used": False,
            "repair_engine_output": {},
        }

        if enable_explicit_repair_engine and not bool(scene_output.get("accepted", False)):
            repair_debug["repair_attempted"] = True

            repair_engine_output = repair_engine.repair_scene(
                scene_generator=scene_generator,
                continuity_package=continuity_package,
                scene_packet=scene_packet,
                original_scene_output=scene_output,
            )
            repair_engine_output = dict(repair_engine_output or {})
            repair_debug["repair_engine_output"] = repair_engine_output

            repair_result = _safe_dict(repair_engine_output.get("repair_result", {}))
            if repair_result and bool(repair_result.get("accepted", False)):
                scene_output = repair_result
                repair_debug["repair_result_used"] = True

        score_report = _safe_dict(scene_output.get("score_report", {}))
        drift_report = _safe_dict(scene_output.get("drift_report", {}))
        prompt_bundle = _safe_dict(scene_output.get("prompt_bundle", {}))
        generation_result = _safe_dict(scene_output.get("generation_result", {}))

        accepted = bool(scene_output.get("accepted", False))
        overall_score = _extract_score(score_report)

        continuity_update = {}
        committed_to_memory = False

        if accepted and overall_score >= score_commit_threshold:
            try:
                continuity_update = continuity_manager.update_after_generation(
                    scene_packet=scene_packet,
                    generation_result=generation_result,
                    score_report=score_report,
                    drift_report=drift_report,
                    prompt_bundle=prompt_bundle,
                )
                continuity_update = dict(continuity_update or {})
                committed_to_memory = True
            except TypeError:
                try:
                    continuity_update = continuity_manager.update_after_generation(
                        scene_packet,
                        generation_result,
                        score_report,
                        drift_report,
                    )
                    continuity_update = dict(continuity_update or {})
                    committed_to_memory = True
                except Exception as e:
                    continuity_update = {
                        "ok": False,
                        "error": f"continuity update failed: {e}",
                    }
                    committed_to_memory = False
            except Exception as e:
                continuity_update = {
                    "ok": False,
                    "error": f"continuity update failed: {e}",
                }
                committed_to_memory = False
        else:
            continuity_update = {
                "ok": True,
                "skipped": True,
                "reason": "scene_not_committed_to_memory",
            }

        invalidated_now: List[str] = []
        if dependency_aware_repair:
            invalidated_now = _compute_downstream_invalidation(
                current_scene_packet=scene_packet,
                all_scene_packets=scene_packets,
                accepted=accepted,
                repair_used=bool(scene_output.get("repair_used", False) or repair_debug.get("repair_result_used", False)),
                drift_report=drift_report,
                score_report=score_report,
            )
            for sid in invalidated_now:
                if sid not in invalidated_scene_ids:
                    invalidated_scene_ids.append(sid)
                if sid not in downstream_regeneration_queue:
                    downstream_regeneration_queue.append(sid)

        scene_record = {
            "ok": bool(scene_output.get("ok", False)),
            "accepted": accepted,
            "scene_id": scene_id,
            "scene_index": scene_index,
            "scene_packet": scene_packet,
            "continuity_package": continuity_package,
            "prompt_bundle": prompt_bundle,
            "generation_result": generation_result,
            "score_report": score_report,
            "drift_report": drift_report,
            "retry_count": int(scene_output.get("retry_count", 0) or 0),
            "repair_used": bool(scene_output.get("repair_used", False) or repair_debug.get("repair_result_used", False)),
            "backend_used": _safe_text(scene_output.get("backend_used", "")),
            "attempts": _safe_list(scene_output.get("attempts", [])),
            "continuity_update": continuity_update,
            "committed_to_memory": committed_to_memory,
            "invalidated_downstream_scene_ids": invalidated_now,
            "repair_debug": repair_debug,
        }

        scene_results.append(scene_record)
        if accepted:
            accepted_scene_results.append(scene_record)
        else:
            failed_scene_results.append(scene_record)

        _write_scene_debug(scenes_debug_dir, scene_id, scene_record)

        if hard_fail_stop and not accepted:
            break

    memory_state = _extract_manager_state(continuity_manager, "memory")
    reference_bank_state = _extract_manager_state(continuity_manager, "reference_bank")

    pipeline_summary = _build_pipeline_summary(
        scene_results=scene_results,
        accepted_scene_results=accepted_scene_results,
        failed_scene_results=failed_scene_results,
        downstream_regeneration_queue=downstream_regeneration_queue,
        invalidated_scene_ids=invalidated_scene_ids,
    )

    output = {
        "ok": len(scene_results) > 0 and len(failed_scene_results) == 0,
        "scene_results": scene_results,
        "accepted_scene_results": accepted_scene_results,
        "failed_scene_results": failed_scene_results,
        "generation_debug": {
            "output_dir": output_dir,
            "debug_dir": pipeline_debug_dir,
            "scenes_debug_dir": scenes_debug_dir,
            "downstream_regeneration_queue": downstream_regeneration_queue,
            "invalidated_scene_ids": invalidated_scene_ids,
        },
        "memory_state": memory_state,
        "reference_bank_state": reference_bank_state,
        "pipeline_summary": pipeline_summary,
    }

    _write_json(os.path.join(output_dir, "generation_pipeline_output.json"), output)
    return output


def _scene_id(scene_packet: Dict[str, Any], scene_index: int) -> str:
    scene_packet = _safe_dict(scene_packet)
    sid = _safe_text(scene_packet.get("scene_id", ""))
    if sid:
        return sid
    sid = _safe_text(scene_packet.get("id", ""))
    if sid:
        return sid
    return f"scene_{scene_index + 1:03d}"


def _extract_score(score_report: Dict[str, Any]) -> float:
    score_report = _safe_dict(score_report)
    if "overall_score" in score_report:
        return _safe_float(score_report.get("overall_score", 0.0), 0.0)
    if "score" in score_report:
        return _safe_float(score_report.get("score", 0.0), 0.0)
    return 0.0


def _severity_rank(severity: str) -> int:
    severity = _safe_text(severity).lower()
    if severity == "none":
        return 0
    if severity == "low":
        return 1
    if severity == "medium":
        return 2
    if severity == "high":
        return 3
    return 2


def _compute_downstream_invalidation(
    current_scene_packet: Dict[str, Any],
    all_scene_packets: List[Dict[str, Any]],
    accepted: bool,
    repair_used: bool,
    drift_report: Dict[str, Any],
    score_report: Dict[str, Any],
) -> List[str]:
    current_scene_packet = _safe_dict(current_scene_packet)
    drift_report = _safe_dict(drift_report)

    if not accepted:
        return []

    drift_flags = [str(x).lower() for x in _safe_list(drift_report.get("drift_flags", []))]
    severity = _severity_rank(drift_report.get("severity", "low"))

    needs_downstream_recheck = False

    if repair_used:
        needs_downstream_recheck = True

    if severity >= 2 and any(
        flag in drift_flags
        for flag in ["identity_drift", "location_drift", "transition_break", "transition_drift"]
    ):
        needs_downstream_recheck = True

    if not needs_downstream_recheck:
        return []

    current_id = _safe_text(current_scene_packet.get("scene_id", ""))
    if not current_id:
        return []

    invalidated = []
    seen_current = False

    for packet in all_scene_packets:
        packet = _safe_dict(packet)
        sid = _safe_text(packet.get("scene_id", ""))

        if sid == current_id:
            seen_current = True
            continue

        if not seen_current:
            continue

        dependent = bool(packet.get("dependent_on_previous", False))
        if dependent and sid:
            invalidated.append(sid)

    return invalidated


def _build_prepare_failure_result(
    scene_packet: Dict[str, Any],
    scene_id: str,
    error_text: str,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "accepted": False,
        "scene_id": scene_id,
        "scene_packet": scene_packet,
        "continuity_package": {},
        "prompt_bundle": {},
        "generation_result": {},
        "score_report": {
            "ok": False,
            "overall_score": 0.0,
            "score": 0.0,
            "accepted": False,
            "hard_fail": True,
        },
        "drift_report": {
            "drift_flags": ["prepare_failure"],
            "severity": "high",
        },
        "retry_count": 0,
        "repair_used": False,
        "backend_used": "",
        "attempts": [],
        "continuity_update": {
            "ok": False,
            "error": error_text,
        },
        "committed_to_memory": False,
        "invalidated_downstream_scene_ids": [],
        "repair_debug": {
            "repair_engine_enabled": False,
            "repair_attempted": False,
            "repair_result_used": False,
            "repair_engine_output": {},
        },
        "error": error_text,
    }


def _extract_manager_state(manager: Any, attr_name: str) -> Dict[str, Any]:
    if manager is None:
        return {}
    state_obj = getattr(manager, attr_name, None)
    if state_obj is None:
        return {}

    if hasattr(state_obj, "to_dict"):
        try:
            return dict(state_obj.to_dict() or {})
        except Exception:
            return {}

    if isinstance(state_obj, dict):
        return dict(state_obj)

    if hasattr(state_obj, "__dict__"):
        try:
            return copy.deepcopy(vars(state_obj))
        except Exception:
            return {}

    return {}


def _write_scene_debug(base_dir: str, scene_id: str, scene_record: Dict[str, Any]) -> None:
    try:
        _write_json(os.path.join(base_dir, f"{scene_id}.json"), scene_record)
    except Exception:
        pass


def _build_pipeline_summary(
    scene_results: List[Dict[str, Any]],
    accepted_scene_results: List[Dict[str, Any]],
    failed_scene_results: List[Dict[str, Any]],
    downstream_regeneration_queue: List[str],
    invalidated_scene_ids: List[str],
) -> Dict[str, Any]:
    backend_counts: Dict[str, int] = {}
    total_score = 0.0
    scored_count = 0
    repaired_count = 0

    for item in scene_results:
        backend = _safe_text(item.get("backend_used", ""))
        if backend:
            backend_counts[backend] = backend_counts.get(backend, 0) + 1

        score = _extract_score(item.get("score_report", {}))
        total_score += score
        scored_count += 1

        if bool(item.get("repair_used", False)):
            repaired_count += 1

    return {
        "num_scenes": len(scene_results),
        "num_accepted": len(accepted_scene_results),
        "num_failed": len(failed_scene_results),
        "accept_rate": (len(accepted_scene_results) / len(scene_results)) if scene_results else 0.0,
        "avg_score": (total_score / scored_count) if scored_count > 0 else 0.0,
        "backend_counts": backend_counts,
        "num_repaired": repaired_count,
        "downstream_regeneration_queue": downstream_regeneration_queue,
        "invalidated_scene_ids": invalidated_scene_ids,
    }


def _build_scorer_from_config(config: Dict[str, Any]):
    try:
        from src.continuity.consistency_scorer import ConsistencyScorer
        return ConsistencyScorer(config)
    except Exception:
        pass

    try:
        from src.continuity.consistency_scorer import ContinuityConsistencyScorer
        return ContinuityConsistencyScorer(config)
    except Exception:
        pass

    return None


def _build_repair_backend_from_config(config: Dict[str, Any]):
    repair_cfg = _safe_dict(config.get("repair", {}))
    backend_name = _safe_text(repair_cfg.get("backend", ""))

    if not backend_name:
        return None

    repair_backend_cfg = copy.deepcopy(config)
    repair_backend_cfg["backend"] = backend_name
    return build_backend(repair_backend_cfg)