# src/video/frames.py
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.utils.io import ensure_dir, write_json


def is_valid_path(path: str) -> bool:
    return bool(path and Path(path).exists())


def build_frame_record(
    scene_id: int,
    frame_path: str,
    frame_type: str,
    score: float = 0.0,
    tags: List[str] = None
) -> Dict[str, Any]:
    if tags is None:
        tags = []

    return {
        "scene_id": scene_id,
        "frame_path": frame_path,
        "frame_type": frame_type,
        "score": float(score),
        "tags": tags
    }


def collect_generation_frames(generation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts frame artifacts from generation_result.
    """
    scene_id = generation_result.get("scene_id", -1)
    out: List[Dict[str, Any]] = []

    mapping = [
        ("last_frame_path", "last_frame", generation_result.get("last_frame_score", 0.0), ["transition"]),
        ("keyframe_path", "keyframe", generation_result.get("keyframe_score", 0.0), ["general"]),
        ("character_frame_path", "character_frame", generation_result.get("character_frame_score", 0.0), ["character"]),
        ("location_frame_path", "location_frame", generation_result.get("location_frame_score", 0.0), ["location"]),
        ("object_frame_path", "object_frame", generation_result.get("object_frame_score", 0.0), ["object"]),
    ]

    for key, frame_type, score, tags in mapping:
        frame_path = generation_result.get(key, "")
        if frame_path:
            out.append(
                build_frame_record(
                    scene_id=scene_id,
                    frame_path=frame_path,
                    frame_type=frame_type,
                    score=score,
                    tags=tags
                )
            )

    return out


def select_best_frame_by_type(
    frame_records: List[Dict[str, Any]],
    frame_type: str
) -> Optional[Dict[str, Any]]:
    candidates = [x for x in frame_records if x.get("frame_type") == frame_type]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[0]


def save_frame_manifest(
    generation_result: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Save all discovered frame artifacts for one scene.
    """
    ensure_dir(output_dir)

    frame_records = collect_generation_frames(generation_result)

    manifest = {
        "scene_id": generation_result.get("scene_id", -1),
        "num_frames": len(frame_records),
        "frames": frame_records,
        "best_keyframe": select_best_frame_by_type(frame_records, "keyframe"),
        "best_character_frame": select_best_frame_by_type(frame_records, "character_frame"),
        "best_location_frame": select_best_frame_by_type(frame_records, "location_frame"),
        "best_object_frame": select_best_frame_by_type(frame_records, "object_frame"),
        "last_frame": select_best_frame_by_type(frame_records, "last_frame"),
    }

    write_json(f"{output_dir}/frame_manifest.json", manifest)
    return manifest