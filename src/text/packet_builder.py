# src/text/packet_builder.py
from typing import Dict, Any, List
from src.text.scene_packet import ScenePacket, ContinuityRequirements


def infer_scene_role(scene: Dict[str, Any]) -> str:
    """
    Simple heuristic for scene role.
    You can improve later.
    """
    text = scene.get("scene_text", "").lower()
    action = scene.get("core_action", "").lower()

    if "enters" in text or "arrives" in text or "establish" in action:
        return "establishing"
    if "close" in scene.get("camera_style", "").lower():
        return "closeup"
    if "runs" in text or "fights" in text or "moves" in action:
        return "action"
    if "sits" in text or "looks" in text or "reads" in text:
        return "calm"
    return "action"


def infer_duration(scene_role: str) -> float:
    """
    Default duration heuristic.
    """
    if scene_role == "establishing":
        return 4.5
    if scene_role == "closeup":
        return 3.0
    if scene_role == "calm":
        return 4.0
    return 4.0


def build_continuity_requirements(dep_types: List[str]) -> ContinuityRequirements:
    req = ContinuityRequirements()

    if "character" in dep_types:
        req.preserve_character = True
    if "location" in dep_types:
        req.preserve_location = True
    if "object" in dep_types:
        req.preserve_object_state = True
    if "style" in dep_types:
        req.preserve_style = True
    if "camera" in dep_types:
        req.preserve_camera = True
    if "action" in dep_types or "time" in dep_types:
        req.preserve_action = True

    return req


def build_dependency_lookup(dependencies: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Key by to_scene_id, because dependency belongs to current scene relative to previous scene.
    """
    lookup = {}
    for dep in dependencies:
        to_scene = dep.get("to_scene_id")
        if isinstance(to_scene, int):
            lookup[to_scene] = dep
    return lookup


def build_scene_packets(
    scenes_obj: Dict[str, Any],
    dependencies_obj: Dict[str, Any]
) -> Dict[str, Any]:
    scenes = scenes_obj.get("scenes", [])
    dependencies = dependencies_obj.get("dependencies", [])

    dep_lookup = build_dependency_lookup(dependencies)

    packets: List[ScenePacket] = []

    for scene in scenes:
        scene_id = scene.get("scene_id", 0)
        dep = dep_lookup.get(scene_id, None)

        dependent_on_previous = False
        dependency_type: List[str] = []
        dependency_strength = 0.0

        if dep is not None:
            dependent_on_previous = bool(dep.get("dependent", False))
            dependency_type = list(dep.get("dependency_type", []) or [])
            dependency_strength = float(dep.get("confidence", 0.0) or 0.0)

        scene_role = infer_scene_role(scene)
        duration_s = infer_duration(scene_role)
        continuity_requirements = build_continuity_requirements(dependency_type)

        packet = ScenePacket(
            scene_id=scene_id,
            scene_text=scene.get("scene_text", ""),
            dependent_on_previous=dependent_on_previous,
            dependency_type=dependency_type,
            dependency_strength=dependency_strength,
            characters=list(scene.get("characters", []) or []),
            location=scene.get("location", "") or "",
            time_hint=scene.get("time_hint", "") or "",
            core_action=scene.get("core_action", "") or "",
            camera_style=scene.get("camera_style", "") or "",
            duration_s=duration_s,
            scene_role=scene_role,
            continuity_requirements=continuity_requirements,
            metadata={
                "start_char_index": scene.get("start_char_index", -1),
                "end_char_index": scene.get("end_char_index", -1),
                "new_scene_reason": scene.get("new_scene_reason", "")
            }
        )

        packets.append(packet)

    return {
        "scene_packets": [p.to_dict() for p in packets],
        "num_scene_packets": len(packets)
    }