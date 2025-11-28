# src/continuity/state_tracker.py
from typing import Dict, Any

from src.continuity.memory import WorldMemory


def extract_scene_state(
    scene_packet: Dict[str, Any],
    generation_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build structured state snapshot from one generated scene.
    """
    return {
        "scene_id": scene_packet.get("scene_id", -1),
        "characters": list(scene_packet.get("characters", []) or []),
        "location": scene_packet.get("location", "") or "",
        "time_hint": scene_packet.get("time_hint", "") or "",
        "core_action": scene_packet.get("core_action", "") or "",
        "camera_style": scene_packet.get("camera_style", "") or "",
        "appearance_text": generation_result.get("appearance_text", "") or "",
        "clothing_text": generation_result.get("clothing_text", "") or "",
        "background_text": generation_result.get("background_text", "") or "",
        "lighting_text": generation_result.get("lighting_text", "") or "",
        "tone_text": generation_result.get("tone_text", "") or "",
        "object_states": generation_result.get("object_states", {}) or {},
        "character_frame_path": generation_result.get("character_frame_path", "") or "",
        "location_frame_path": generation_result.get("location_frame_path", "") or "",
        "object_frame_path": generation_result.get("object_frame_path", "") or "",
    }


def update_world_memory_from_state(
    world_memory: WorldMemory,
    scene_state: Dict[str, Any]
) -> None:
    """
    Apply extracted scene state into world memory.
    """
    world_memory.update_from_scene_packet({
        "scene_id": scene_state.get("scene_id", -1),
        "scene_text": "",
        "characters": scene_state.get("characters", []),
        "location": scene_state.get("location", ""),
        "core_action": scene_state.get("core_action", ""),
        "camera_style": scene_state.get("camera_style", "")
    })

    for char_name in scene_state.get("characters", []):
        world_memory.update_character_appearance(
            character_name=char_name,
            appearance_text=scene_state.get("appearance_text", ""),
            clothing_text=scene_state.get("clothing_text", ""),
            reference_frame_path=scene_state.get("character_frame_path", "")
        )

    world_memory.update_location_state(
        background_text=scene_state.get("background_text", ""),
        lighting_text=scene_state.get("lighting_text", ""),
        tone_text=scene_state.get("tone_text", ""),
        reference_frame_path=scene_state.get("location_frame_path", "")
    )

    object_states = scene_state.get("object_states", {})
    if isinstance(object_states, dict):
        for obj_name, state_text in object_states.items():
            world_memory.update_object_state(
                object_name=str(obj_name),
                state_text=str(state_text),
                reference_frame_path=scene_state.get("object_frame_path", "")
            )