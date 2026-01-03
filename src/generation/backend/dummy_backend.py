from typing import Dict, Any

from src.generation.backend.base import BaseGenerationBackend


class DummyBackend(BaseGenerationBackend):
    def __init__(self, output_dir: str = "outputs/dummy"):
        self.output_dir = output_dir

    def generate(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = prompt_bundle.get("scene_id", "scene_unknown")
        prompt = prompt_bundle.get("prompt", "")
        same_constraints = prompt_bundle.get("same_constraints", {})
        change_constraints = prompt_bundle.get("change_constraints", {})

        generated_metadata = {
            "characters": [],
            "location": {},
            "style": {},
            "props": [],
        }

        for ch in same_constraints.get("characters", []):
            generated_metadata["characters"].append({
                "name": ch.get("name", ""),
                "clothing": ch.get("clothing", []),
                "accessories": ch.get("accessories", []),
            })

        location = same_constraints.get("location", {})
        if location:
            generated_metadata["location"] = {
                "name": location.get("name", ""),
                "anchors": location.get("anchors", []),
            }

        style = same_constraints.get("style", {})
        if style:
            generated_metadata["style"] = {
                "visual_style": style.get("visual_style", ""),
                "color_tone": style.get("color_tone", ""),
                "shot_type": style.get("shot_type", ""),
                "camera_angle": style.get("camera_angle", ""),
                "camera_motion": style.get("camera_motion", ""),
                "mood": style.get("mood", ""),
            }

        generated_metadata["props"] = [
            p.get("name", "") for p in same_constraints.get("props", [])
        ]

        return {
            "ok": True,
            "output_path": f"{self.output_dir}/{scene_id}.mp4",
            "selected_keyframe": f"{self.output_dir}/{scene_id}_keyframe.png",
            "quality_score": 0.82,
            "generated_metadata": generated_metadata,
            "transition_metadata": {
                "smooth": True
            },
            "used_prompt": prompt,
            "used_control_weights": prompt_bundle.get("control_weights", {}),
            "used_change_constraints": change_constraints,
        }