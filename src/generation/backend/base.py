# src/generation/backend/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseGenerationBackend(ABC):
    """
    Base interface for all generation backends.

    Every backend should support:
    - generate(prompt_bundle)
    - generate_repair(prompt_bundle)  [default: falls back to generate]
    - unload()                        [default: no-op]

    Expected prompt_bundle:
    {
        "scene_id": str,
        "prompt": str,                 # backward-compatible
        "analysis_prompt": str,
        "story_prompt": str,
        "continuity_prompt": str,
        "model_prompt": str,
        "positive_prompt": str,
        "negative_prompt": str,
        "repair_prompt": str,
        "prompt_contract": dict,
        "prompt_metadata": dict,
        "reference_bundle": dict,
        "control_weights": dict,
        "generation_params": dict,
        "scene_packet": dict,
        ...
    }

    Expected backend output:
    {
        "ok": bool,
        "scene_id": str,
        "video_path": str,
        "output_video_path": str,
        "keyframe_path": str,
        "selected_keyframe_path": str,
        "first_frame_path": str,
        "middle_frame_path": str,
        "last_frame_path": str,
        "best_keyframe_path": str,
        "frames": list,                # optional
        "metadata": dict,              # important
        "error": str,                  # optional
    }
    """

    @abstractmethod
    def generate(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a normal scene result from prompt bundle.
        Must be implemented by every backend.
        """
        raise NotImplementedError

    def generate_repair(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair generation path.
        Default behavior falls back to normal generation unless overridden.
        """
        return self.generate(prompt_bundle)

    def unload(self) -> None:
        """
        Optional cleanup hook.
        Backends can override this to free GPU memory / close resources.
        """
        return None