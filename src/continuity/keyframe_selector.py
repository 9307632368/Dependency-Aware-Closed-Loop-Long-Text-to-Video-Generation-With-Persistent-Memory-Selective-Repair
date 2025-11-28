# src/continuity/keyframe_selector.py
from __future__ import annotations

import os
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter


class KeyframeRoleSelector:
    """
    Role-based keyframe selector for long text-to-video continuity.

    It selects different reference frames for:
    - identity
    - location
    - prop
    - transition
    - style

    Current version is lightweight and practical:
    - no heavy ML dependency required
    - uses image sharpness, brightness, color, frame position, and text hints
    - later we can upgrade this with CLIP / GroundingDINO / face detector
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        cfg = self.config.get("keyframe_selector", {})
        self.sample_limit = int(cfg.get("sample_limit", 24))
        self.min_sharpness = float(cfg.get("min_sharpness", 4.0))
        self.min_detail = float(cfg.get("min_detail", 8.0))

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def select_from_frames(
        self,
        frames: List[Any],
        scene_id: str,
        output_dir: str,
        scene_packet: Dict[str, Any] = None,
        prompt_bundle: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        scene_packet = scene_packet or {}
        prompt_bundle = prompt_bundle or {}

        os.makedirs(output_dir, exist_ok=True)

        pil_frames = self._to_pil_frames(frames)
        if not pil_frames:
            return self._empty_result()

        sampled = self._sample_frames(pil_frames)
        scored = []

        for item in sampled:
            idx = item["index"]
            frame = item["frame"]

            stats = self._frame_stats(frame)
            scores = self._role_scores(
                frame_index=idx,
                total_frames=len(pil_frames),
                stats=stats,
                scene_packet=scene_packet,
                prompt_bundle=prompt_bundle,
            )

            scored.append({
                "index": idx,
                "frame": frame,
                "stats": stats,
                "scores": scores,
            })

        selected = {
            "identity": self._best(scored, "identity"),
            "location": self._best(scored, "location"),
            "prop": self._best(scored, "prop"),
            "transition": self._best(scored, "transition"),
            "style": self._best(scored, "style"),
        }

        paths = {}
        for role, item in selected.items():
            if item is None:
                paths[f"{role}_frame_path"] = ""
                continue

            path = os.path.join(output_dir, f"{scene_id}_{role}_frame.png")
            item["frame"].save(path)
            paths[f"{role}_frame_path"] = path

        middle_item = scored[len(scored) // 2]
        first_item = scored[0]
        last_item = scored[-1]

        first_path = os.path.join(output_dir, f"{scene_id}_first.png")
        middle_path = os.path.join(output_dir, f"{scene_id}_middle.png")
        last_path = os.path.join(output_dir, f"{scene_id}_last.png")

        first_item["frame"].save(first_path)
        middle_item["frame"].save(middle_path)
        last_item["frame"].save(last_path)

        paths["first_frame_path"] = first_path
        paths["middle_frame_path"] = middle_path
        paths["last_frame_path"] = last_path

        paths["best_keyframe_path"] = self._choose_default_best(paths)

        return {
            "ok": True,
            "paths": paths,
            "selected_indices": {
                role: selected[role]["index"] if selected[role] else None
                for role in selected
            },
            "role_scores": {
                role: selected[role]["scores"][role] if selected[role] else 0.0
                for role in selected
            },
            "debug": {
                "num_frames": len(pil_frames),
                "num_sampled": len(scored),
            },
        }

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _role_scores(
        self,
        frame_index: int,
        total_frames: int,
        stats: Dict[str, float],
        scene_packet: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
    ) -> Dict[str, float]:
        pos = frame_index / max(1, total_frames - 1)

        sharp = self._norm(stats["sharpness"], 0, 80)
        detail = self._norm(stats["detail"], 0, 60)
        brightness = self._brightness_quality(stats["brightness"])
        color_balance = self._color_balance_quality(stats["color_balance"])
        edge = self._norm(stats["edge_strength"], 0, 40)

        dependent = bool(scene_packet.get("dependent_on_previous", False))
        same = scene_packet.get("same_as_previous", {}) or {}

        identity_need = 1.0 if same.get("character_identity", False) else 0.4
        location_need = 1.0 if same.get("location", False) else 0.4
        prop_need = 1.0 if same.get("props", False) else 0.4
        style_need = 1.0 if same.get("style", False) else 0.5
        transition_need = 1.0 if dependent else 0.5

        center_bias = 1.0 - abs(pos - 0.50) * 1.2
        center_bias = self._clamp(center_bias)

        early_mid_bias = 1.0 - abs(pos - 0.35) * 1.2
        early_mid_bias = self._clamp(early_mid_bias)

        late_bias = self._clamp((pos - 0.65) / 0.35)
        final_stable_bias = self._clamp((pos - 0.75) / 0.25)

        identity_score = (
            0.30 * sharp +
            0.25 * detail +
            0.20 * brightness +
            0.15 * center_bias +
            0.10 * identity_need
        )

        location_score = (
            0.25 * detail +
            0.25 * color_balance +
            0.20 * brightness +
            0.15 * early_mid_bias +
            0.15 * location_need
        )

        prop_score = (
            0.35 * sharp +
            0.25 * edge +
            0.15 * detail +
            0.15 * center_bias +
            0.10 * prop_need
        )

        transition_score = (
            0.30 * final_stable_bias +
            0.25 * sharp +
            0.20 * detail +
            0.15 * brightness +
            0.10 * transition_need
        )

        style_score = (
            0.30 * color_balance +
            0.25 * brightness +
            0.20 * detail +
            0.15 * center_bias +
            0.10 * style_need
        )

        if stats["sharpness"] < self.min_sharpness:
            identity_score *= 0.55
            prop_score *= 0.55
            transition_score *= 0.60

        if stats["detail"] < self.min_detail:
            location_score *= 0.60
            style_score *= 0.65

        return {
            "identity": self._clamp(identity_score),
            "location": self._clamp(location_score),
            "prop": self._clamp(prop_score),
            "transition": self._clamp(transition_score),
            "style": self._clamp(style_score),
        }

    def _frame_stats(self, frame: Image.Image) -> Dict[str, float]:
        img = frame.convert("RGB").resize((256, 256))
        arr = np.asarray(img).astype(np.float32)

        gray = np.asarray(img.convert("L")).astype(np.float32)

        brightness = float(gray.mean())
        detail = float(gray.std())

        lap = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_arr = np.asarray(lap).astype(np.float32)
        edge_strength = float(edge_arr.mean())

        sharpness = float(edge_arr.var())

        avg = arr.mean(axis=(0, 1))
        color_balance = float(255.0 - (max(avg) - min(avg)))

        return {
            "brightness": brightness,
            "detail": detail,
            "edge_strength": edge_strength,
            "sharpness": sharpness,
            "color_balance": color_balance,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_frames(self, frames: List[Image.Image]) -> List[Dict[str, Any]]:
        n = len(frames)
        if n <= self.sample_limit:
            return [{"index": i, "frame": frames[i]} for i in range(n)]

        indices = np.linspace(0, n - 1, self.sample_limit).astype(int).tolist()
        indices = sorted(set(indices))
        return [{"index": i, "frame": frames[i]} for i in indices]

    def _best(self, scored: List[Dict[str, Any]], role: str) -> Optional[Dict[str, Any]]:
        if not scored:
            return None
        return max(scored, key=lambda x: x["scores"].get(role, 0.0))

    def _choose_default_best(self, paths: Dict[str, str]) -> str:
        for key in [
            "transition_frame_path",
            "identity_frame_path",
            "middle_frame_path",
            "last_frame_path",
            "first_frame_path",
        ]:
            path = paths.get(key, "")
            if path:
                return path
        return ""

    def _to_pil_frames(self, frames: List[Any]) -> List[Image.Image]:
        out = []
        for frame in frames or []:
            img = self._to_pil(frame)
            if img is not None:
                out.append(img)
        return out

    def _to_pil(self, frame: Any) -> Optional[Image.Image]:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        return None

    def _brightness_quality(self, brightness: float) -> float:
        # Best around 120–170, penalize too dark / too bright.
        return self._clamp(1.0 - abs(brightness - 145.0) / 145.0)

    def _color_balance_quality(self, color_balance: float) -> float:
        return self._norm(color_balance, 0, 255)

    def _norm(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return self._clamp((x - lo) / (hi - lo))

    def _clamp(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "paths": {
                "identity_frame_path": "",
                "location_frame_path": "",
                "prop_frame_path": "",
                "transition_frame_path": "",
                "style_frame_path": "",
                "first_frame_path": "",
                "middle_frame_path": "",
                "last_frame_path": "",
                "best_keyframe_path": "",
            },
            "selected_indices": {},
            "role_scores": {},
            "debug": {},
        }


def select_role_keyframes(
    frames: List[Any],
    scene_id: str,
    output_dir: str,
    scene_packet: Dict[str, Any] = None,
    prompt_bundle: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    selector = KeyframeRoleSelector(config=config)
    return selector.select_from_frames(
        frames=frames,
        scene_id=scene_id,
        output_dir=output_dir,
        scene_packet=scene_packet,
        prompt_bundle=prompt_bundle,
    )