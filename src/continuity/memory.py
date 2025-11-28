# src/continuity/memory.py
from __future__ import annotations

import copy
from typing import Dict, Any, List


class ContinuityMemory:
    """
    Lightweight persistent continuity memory.

    Purpose:
    - store continuity state across accepted scenes
    - keep track of:
        characters
        location
        props
        style
        recent scenes
        reference frames
    - provide a stable interface for ContinuityManager

    Design notes:
    - this is still a symbolic memory layer
    - later you can extend it into:
        entity memory graph
        provisional / committed memory
        embedding-backed identity memory
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        memory_cfg = self._safe_dict(
            continuity_cfg.get("memory", self.config.get("memory", {}))
        )

        self.max_recent_scenes = int(memory_cfg.get("max_recent_scenes", 20))
        self.max_reference_frames = int(memory_cfg.get("max_reference_frames", 20))

        self.characters: List[Dict[str, Any]] = []
        self.location: Dict[str, Any] = {}
        self.props: List[Dict[str, Any]] = []
        self.style: Dict[str, Any] = {}

        self.recent_scenes: List[Dict[str, Any]] = []
        self.reference_frames: List[Dict[str, Any]] = []

        self.last_scene_id: str = ""
        self.last_update: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, Any]) -> None:
        """
        Update memory from accepted scene output.

        Expected payload:
        {
            "scene_id": "...",
            "characters": [...],
            "location": {...},
            "props": [...],
            "style": {...},
            "selected_keyframe": "...",
            "score": 0.82,
        }
        """
        payload = self._safe_dict(payload)

        scene_id = self._safe_text(payload.get("scene_id", ""))
        characters = self._safe_list(payload.get("characters", []))
        location = self._safe_dict(payload.get("location", {}))
        props = self._safe_list(payload.get("props", []))
        style = self._safe_dict(payload.get("style", {}))
        selected_keyframe = self._safe_text(payload.get("selected_keyframe", ""))
        score = self._safe_float(payload.get("score", 0.0), 0.0)

        # merge character state
        for ch in characters:
            self._merge_character(ch)

        # merge location
        if location:
            self.location = self._merge_location(self.location, location)

        # merge props
        self.props = self._merge_props(self.props, props)

        # merge style
        if style:
            self.style = self._merge_style(self.style, style)

        # recent scene trace
        if scene_id:
            self._push_recent_scene({
                "scene_id": scene_id,
                "characters": copy.deepcopy(characters),
                "location": copy.deepcopy(location),
                "props": copy.deepcopy(props),
                "style": copy.deepcopy(style),
                "selected_keyframe": selected_keyframe,
                "score": score,
            })

        # reference frame memory
        if scene_id and selected_keyframe:
            self._push_reference_frame({
                "scene_id": scene_id,
                "path": selected_keyframe,
                "score": score,
            })

        self.last_scene_id = scene_id
        self.last_update = {
            "scene_id": scene_id,
            "characters": copy.deepcopy(characters),
            "location": copy.deepcopy(location),
            "props": copy.deepcopy(props),
            "style": copy.deepcopy(style),
            "selected_keyframe": selected_keyframe,
            "score": score,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "characters": copy.deepcopy(self.characters),
            "location": copy.deepcopy(self.location),
            "props": copy.deepcopy(self.props),
            "style": copy.deepcopy(self.style),
            "recent_scenes": copy.deepcopy(self.recent_scenes),
            "reference_frames": copy.deepcopy(self.reference_frames),
            "last_scene_id": self.last_scene_id,
            "last_update": copy.deepcopy(self.last_update),
        }

    # ------------------------------------------------------------------
    # character memory
    # ------------------------------------------------------------------

    def _merge_character(self, incoming: Dict[str, Any]) -> None:
        incoming = self._safe_dict(incoming)
        if not incoming:
            return

        incoming_name = self._safe_text(incoming.get("name", ""))
        target_idx = self._find_character_index(incoming_name, incoming)

        if target_idx is None:
            self.characters.append(self._normalize_character(incoming))
            return

        existing = self._safe_dict(self.characters[target_idx])
        merged = self._normalize_character(existing)

        # merge textual scalar fields
        for key in ["name", "face_desc", "hair", "pose", "emotion", "action"]:
            new_val = self._safe_text(incoming.get(key, ""))
            if new_val:
                merged[key] = new_val

        # merge list fields
        for key in ["clothing", "accessories", "aliases"]:
            merged[key] = self._dedupe_keep_order(
                self._safe_list(merged.get(key, []))
                + self._safe_list(incoming.get(key, []))
            )

        self.characters[target_idx] = merged

    def _find_character_index(self, incoming_name: str, incoming: Dict[str, Any]) -> int | None:
        incoming_name_low = incoming_name.lower()

        # match by name
        if incoming_name_low:
            for idx, ch in enumerate(self.characters):
                ch = self._safe_dict(ch)
                existing_name = self._safe_text(ch.get("name", "")).lower()
                aliases = [self._safe_text(x).lower() for x in self._safe_list(ch.get("aliases", []))]
                if incoming_name_low == existing_name or incoming_name_low in aliases:
                    return idx

        # fallback soft match by face_desc + hair if available
        incoming_face = self._safe_text(incoming.get("face_desc", "")).lower()
        incoming_hair = self._safe_text(incoming.get("hair", "")).lower()

        if incoming_face or incoming_hair:
            for idx, ch in enumerate(self.characters):
                ch = self._safe_dict(ch)
                face = self._safe_text(ch.get("face_desc", "")).lower()
                hair = self._safe_text(ch.get("hair", "")).lower()

                face_match = (incoming_face and face and (incoming_face == face or incoming_face in face or face in incoming_face))
                hair_match = (incoming_hair and hair and (incoming_hair == hair or incoming_hair in hair or hair in incoming_hair))

                if face_match or hair_match:
                    return idx

        return None

    def _normalize_character(self, ch: Dict[str, Any]) -> Dict[str, Any]:
        ch = self._safe_dict(ch)

        return {
            "name": self._safe_text(ch.get("name", "")),
            "face_desc": self._safe_text(ch.get("face_desc", "")),
            "hair": self._safe_text(ch.get("hair", "")),
            "clothing": self._dedupe_keep_order(self._safe_list(ch.get("clothing", []))),
            "accessories": self._dedupe_keep_order(self._safe_list(ch.get("accessories", []))),
            "pose": self._safe_text(ch.get("pose", "")),
            "emotion": self._safe_text(ch.get("emotion", "")),
            "action": self._safe_text(ch.get("action", "")),
            "aliases": self._dedupe_keep_order(self._safe_list(ch.get("aliases", []))),
        }

    # ------------------------------------------------------------------
    # location / props / style memory
    # ------------------------------------------------------------------

    def _merge_location(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        existing = self._safe_dict(existing)
        incoming = self._safe_dict(incoming)

        merged = copy.deepcopy(existing)

        for key in ["name", "category", "lighting", "weather", "time_of_day", "atmosphere"]:
            val = self._safe_text(incoming.get(key, ""))
            if val:
                merged[key] = val

        merged["anchors"] = self._dedupe_keep_order(
            self._safe_list(existing.get("anchors", []))
            + self._safe_list(incoming.get("anchors", []))
        )

        return merged

    def _merge_props(
        self,
        existing_props: List[Dict[str, Any]],
        incoming_props: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        existing_props = [self._safe_dict(x) for x in existing_props]
        incoming_props = [self._safe_dict(x) for x in incoming_props]

        merged = copy.deepcopy(existing_props)

        for prop in incoming_props:
            name = self._safe_text(prop.get("name", ""))
            if not name:
                continue

            idx = self._find_prop_index(merged, name)
            if idx is None:
                merged.append({
                    "name": name,
                    "holder": self._safe_text(prop.get("holder", "")),
                    "status": self._safe_text(prop.get("status", "")),
                })
            else:
                if self._safe_text(prop.get("holder", "")):
                    merged[idx]["holder"] = self._safe_text(prop.get("holder", ""))
                if self._safe_text(prop.get("status", "")):
                    merged[idx]["status"] = self._safe_text(prop.get("status", ""))

        return merged

    def _find_prop_index(self, props: List[Dict[str, Any]], name: str) -> int | None:
        name_low = self._safe_text(name).lower()
        for idx, prop in enumerate(props):
            if self._safe_text(prop.get("name", "")).lower() == name_low:
                return idx
        return None

    def _merge_style(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        existing = self._safe_dict(existing)
        incoming = self._safe_dict(incoming)

        merged = copy.deepcopy(existing)

        for key in ["visual_style", "color_tone", "shot_type", "camera_angle", "camera_motion", "mood"]:
            val = self._safe_text(incoming.get(key, ""))
            if val:
                merged[key] = val

        return merged

    # ------------------------------------------------------------------
    # recent scenes / reference frames
    # ------------------------------------------------------------------

    def _push_recent_scene(self, scene_info: Dict[str, Any]) -> None:
        self.recent_scenes.append(copy.deepcopy(scene_info))
        if len(self.recent_scenes) > self.max_recent_scenes:
            self.recent_scenes = self.recent_scenes[-self.max_recent_scenes :]

    def _push_reference_frame(self, ref_info: Dict[str, Any]) -> None:
        self.reference_frames.append(copy.deepcopy(ref_info))
        if len(self.reference_frames) > self.max_reference_frames:
            self.reference_frames = self.reference_frames[-self.max_reference_frames :]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _dedupe_keep_order(self, items: List[Any]) -> List[str]:
        out = []
        seen = set()

        for item in items or []:
            text = self._safe_text(item)
            if not text:
                continue
            low = text.lower()
            if low not in seen:
                seen.add(low)
                out.append(text)

        return out

    def _safe_text(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {}

    def _safe_list(self, x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return []

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


# backward-compatible alias
Memory = ContinuityMemory