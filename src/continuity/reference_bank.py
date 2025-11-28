# src/continuity/reference_bank.py
from __future__ import annotations

import copy
from typing import Dict, Any, List


class ReferenceBank:
    """
    Lightweight reference bank for continuity-guided generation.

    Purpose:
    - keep the best reusable visual references across accepted scenes
    - organize references by role:
        character refs
        location ref
        prop refs
    - help backend choose strong identity / location / transition / prop anchors

    Design notes:
    - stores symbolic reference entries with score and path
    - later can be extended to:
        ranked retrieval
        embedding similarity
        multiple refs per entity with recency decay
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        ref_cfg = self._safe_dict(
            continuity_cfg.get("reference_bank", self.config.get("reference_bank", {}))
        )

        self.max_character_refs_per_name = int(ref_cfg.get("max_character_refs_per_name", 3))
        self.max_prop_refs_per_name = int(ref_cfg.get("max_prop_refs_per_name", 3))

        # storage
        self.character_refs: List[Dict[str, Any]] = []
        self.location_ref: Dict[str, Any] = {}
        self.prop_refs: List[Dict[str, Any]] = []

        self.last_update: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, Any]) -> None:
        """
        Update bank from accepted scene output.

        Expected payload:
        {
            "scene_id": "...",
            "selected_keyframe": "...",
            "character_refs": [...],
            "location_ref": {...},
            "prop_refs": [...],
        }
        """
        payload = self._safe_dict(payload)

        for item in self._safe_list(payload.get("character_refs", [])):
            self._merge_character_ref(item)

        location_ref = self._safe_dict(payload.get("location_ref", {}))
        if location_ref:
            self._merge_location_ref(location_ref)

        for item in self._safe_list(payload.get("prop_refs", [])):
            self._merge_prop_ref(item)

        self.last_update = copy.deepcopy(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "character_refs": copy.deepcopy(self.character_refs),
            "location_ref": copy.deepcopy(self.location_ref),
            "prop_refs": copy.deepcopy(self.prop_refs),
            "last_update": copy.deepcopy(self.last_update),
        }

    # ------------------------------------------------------------------
    # character refs
    # ------------------------------------------------------------------

    def _merge_character_ref(self, item: Dict[str, Any]) -> None:
        item = self._normalize_character_ref(item)
        if not item["name"] or not item["path"]:
            return

        same_name_refs = [
            ref for ref in self.character_refs
            if self._safe_text(ref.get("name", "")).lower() == item["name"].lower()
        ]

        # If same path already exists, keep better score / fresher reason.
        replaced_existing = False
        for idx, ref in enumerate(self.character_refs):
            if (
                self._safe_text(ref.get("name", "")).lower() == item["name"].lower()
                and self._safe_text(ref.get("path", "")) == item["path"]
            ):
                if self._safe_float(item.get("score", 0.0), 0.0) >= self._safe_float(ref.get("score", 0.0), 0.0):
                    self.character_refs[idx] = item
                replaced_existing = True
                break

        if not replaced_existing:
            self.character_refs.append(item)

        # Keep strongest refs first, capped per character name.
        self.character_refs = self._prune_character_refs(self.character_refs)

    def _prune_character_refs(self, refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs = [self._normalize_character_ref(x) for x in refs if self._safe_dict(x)]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ref in refs:
            name = ref["name"].lower()
            grouped.setdefault(name, []).append(ref)

        pruned: List[Dict[str, Any]] = []
        for _, items in grouped.items():
            items.sort(
                key=lambda x: (
                    self._safe_float(x.get("score", 0.0), 0.0),
                    self._safe_text(x.get("scene_id", "")),
                ),
                reverse=True,
            )
            pruned.extend(items[: self.max_character_refs_per_name])

        # stable global ordering by score
        pruned.sort(
            key=lambda x: (
                self._safe_float(x.get("score", 0.0), 0.0),
                self._safe_text(x.get("name", "")),
            ),
            reverse=True,
        )
        return pruned

    def _normalize_character_ref(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = self._safe_dict(item)
        return {
            "name": self._safe_text(item.get("name", "")),
            "path": self._safe_text(item.get("path", "")),
            "score": self._safe_float(item.get("score", 0.0), 0.0),
            "reason": self._safe_text(item.get("reason", "")),
            "scene_id": self._safe_text(item.get("scene_id", "")),
        }

    # ------------------------------------------------------------------
    # location ref
    # ------------------------------------------------------------------

    def _merge_location_ref(self, item: Dict[str, Any]) -> None:
        item = self._normalize_location_ref(item)
        if not item["path"]:
            return

        if not self.location_ref:
            self.location_ref = item
            return

        existing_score = self._safe_float(self.location_ref.get("score", 0.0), 0.0)
        incoming_score = self._safe_float(item.get("score", 0.0), 0.0)

        # replace only if stronger
        if incoming_score >= existing_score:
            self.location_ref = item

    def _normalize_location_ref(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = self._safe_dict(item)
        return {
            "name": self._safe_text(item.get("name", "")),
            "path": self._safe_text(item.get("path", "")),
            "score": self._safe_float(item.get("score", 0.0), 0.0),
            "reason": self._safe_text(item.get("reason", "")),
            "scene_id": self._safe_text(item.get("scene_id", "")),
        }

    # ------------------------------------------------------------------
    # prop refs
    # ------------------------------------------------------------------

    def _merge_prop_ref(self, item: Dict[str, Any]) -> None:
        item = self._normalize_prop_ref(item)
        if not item["name"] or not item["path"]:
            return

        replaced_existing = False
        for idx, ref in enumerate(self.prop_refs):
            if (
                self._safe_text(ref.get("name", "")).lower() == item["name"].lower()
                and self._safe_text(ref.get("path", "")) == item["path"]
            ):
                if self._safe_float(item.get("score", 0.0), 0.0) >= self._safe_float(ref.get("score", 0.0), 0.0):
                    self.prop_refs[idx] = item
                replaced_existing = True
                break

        if not replaced_existing:
            self.prop_refs.append(item)

        self.prop_refs = self._prune_prop_refs(self.prop_refs)

    def _prune_prop_refs(self, refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refs = [self._normalize_prop_ref(x) for x in refs if self._safe_dict(x)]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ref in refs:
            name = ref["name"].lower()
            grouped.setdefault(name, []).append(ref)

        pruned: List[Dict[str, Any]] = []
        for _, items in grouped.items():
            items.sort(
                key=lambda x: (
                    self._safe_float(x.get("score", 0.0), 0.0),
                    self._safe_text(x.get("scene_id", "")),
                ),
                reverse=True,
            )
            pruned.extend(items[: self.max_prop_refs_per_name])

        pruned.sort(
            key=lambda x: (
                self._safe_float(x.get("score", 0.0), 0.0),
                self._safe_text(x.get("name", "")),
            ),
            reverse=True,
        )
        return pruned

    def _normalize_prop_ref(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item = self._safe_dict(item)
        return {
            "name": self._safe_text(item.get("name", "")),
            "path": self._safe_text(item.get("path", "")),
            "score": self._safe_float(item.get("score", 0.0), 0.0),
            "reason": self._safe_text(item.get("reason", "")),
            "scene_id": self._safe_text(item.get("scene_id", "")),
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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