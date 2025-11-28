# src/continuity/extract.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

try:
    from PIL import Image
    import numpy as np
except Exception:  # pragma: no cover
    Image = None
    np = None


class GeneratedStateExtractor:
    """
    First practical generated-state extractor.

    Goal:
    - produce a usable generated_summary from actually generated outputs
    - stay conservative when evidence is weak
    - avoid hallucinating observed state from expected prompt state

    Current version:
    - reads available generated keyframes / frame images
    - performs lightweight visual sanity checks
    - uses prompt/reference context only as weak hints, never as confirmed observed truth
    - returns:
        {
            "characters": [...],
            "location": {...},
            "props": [...],
            "style": {...},
            "source": "visual_heuristic" | "weak_context" | "missing"
        }

    Important:
    - expected prompt content is not copied as observed content
    - if evidence is weak, output remains sparse
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = dict(config or {})

        continuity_cfg = self._safe_dict(self.config.get("continuity", {}))
        extract_cfg = self._safe_dict(
            continuity_cfg.get("extract", self.config.get("extract", {}))
        )

        self.enabled = bool(extract_cfg.get("enabled", True))
        self.sample_resize = int(extract_cfg.get("sample_resize", 256))
        self.min_std_threshold = float(extract_cfg.get("min_std_threshold", 8.0))
        self.max_dominant_ratio = float(extract_cfg.get("max_dominant_ratio", 0.90))
        self.enable_color_analysis = bool(extract_cfg.get("enable_color_analysis", True))
        self.enable_brightness_analysis = bool(extract_cfg.get("enable_brightness_analysis", True))
        self.enable_hint_transfer = bool(extract_cfg.get("enable_hint_transfer", True))

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def extract(
        self,
        generation_result: Dict[str, Any],
        prompt_bundle: Dict[str, Any] = None,
        scene_packet: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        generation_result = self._safe_dict(generation_result)
        prompt_bundle = self._safe_dict(prompt_bundle)
        scene_packet = self._safe_dict(scene_packet)

        if not self.enabled:
            return self._missing_summary()

        metadata = self._safe_dict(generation_result.get("metadata", {}))
        paths = self._candidate_frame_paths(generation_result, metadata)

        chosen_path, img = self._load_best_image(paths)
        visual_report = self._analyze_image(img) if img is not None else self._empty_visual_report()

        prompt_contract = self._safe_dict(prompt_bundle.get("prompt_contract", {}))
        scene_packet = scene_packet or self._safe_dict(prompt_bundle.get("scene_packet", {}))

        summary = self._build_summary(
            visual_report=visual_report,
            prompt_contract=prompt_contract,
            scene_packet=scene_packet,
            metadata=metadata,
            has_real_image=img is not None,
        )

        semantic_evidence_status = self._semantic_evidence_status(summary, visual_report, has_real_image=(img is not None))
        has_semantic_evidence = self._has_semantic_evidence(summary)

        result = {
            "generated_summary": summary,
            "semantic_evidence_status": semantic_evidence_status,
            "has_semantic_evidence": has_semantic_evidence,
            "visual_report": visual_report,
            "selected_image_path": chosen_path,
        }
        return result

    def attach_to_generation_result(
        self,
        generation_result: Dict[str, Any],
        prompt_bundle: Dict[str, Any] = None,
        scene_packet: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        generation_result = self._safe_dict(generation_result)
        metadata = self._safe_dict(generation_result.get("metadata", {}))

        extracted = self.extract(
            generation_result=generation_result,
            prompt_bundle=prompt_bundle,
            scene_packet=scene_packet,
        )

        metadata["generated_summary"] = extracted["generated_summary"]
        metadata["semantic_evidence_status"] = extracted["semantic_evidence_status"]
        metadata["has_semantic_evidence"] = extracted["has_semantic_evidence"]
        metadata["visual_report"] = extracted["visual_report"]
        metadata["selected_analysis_image_path"] = extracted["selected_image_path"]

        generation_result["metadata"] = metadata
        return generation_result

    # ------------------------------------------------------------------
    # image selection
    # ------------------------------------------------------------------

    def _candidate_frame_paths(
        self,
        generation_result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[str]:
        candidates = [
            generation_result.get("best_keyframe_path", ""),
            generation_result.get("selected_keyframe_path", ""),
            generation_result.get("keyframe_path", ""),
            generation_result.get("middle_frame_path", ""),
            generation_result.get("first_frame_path", ""),
            generation_result.get("last_frame_path", ""),
            metadata.get("best_keyframe_path", ""),
            metadata.get("selected_keyframe_path", ""),
            metadata.get("keyframe_path", ""),
            metadata.get("middle_frame_path", ""),
            metadata.get("first_frame_path", ""),
            metadata.get("last_frame_path", ""),
        ]
        out: List[str] = []
        seen = set()
        for p in candidates:
            p = self._safe_text(p)
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            if os.path.isfile(p):
                out.append(p)
        return out

    def _load_best_image(self, paths: List[str]) -> Tuple[str, Any]:
        if Image is None:
            return "", None

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                return p, img
            except Exception:
                continue
        return "", None

    # ------------------------------------------------------------------
    # visual analysis
    # ------------------------------------------------------------------

    def _analyze_image(self, img: Any) -> Dict[str, Any]:
        if img is None or Image is None or np is None:
            return self._empty_visual_report()

        try:
            small = img.resize((self.sample_resize, self.sample_resize)).convert("RGB")
            arr = np.array(small)

            brightness = float(arr.mean())
            std_val = float(arr.std())

            pixels = arr.reshape(-1, 3)
            sample = pixels[::max(1, len(pixels) // 5000)]
            if len(sample) == 0:
                dominant_ratio = 1.0
                unique_colors = 1
            else:
                unique, counts = np.unique(sample, axis=0, return_counts=True)
                dominant_ratio = float(counts.max()) / float(counts.sum()) if len(counts) > 0 else 1.0
                unique_colors = int(len(unique))

            avg_rgb = [float(x) for x in arr.mean(axis=(0, 1))]
            color_tone = self._infer_color_tone(avg_rgb)
            lighting = self._infer_lighting(brightness)
            detail_level = self._infer_detail_level(std_val, dominant_ratio, unique_colors)

            return {
                "ok": True,
                "brightness": brightness,
                "std": std_val,
                "dominant_ratio": dominant_ratio,
                "unique_colors": unique_colors,
                "avg_rgb": avg_rgb,
                "color_tone": color_tone,
                "lighting": lighting,
                "detail_level": detail_level,
                "degenerate_like": bool(
                    std_val < self.min_std_threshold or dominant_ratio >= self.max_dominant_ratio
                ),
            }
        except Exception:
            return self._empty_visual_report()

    def _infer_color_tone(self, avg_rgb: List[float]) -> str:
        if len(avg_rgb) != 3:
            return ""

        r, g, b = avg_rgb
        if r > b + 20 and r > g + 10:
            return "warm"
        if b > r + 20 and b > g:
            return "cool"
        if g > r + 15 and g > b + 10:
            return "greenish"
        if max(avg_rgb) - min(avg_rgb) < 12:
            return "neutral"
        return "mixed"

    def _infer_lighting(self, brightness: float) -> str:
        if brightness < 60:
            return "dark"
        if brightness < 110:
            return "dim"
        if brightness < 180:
            return "balanced"
        return "bright"

    def _infer_detail_level(
        self,
        std_val: float,
        dominant_ratio: float,
        unique_colors: int,
    ) -> str:
        if std_val < self.min_std_threshold or dominant_ratio > self.max_dominant_ratio:
            return "low"
        if std_val < 20 or unique_colors < 100:
            return "medium"
        return "high"

    def _empty_visual_report(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "brightness": 0.0,
            "std": 0.0,
            "dominant_ratio": 1.0,
            "unique_colors": 0,
            "avg_rgb": [],
            "color_tone": "",
            "lighting": "",
            "detail_level": "low",
            "degenerate_like": True,
        }

    # ------------------------------------------------------------------
    # summary construction
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        visual_report: Dict[str, Any],
        prompt_contract: Dict[str, Any],
        scene_packet: Dict[str, Any],
        metadata: Dict[str, Any],
        has_real_image: bool,
    ) -> Dict[str, Any]:
        same_as_previous = self._safe_dict(prompt_contract.get("same_as_previous", {}))
        identity_lock = self._safe_dict(prompt_contract.get("identity_lock", {}))
        location_lock = self._safe_dict(prompt_contract.get("location_lock", {}))
        prop_lock = self._safe_dict(prompt_contract.get("prop_lock", {}))
        style_lock = self._safe_dict(prompt_contract.get("style_lock", {}))

        summary = self._missing_summary()

        # --------------------------------------------------------------
        # style: safest area for weak visual extraction
        # --------------------------------------------------------------
        style = {}
        if has_real_image and visual_report.get("ok", False):
            if self.enable_color_analysis:
                color_tone = self._safe_text(visual_report.get("color_tone", ""))
                if color_tone:
                    style["color_tone"] = color_tone

            if self.enable_brightness_analysis:
                lighting = self._safe_text(visual_report.get("lighting", ""))
                if lighting:
                    style["lighting"] = lighting

            detail_level = self._safe_text(visual_report.get("detail_level", ""))
            if detail_level:
                style["detail_level"] = detail_level

        # weak transfer from expected style only for fields that are hard to observe visually
        if self.enable_hint_transfer and has_real_image and visual_report.get("ok", False):
            for field in ["visual_style", "shot_type", "camera_angle", "camera_motion", "mood"]:
                value = self._safe_text(style_lock.get(field, ""))
                if value:
                    # mark as weak hint, not confirmed observation
                    style[f"{field}_hint"] = value

        if style:
            summary["style"] = style

        # --------------------------------------------------------------
        # location: only weak, conservative extraction
        # --------------------------------------------------------------
        location = {}
        if has_real_image and visual_report.get("ok", False):
            location["visual_lighting"] = self._safe_text(visual_report.get("lighting", ""))
            location["visual_color_tone"] = self._safe_text(visual_report.get("color_tone", ""))

        # if image is present and continuity expects same location, keep only hint fields
        if self.enable_hint_transfer and has_real_image:
            expected_location_name = self._safe_text(location_lock.get("name", ""))
            if expected_location_name:
                location["name_hint"] = expected_location_name

            expected_anchors = [
                self._safe_text(x) for x in self._safe_list(location_lock.get("anchors", []))
                if self._safe_text(x)
            ]
            if expected_anchors:
                location["anchors_hint"] = expected_anchors[:5]

        if location:
            summary["location"] = location

        # --------------------------------------------------------------
        # characters: never hallucinate confirmed identity from prompt
        # --------------------------------------------------------------
        characters = []
        if self.enable_hint_transfer and has_real_image:
            locked_names = [
                self._safe_text(x) for x in self._safe_list(identity_lock.get("locked_names", []))
                if self._safe_text(x)
            ]

            if locked_names and bool(same_as_previous.get("character_identity", False)):
                for name in locked_names[:3]:
                    characters.append({
                        "name_hint": name,
                        "evidence": "weak_context",
                    })

        if characters:
            summary["characters"] = characters

        # --------------------------------------------------------------
        # props: weak hints only
        # --------------------------------------------------------------
        props = []
        if self.enable_hint_transfer and has_real_image:
            expected_props = self._safe_list(prop_lock.get("props", []))
            for item in expected_props[:5]:
                item = self._safe_dict(item)
                name = self._safe_text(item.get("name", ""))
                if name:
                    props.append({
                        "name_hint": name,
                        "evidence": "weak_context",
                    })

        if props:
            summary["props"] = props

        # source tagging
        if has_real_image and visual_report.get("ok", False):
            if summary["style"] or summary["location"] or summary["characters"] or summary["props"]:
                if summary["characters"] or summary["props"]:
                    summary["source"] = "visual_heuristic+weak_context"
                else:
                    summary["source"] = "visual_heuristic"
            else:
                summary["source"] = "missing"
        else:
            summary["source"] = "missing"

        return summary

    # ------------------------------------------------------------------
    # evidence status
    # ------------------------------------------------------------------

    def _semantic_evidence_status(
        self,
        summary: Dict[str, Any],
        visual_report: Dict[str, Any],
        has_real_image: bool,
    ) -> str:
        if not has_real_image:
            return "missing"

        if not visual_report.get("ok", False):
            return "missing"

        if visual_report.get("degenerate_like", False):
            return "missing"

        style = self._safe_dict(summary.get("style", {}))
        location = self._safe_dict(summary.get("location", {}))
        characters = self._safe_list(summary.get("characters", []))
        props = self._safe_list(summary.get("props", []))

        strong_fields = 0
        weak_fields = 0

        if style:
            strong_fields += 1
        if location:
            strong_fields += 1

        if characters:
            weak_fields += 1
        if props:
            weak_fields += 1

        if strong_fields >= 2:
            return "partial"
        if strong_fields >= 1 and weak_fields >= 1:
            return "partial"
        if strong_fields >= 1:
            return "partial"
        return "missing"

    def _has_semantic_evidence(self, summary: Dict[str, Any]) -> bool:
        summary = self._safe_dict(summary)
        if self._safe_dict(summary.get("style", {})):
            return True
        if self._safe_dict(summary.get("location", {})):
            return True
        if self._safe_list(summary.get("characters", [])):
            return True
        if self._safe_list(summary.get("props", [])):
            return True
        return False

    # ------------------------------------------------------------------
    # defaults / utils
    # ------------------------------------------------------------------

    def _missing_summary(self) -> Dict[str, Any]:
        return {
            "characters": [],
            "location": {},
            "props": [],
            "style": {},
            "source": "missing",
        }

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


def attach_generated_summary(
    generation_result: Dict[str, Any],
    prompt_bundle: Dict[str, Any] = None,
    scene_packet: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Small helper function for easy integration from backends / scene generator / manager.
    """
    extractor = GeneratedStateExtractor(config=config)
    return extractor.attach_to_generation_result(
        generation_result=generation_result,
        prompt_bundle=prompt_bundle,
        scene_packet=scene_packet,
    )