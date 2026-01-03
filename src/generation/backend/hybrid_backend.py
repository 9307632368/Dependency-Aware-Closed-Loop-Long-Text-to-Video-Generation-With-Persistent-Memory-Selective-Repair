# src/generation/backend/hybrid_backend.py
from __future__ import annotations

from typing import Dict, Any, Optional

from src.generation.backend.base import BaseGenerationBackend
from src.generation.backend.router import choose_backend_route
from src.generation.backend.svd_backend import SVDBackend
from src.generation.backend.cogvideox_backend import CogVideoXBackend


class HybridBackend(BaseGenerationBackend):
    """
    Final low-VRAM hybrid backend.

    Rules:
    - scene 1 must stay text-first cogvideox
    - no fallback from scene 1 to svd
    - dependent later scenes prefer svd
    - repair prefers svd
    - lazy-load one backend at a time
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.backend_kind = "hybrid"

        hybrid_cfg = self._safe_dict(self.config.get("hybrid", {}))
        self.enable_fallback = bool(hybrid_cfg.get("enable_fallback", True))
        self.disable_fallback_on_oom = bool(hybrid_cfg.get("disable_fallback_on_oom", True))
        self.unload_after_each_scene = bool(hybrid_cfg.get("unload_after_each_scene", True))
        self.disable_scene1_fallback = bool(hybrid_cfg.get("disable_scene1_fallback", True))

    def generate(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_core(prompt_bundle=prompt_bundle, repair_mode=False)

    def generate_repair(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_core(prompt_bundle=prompt_bundle, repair_mode=True)

    def unload(self) -> None:
        pass

    def _generate_core(self, prompt_bundle: Dict[str, Any], repair_mode: bool) -> Dict[str, Any]:
        prompt_bundle = dict(prompt_bundle or {})

        route = self._choose_route(
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
        )

        primary_name = route.get("primary_backend", "")
        fallback_name = route.get("fallback_backend", "")

        scene_packet = self._safe_dict(prompt_bundle.get("scene_packet", {}))
        scene_id = str(prompt_bundle.get("scene_id", "") or scene_packet.get("scene_id", "")).lower()
        first_scene = scene_id.endswith("001") or scene_id in {"scene1", "scene_1", "1"}

        primary_result = self._run_backend_once(
            backend_name=primary_name,
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
            route=route,
            attempt_label="primary",
            primary_error="",
        )

        if self._is_usable_result(primary_result):
            return primary_result

        # scene 1 must not fallback to svd
        if first_scene and self.disable_scene1_fallback:
            return primary_result

        # do not same-process fallback when OOM happens
        if self.disable_fallback_on_oom and self._is_oom_result(primary_result):
            return primary_result

        if (not self.enable_fallback) or (not fallback_name) or (fallback_name == primary_name):
            return primary_result

        fallback_result = self._run_backend_once(
            backend_name=fallback_name,
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
            route=route,
            attempt_label="fallback",
            primary_error=primary_result.get("error", ""),
        )

        if self._is_usable_result(fallback_result):
            return fallback_result

        return self._merge_failed_results(
            primary_result=primary_result,
            fallback_result=fallback_result,
            route=route,
            repair_mode=repair_mode,
        )

    def _choose_route(self, prompt_bundle: Dict[str, Any], repair_mode: bool) -> Dict[str, Any]:
        try:
            route = choose_backend_route(
                prompt_bundle=prompt_bundle,
                available_backends=["svd", "cogvideox"],
                repair_mode=repair_mode,
                config=self.config,
            )
            route = dict(route or {})
        except Exception:
            route = {}

        if route.get("primary_backend"):
            route.setdefault("fallback_backend", self._default_fallback(route["primary_backend"]))
            route.setdefault("route_reason", "router_decision")
            route.setdefault("route_scores", {})
            return route

        scene_packet = self._safe_dict(prompt_bundle.get("scene_packet", {}))
        scene_id = str(prompt_bundle.get("scene_id", "") or scene_packet.get("scene_id", "")).lower()

        if repair_mode:
            primary = "svd"
            reason = "repair_prefers_svd"
        elif scene_id.endswith("001") or scene_id in {"scene1", "scene_1", "1"}:
            primary = "cogvideox"
            reason = "first_scene_text_only"
        elif bool(scene_packet.get("dependent_on_previous", False)):
            primary = "svd"
            reason = "dependent_scene_prefers_svd"
        else:
            primary = "cogvideox"
            reason = "independent_scene_prefers_cogvideox"

        return {
            "primary_backend": primary,
            "fallback_backend": self._default_fallback(primary),
            "route_reason": reason,
            "route_scores": {},
        }

    def _default_fallback(self, primary_name: str) -> str:
        if primary_name == "svd":
            return "cogvideox"
        if primary_name == "cogvideox":
            return "svd"
        return ""

    def _run_backend_once(
        self,
        backend_name: str,
        prompt_bundle: Dict[str, Any],
        repair_mode: bool,
        route: Dict[str, Any],
        attempt_label: str,
        primary_error: str,
    ) -> Dict[str, Any]:
        backend = None
        try:
            backend = self._build_backend_instance(backend_name)

            if backend is None:
                out = {
                    "ok": False,
                    "error": f"Backend '{backend_name}' is not available.",
                    "metadata": {},
                }
            else:
                if repair_mode and hasattr(backend, "generate_repair"):
                    out = backend.generate_repair(prompt_bundle)
                else:
                    out = backend.generate(prompt_bundle)

            out = dict(out or {})
            out = self._attach_route_metadata(
                result=out,
                chosen_backend=backend_name,
                fallback_backend=route.get("fallback_backend", ""),
                route=route,
                repair_mode=repair_mode,
                attempt_label=attempt_label,
                primary_error=primary_error,
            )
            return out

        except Exception as e:
            out = {
                "ok": False,
                "error": f"{backend_name} execution failed: {e}",
                "metadata": {
                    "backend_used": backend_name,
                    "debug_error_stage": "hybrid_run_backend_once",
                    "debug_error_text": str(e)[:1200],
                },
            }
            return self._attach_route_metadata(
                result=out,
                chosen_backend=backend_name,
                fallback_backend=route.get("fallback_backend", ""),
                route=route,
                repair_mode=repair_mode,
                attempt_label=attempt_label,
                primary_error=primary_error,
            )
        finally:
            if backend is not None and self.unload_after_each_scene and hasattr(backend, "unload"):
                try:
                    backend.unload()
                except Exception:
                    pass

    def _build_backend_instance(self, backend_name: str) -> Optional[Any]:
        backend_cfg = self._safe_dict(self.config.get("backends", {}))

        if backend_name == "svd":
            cfg = dict(self.config)
            cfg.update(self._safe_dict(backend_cfg.get("svd", {})))
            cfg["backend"] = "svd"
            return SVDBackend(cfg)

        if backend_name == "cogvideox":
            cfg = dict(self.config)
            cfg.update(self._safe_dict(backend_cfg.get("cogvideox", {})))
            cfg["backend"] = "cogvideox"
            return CogVideoXBackend(cfg)

        return None

    def _is_usable_result(self, result: Dict[str, Any]) -> bool:
        result = dict(result or {})
        if not bool(result.get("ok", False)):
            return False

        video_path = str(result.get("video_path", "") or result.get("output_video_path", "")).strip()
        if video_path:
            return True

        frames = result.get("frames", None)
        if isinstance(frames, list) and len(frames) > 0:
            return True

        return False

    def _is_oom_result(self, result: Dict[str, Any]) -> bool:
        result = dict(result or {})
        error_text = str(result.get("error", "") or "").lower()
        metadata = self._safe_dict(result.get("metadata", {}))
        debug_text = str(metadata.get("debug_error_text", "") or "").lower()

        combined = error_text + " " + debug_text
        return ("cuda out of memory" in combined) or ("out of memory" in combined)

    def _attach_route_metadata(
        self,
        result: Dict[str, Any],
        chosen_backend: str,
        fallback_backend: str,
        route: Dict[str, Any],
        repair_mode: bool,
        attempt_label: str,
        primary_error: str = "",
    ) -> Dict[str, Any]:
        result = dict(result or {})
        metadata = dict(result.get("metadata", {}) or {})

        metadata["backend_used"] = chosen_backend
        metadata["route_backend"] = chosen_backend
        metadata["route_fallback_backend"] = fallback_backend
        metadata["route_reason"] = route.get("route_reason", "")
        metadata["route_scores"] = route.get("route_scores", {}) or {}
        metadata["route_attempt"] = attempt_label
        metadata["repair_mode"] = bool(repair_mode)

        if primary_error:
            metadata["primary_backend_error"] = primary_error

        if route.get("primary_backend"):
            metadata["route_primary_backend"] = route["primary_backend"]
        if route.get("fallback_backend"):
            metadata["route_declared_fallback_backend"] = route["fallback_backend"]

        result["metadata"] = metadata
        result.setdefault("backend_used", chosen_backend)
        return result

    def _merge_failed_results(
        self,
        primary_result: Dict[str, Any],
        fallback_result: Dict[str, Any],
        route: Dict[str, Any],
        repair_mode: bool,
    ) -> Dict[str, Any]:
        primary_result = dict(primary_result or {})
        fallback_result = dict(fallback_result or {})

        scene_id = (
            primary_result.get("scene_id")
            or fallback_result.get("scene_id")
            or ""
        )

        return {
            "ok": False,
            "scene_id": scene_id,
            "error": "Both primary and fallback backends failed.",
            "primary_error": primary_result.get("error", ""),
            "fallback_error": fallback_result.get("error", ""),
            "metadata": {
                "backend_used": fallback_result.get("backend_used", "") or primary_result.get("backend_used", ""),
                "route_primary_backend": route.get("primary_backend", ""),
                "route_fallback_backend": route.get("fallback_backend", ""),
                "route_reason": route.get("route_reason", ""),
                "route_scores": route.get("route_scores", {}) or {},
                "repair_mode": bool(repair_mode),
            },
        }

    def _safe_dict(self, x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {}