# src/generation/backend/cogvideox_backend.py
from __future__ import annotations

import gc
import os
import random
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

from src.continuity.keyframe_selector import select_role_keyframes

from src.generation.backend.base import BaseGenerationBackend
from src.generation.backend.common import (
    ensure_dir,
    build_clean_scene_prompt,
    build_negative_prompt,
    choose_reference_plan,
    strengthen_params_for_retry,
    extract_generation_params,
    normalize_control_weights,
    build_generated_metadata_from_constraints,
    is_first_scene,
    is_dependent_scene,
)


class CogVideoXBackend(BaseGenerationBackend):
    """
    CogVideoX backend using a safer, working-style text-to-video runtime path.

    Main rules:
    - scene 1 must remain true text-to-video
    - no fake anchor conditioning for scene 1
    - if generation fails, fail honestly
    - fallback output is debug-only, never treated as success
    """

    backend_kind = "cogvideox"

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})

        self.output_dir = self.config.get("output_dir", "outputs/generated/cogvideox")
        self.device = self.config.get("device", "cuda")
        self.model_id = self.config.get("model_id", "THUDM/CogVideoX-2b")

        # safer defaults based on working old backend
        self.default_fps = int(self.config.get("fps", 8))
        self.default_num_frames = int(self.config.get("num_frames", 8))
        self.default_width = int(self.config.get("width", 512))
        self.default_height = int(self.config.get("height", 320))
        self.seed = self.config.get("seed", None)

        self.guidance_scale = float(self.config.get("guidance_scale", 4.5))
        self.num_inference_steps = int(self.config.get("num_inference_steps", 12))
        self.max_sequence_length = int(self.config.get("max_sequence_length", 224))

        # old working style toggles
        self.enable_model_cpu_offload = bool(self.config.get("enable_model_cpu_offload", True))
        self.enable_sequential_cpu_offload = bool(self.config.get("enable_sequential_cpu_offload", False))
        self.enable_vae_slicing = bool(self.config.get("enable_vae_slicing", True))
        self.enable_vae_tiling = bool(self.config.get("enable_vae_tiling", True))
        self.use_dynamic_cfg = bool(self.config.get("use_dynamic_cfg", False))

        self.pipeline = None
        self._loaded = False

        ensure_dir(self.output_dir)

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def generate(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_core(prompt_bundle=prompt_bundle, repair_mode=False)

    def generate_repair(self, prompt_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_core(prompt_bundle=prompt_bundle, repair_mode=True)

    def unload(self) -> None:
        try:
            if getattr(self, "pipeline", None) is not None:
                try:
                    del self.pipeline
                except Exception:
                    pass
                self.pipeline = None
                self._loaded = False
        finally:
            gc.collect()
            self._safe_cuda_cleanup()

    # ------------------------------------------------------------------
    # main generation
    # ------------------------------------------------------------------

    def _generate_core(self, prompt_bundle: Dict[str, Any], repair_mode: bool) -> Dict[str, Any]:
        prompt_bundle = dict(prompt_bundle or {})
        scene_packet = dict(prompt_bundle.get("scene_packet", {}) or {})
        scene_id = self._scene_id(prompt_bundle)
        scene_dir = ensure_dir(os.path.join(self.output_dir, scene_id))

        prompt_text = build_clean_scene_prompt(
            prompt_bundle=prompt_bundle,
            backend_kind=self.backend_kind,
            prefer_repair=repair_mode,
        )
        negative_prompt = build_negative_prompt(
            prompt_bundle=prompt_bundle,
            backend_kind=self.backend_kind,
            repair_mode=repair_mode,
        )

        reference_plan = choose_reference_plan(
            prompt_bundle=prompt_bundle,
            backend_kind=self.backend_kind,
        )

        generation_params = extract_generation_params(
            prompt_bundle=prompt_bundle,
            backend_defaults={
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "num_frames": self.default_num_frames,
                "fps": self.default_fps,
                "width": self.default_width,
                "height": self.default_height,
                "reference_strength": 0.65,
            },
        )

        control_weights = normalize_control_weights(
            prompt_bundle.get("control_weights", {})
        )

        retry_index = int(
            ((prompt_bundle.get("prompt_metadata", {}) or {}).get("retry_index", 0)) or 0
        )
        if retry_index > 0:
            generation_params, control_weights = strengthen_params_for_retry(
                generation_params=generation_params,
                control_weights=control_weights,
                prompt_bundle=prompt_bundle,
                retry_index=retry_index,
            )

        policy = self._build_scene_policy(
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
            reference_plan=reference_plan,
            control_weights=control_weights,
        )

        paths = self._scene_output_paths(scene_id)

        result = self._call_real_pipeline(
            prompt_text=prompt_text,
            negative_prompt=negative_prompt,
            output_video_path=paths["video_path"],
            generation_params=generation_params,
            control_weights=control_weights,
            policy=policy,
            prompt_bundle=prompt_bundle,
            paths=paths,
        )

        if not result.get("ok", False):
            # debug-only fallback artifact; never treated as success
            result = self._fallback_generate_video(
                prompt_text=prompt_text,
                output_video_path=paths["video_path"],
                width=int(generation_params.get("width", self.default_width)),
                height=int(generation_params.get("height", self.default_height)),
                num_frames=int(generation_params.get("num_frames", self.default_num_frames)),
                fps=int(generation_params.get("fps", self.default_fps)),
                reason=result.get("error", "unknown_error"),
            )

        keyframe_paths = self._extract_and_store_keyframes(
            result=result,
            scene_dir=scene_dir,
            scene_id=scene_id,
            existing_paths=paths,
            prompt_bundle=prompt_bundle,
        )

        metadata = build_generated_metadata_from_constraints(
            prompt_bundle=prompt_bundle,
            reference_plan=reference_plan,
            generation_params=generation_params,
            backend_kind=self.backend_kind,
            video_path=result.get("video_path", paths["video_path"]),
            keyframe_paths=keyframe_paths,
        )
        metadata["repair_mode"] = bool(repair_mode)
        metadata["scene_policy"] = policy
        metadata["control_weights"] = control_weights
        metadata["negative_prompt"] = negative_prompt
        metadata["prompt_text"] = prompt_text
        metadata["quality_score"] = result.get("quality_score", 0.0)
        metadata["attempt_used"] = result.get("attempt_used", 0)
        metadata["used_num_frames"] = result.get("used_num_frames", 0)
        metadata["used_width"] = result.get("used_width", 0)
        metadata["used_height"] = result.get("used_height", 0)
        metadata["is_fallback_output"] = bool(result.get("is_fallback_output", False))
        metadata["fallback_reason"] = result.get("fallback_reason", "")
        metadata["generation_failed"] = bool(result.get("generation_failed", False))
        metadata["placeholder_conditioning_only"] = bool(result.get("placeholder_conditioning_only", False))
        metadata["has_real_init_image"] = bool(result.get("has_real_init_image", False))
        metadata["has_real_reference_assets"] = bool(result.get("has_real_reference_assets", False))
        metadata["reference_source_type"] = result.get("reference_source_type", "missing")
        metadata["semantic_evidence_status"] = result.get(
            "semantic_evidence_status",
            metadata.get("semantic_evidence_status", "missing"),
        )

        if result.get("metadata"):
            for k, v in dict(result["metadata"]).items():
                metadata[k] = v

        out = {
            "ok": bool(result.get("ok", False)),
            "scene_id": scene_id,
            "video_path": result.get("video_path", paths["video_path"]),
            "output_video_path": result.get("video_path", paths["video_path"]),
            "keyframe_path": keyframe_paths.get("best_keyframe_path", ""),
            "selected_keyframe_path": keyframe_paths.get("best_keyframe_path", ""),
            "first_frame_path": keyframe_paths.get("first_frame_path", ""),
            "middle_frame_path": keyframe_paths.get("middle_frame_path", ""),
            "last_frame_path": keyframe_paths.get("last_frame_path", ""),
            "best_keyframe_path": keyframe_paths.get("best_keyframe_path", ""),
            "metadata": metadata,
        }

        if result.get("frames"):
            out["frames"] = result["frames"]
        if result.get("error"):
            out["error"] = result["error"]

        return out

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------

    def _build_scene_policy(
        self,
        prompt_bundle: Dict[str, Any],
        repair_mode: bool,
        reference_plan: Dict[str, Any],
        control_weights: Dict[str, Any],
    ) -> Dict[str, Any]:
        first_scene = is_first_scene(prompt_bundle)
        dependent_scene = is_dependent_scene(prompt_bundle)

        policy = {
            "first_scene": first_scene,
            "dependent_scene": dependent_scene,
            "repair_mode": bool(repair_mode),
            "pure_text_to_video": bool(first_scene),
            "allow_reference_conditioning": False,   # important
            "allow_motion_freedom": True,
            "tight_prompt_mode": False,
        }

        # current CogVideoX path is kept pure T2V even for later scenes
        # continuity is carried through prompt/control, not image injection
        if dependent_scene:
            policy["allow_motion_freedom"] = control_weights.get("motion_strength", 0.5) >= 0.50

        if repair_mode:
            policy["tight_prompt_mode"] = True
            policy["allow_motion_freedom"] = False

        return policy

    # ------------------------------------------------------------------
    # lazy pipeline load
    # ------------------------------------------------------------------

    def _safe_cuda_cleanup(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_pipeline(self):
        if self._loaded and self.pipeline is not None:
            return

        import torch
        from diffusers import CogVideoXPipeline

        self._safe_cuda_cleanup()

        dtype_name = str(self.config.get("torch_dtype", "float16")).lower()
        if dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        self.pipeline = CogVideoXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        if self.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()

        if self.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()

        try:
            if self.enable_vae_slicing and hasattr(self.pipeline.vae, "enable_slicing"):
                self.pipeline.vae.enable_slicing()
        except Exception:
            pass

        try:
            if self.enable_vae_tiling and hasattr(self.pipeline.vae, "enable_tiling"):
                self.pipeline.vae.enable_tiling()
        except Exception:
            pass

        self._loaded = True

    # ------------------------------------------------------------------
    # real pipeline call
    # ------------------------------------------------------------------

    def _call_real_pipeline(
        self,
        prompt_text: str,
        negative_prompt: str,
        output_video_path: str,
        generation_params: Dict[str, Any],
        control_weights: Dict[str, Any],
        policy: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        paths: Dict[str, str],
    ) -> Dict[str, Any]:
        try:
            self._load_pipeline()

            import torch

            num_inference_steps = int(generation_params.get("num_inference_steps", self.num_inference_steps))
            guidance_scale = float(generation_params.get("guidance_scale", self.guidance_scale))
            num_frames = int(generation_params.get("num_frames", self.default_num_frames))
            fps = int(generation_params.get("fps", self.default_fps))
            height = int(generation_params.get("height", self.default_height))
            width = int(generation_params.get("width", self.default_width))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            seed = self.seed
            if seed is None:
                seed = random.randint(0, 10_000_000)

            # old working approach
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

            # for first scene this remains pure T2V
            kwargs = {
                "prompt": prompt_text,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "generator": generator,
                "num_videos_per_prompt": 1,
            }

            if self.use_dynamic_cfg:
                kwargs["use_dynamic_cfg"] = True

            try:
                result_obj = self.pipeline(**kwargs)
            except TypeError:
                # fallback for version compatibility
                kwargs.pop("use_dynamic_cfg", None)
                kwargs.pop("max_sequence_length", None)
                result_obj = self.pipeline(**kwargs)

            frames = self._extract_frames_from_output(result_obj)

            saved = self._save_frames_and_video(
                frames=frames,
                paths=paths,
                fps=fps,
            )
            if not saved.get("ok", False):
                return self._attach_debug_error(
                    {
                        **saved,
                        "quality_score": 0.0,
                        "generation_failed": True,
                        "is_fallback_output": False,
                    },
                    stage="save_frames_and_video",
                    err=saved.get("error", "save_failed"),
                    extra={
                        "model_id": self.model_id,
                        "device": self.device,
                    },
                )

            return {
                **saved,
                "quality_score": 0.82,
                "attempt_used": 1,
                "used_num_frames": num_frames,
                "used_width": width,
                "used_height": height,
                "generation_failed": False,
                "is_fallback_output": False,
                "placeholder_conditioning_only": False,
                "has_real_init_image": False,
                "has_real_reference_assets": False,
                "reference_source_type": "missing",
                "semantic_evidence_status": "missing",
                "backend_seed": int(seed),
            }

        except Exception as e:
            return self._attach_debug_error(
                {
                    "ok": False,
                    "error": f"CogVideoX generation failed: {e}",
                    "video_path": "",
                    "selected_keyframe": "",
                    "first_frame": "",
                    "last_frame": "",
                    "quality_score": 0.0,
                    "generation_failed": True,
                    "is_fallback_output": False,
                    "placeholder_conditioning_only": False,
                    "has_real_init_image": False,
                    "has_real_reference_assets": False,
                    "reference_source_type": "missing",
                    "semantic_evidence_status": "missing",
                },
                stage="real_pipeline",
                err=e,
                extra={
                    "model_id": self.model_id,
                    "device": self.device,
                    "scene1_pure_t2v": bool(policy.get("pure_text_to_video", False)),
                },
            )

    def _extract_frames_from_output(self, output):
        if output is None:
            raise RuntimeError("CogVideoX returned no output")

        frames = getattr(output, "frames", None)
        if frames is None and isinstance(output, dict):
            frames = output.get("frames", None)

        if frames is None:
            raise RuntimeError("CogVideoX output did not contain frames")

        if isinstance(frames, list):
            if len(frames) == 0:
                raise RuntimeError("CogVideoX returned empty frames")
            if isinstance(frames[0], list):
                return frames[0]
            return frames

        raise RuntimeError(f"Unsupported CogVideoX frames format: {type(frames).__name__}")

    def _save_frames_and_video(self, frames, paths: Dict[str, str], fps: int) -> Dict[str, Any]:
        from diffusers.utils import export_to_video

        if not frames:
            return {
                "ok": False,
                "error": "No frames returned by CogVideoX pipeline."
            }

        first_frame = self._to_pil(frames[0])
        mid_frame = self._to_pil(frames[len(frames) // 2])
        last_frame = self._to_pil(frames[-1])

        if first_frame is None or mid_frame is None or last_frame is None:
            return {
                "ok": False,
                "error": "Could not convert generated frames to PIL."
            }

        first_frame.save(paths["first_frame"])
        mid_frame.save(paths["selected_keyframe"])
        last_frame.save(paths["last_frame"])

        export_to_video(frames, paths["video_path"], fps=fps)

        return {
            "ok": True,
            "video_path": paths["video_path"],
            "selected_keyframe": paths["selected_keyframe"],
            "first_frame": paths["first_frame"],
            "last_frame": paths["last_frame"],
            "frames": frames,
            "is_fallback_output": False,
            "generation_failed": False,
        }

    # ------------------------------------------------------------------
    # fallback debug artifact
    # ------------------------------------------------------------------

    def _fallback_generate_video(
        self,
        prompt_text: str,
        output_video_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Debug-only placeholder artifact.
        This should never be treated as valid generation.
        """
        try:
            frames: List[Image.Image] = []

            bg = np.zeros((height, width, 3), dtype=np.uint8)
            bg[:, :] = np.array(self._background_color_from_text(prompt_text), dtype=np.uint8)

            for i in range(max(1, num_frames)):
                frame = bg.copy()
                band_y = (i * max(1, height // max(1, num_frames))) % max(1, height)
                y2 = min(height, band_y + max(6, height // 20))
                frame[band_y:y2, :, :] = np.clip(frame[band_y:y2, :, :] + 12, 0, 255)
                frames.append(Image.fromarray(frame))

            self._save_frames_to_video(
                frames=frames,
                output_video_path=output_video_path,
                fps=fps,
            )

            return self._attach_debug_error(
                {
                    "ok": False,
                    "error": f"Fallback output used because real CogVideoX generation failed: {reason}",
                    "video_path": output_video_path,
                    "frames": frames,
                    "fallback_reason": reason,
                    "quality_score": 0.05,
                    "is_fallback_output": True,
                    "generation_failed": True,
                    "placeholder_conditioning_only": False,
                    "has_real_init_image": False,
                    "has_real_reference_assets": False,
                    "reference_source_type": "missing",
                    "semantic_evidence_status": "missing",
                },
                stage="fallback_generate_video",
                err=reason,
                extra={
                    "model_id": self.model_id,
                    "device": self.device,
                },
            )
        except Exception as e:
            return self._attach_debug_error(
                {
                    "ok": False,
                    "error": f"fallback video generation failed: {e}",
                    "video_path": "",
                    "quality_score": 0.0,
                    "is_fallback_output": True,
                    "generation_failed": True,
                    "placeholder_conditioning_only": False,
                    "has_real_init_image": False,
                    "has_real_reference_assets": False,
                    "reference_source_type": "missing",
                    "semantic_evidence_status": "missing",
                },
                stage="fallback_generate_video",
                err=e,
                extra={
                    "model_id": self.model_id,
                    "device": self.device,
                },
            )

    def _save_frames_to_video(
        self,
        frames: List[Image.Image],
        output_video_path: str,
        fps: int,
    ) -> None:
        try:
            import imageio.v2 as imageio
            arrs = [np.array(f.convert("RGB")) for f in frames]
            imageio.mimsave(output_video_path, arrs, fps=fps)
            return
        except Exception:
            pass

        if frames:
            frames[0].save(
                output_video_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(1, int(1000 / max(1, fps))),
                loop=0,
            )

    def _extract_and_store_keyframes(
        self,
        result: Dict[str, Any],
        scene_dir: str,
        scene_id: str,
        existing_paths: Dict[str, str],
        prompt_bundle: Dict[str, Any] = None,
    ) -> Dict[str, str]:

        frames = result.get("frames", None)

        # fallback (no frames)
        if not isinstance(frames, list) or len(frames) == 0:
            return {
                "identity_frame_path": "",
                "location_frame_path": "",
                "prop_frame_path": "",
                "transition_frame_path": "",
                "style_frame_path": "",
                "first_frame_path": "",
                "middle_frame_path": "",
                "last_frame_path": "",
                "best_keyframe_path": "",
            }

        # MAIN LOGIC (NEW)
        selector_result = select_role_keyframes(
            frames=frames,
            scene_id=scene_id,
            output_dir=scene_dir,
            scene_packet=(prompt_bundle or {}).get("scene_packet", {}),
            prompt_bundle=prompt_bundle,
            config=self.config,
        )

        if not selector_result.get("ok", False):
            return {
                "identity_frame_path": "",
                "location_frame_path": "",
                "prop_frame_path": "",
                "transition_frame_path": "",
                "style_frame_path": "",
                "first_frame_path": "",
                "middle_frame_path": "",
                "last_frame_path": "",
                "best_keyframe_path": "",
            }

        paths = selector_result.get("paths", {})

        return {
            "identity_frame_path": paths.get("identity_frame_path", ""),
            "location_frame_path": paths.get("location_frame_path", ""),
            "prop_frame_path": paths.get("prop_frame_path", ""),
            "transition_frame_path": paths.get("transition_frame_path", ""),
            "style_frame_path": paths.get("style_frame_path", ""),
            "first_frame_path": paths.get("first_frame_path", ""),
            "middle_frame_path": paths.get("middle_frame_path", ""),
            "last_frame_path": paths.get("last_frame_path", ""),
            "best_keyframe_path": paths.get("best_keyframe_path", ""),
        }

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------

    def _scene_output_paths(self, scene_id: str) -> Dict[str, str]:
        scene_dir = os.path.join(self.output_dir, scene_id)
        ensure_dir(scene_dir)
        return {
            "scene_dir": scene_dir,
            "video_path": os.path.join(scene_dir, f"{scene_id}.mp4"),
            "selected_keyframe": os.path.join(scene_dir, f"{scene_id}_keyframe.png"),
            "first_frame": os.path.join(scene_dir, f"{scene_id}_first.png"),
            "last_frame": os.path.join(scene_dir, f"{scene_id}_last.png"),
        }

    def _scene_id(self, prompt_bundle: Dict[str, Any]) -> str:
        scene_id = str(prompt_bundle.get("scene_id", "") or "").strip()
        if scene_id:
            return scene_id
        scene_packet = prompt_bundle.get("scene_packet", {}) or {}
        scene_id = str(scene_packet.get("scene_id", "") or "").strip()
        if scene_id:
            return scene_id
        return "scene_unknown"

    def _to_pil(self, frame: Any) -> Optional[Image.Image]:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        return None

    def _background_color_from_text(self, text: str) -> List[int]:
        text = (text or "").lower()
        if "night" in text:
            return [18, 22, 60]
        if "sunset" in text or "warm" in text:
            return [200, 130, 70]
        if "rain" in text or "cloud" in text:
            return [95, 105, 125]
        if "forest" in text or "green" in text:
            return [75, 125, 80]
        if "cold" in text or "blue" in text:
            return [80, 110, 165]
        return [125, 135, 150]

    def _final_error_text(self, err: Any) -> str:
        text = str(err or "").strip()
        if not text:
            return "unknown_generation_error"
        return text[:1200]

    def _attach_debug_error(
        self,
        result: Dict[str, Any],
        stage: str,
        err: Any,
        extra: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        result = dict(result or {})
        metadata = dict(result.get("metadata", {}) or {})
        metadata["debug_error_stage"] = stage
        metadata["debug_error_text"] = self._final_error_text(err)
        if extra:
            metadata["debug_error_extra"] = extra
        result["metadata"] = metadata
        return result