# src/generation/backend/svd_backend.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from diffusers import StableVideoDiffusionPipeline
except Exception:  # pragma: no cover
    StableVideoDiffusionPipeline = None

from src.generation.backend.base import BaseGenerationBackend
from src.generation.backend.common import (
    ensure_dir,
    build_clean_scene_prompt,
    build_negative_prompt,
    normalize_control_weights,
    strengthen_controls_for_retry,
    strengthen_params_for_retry,
    choose_reference_plan,
    rank_reference_paths,
    extract_generation_params,
    safe_int,
    safe_float,
    build_generated_metadata_from_constraints,
)


class SVDBackend(BaseGenerationBackend):
    """
    Stable Video Diffusion backend.

    Important rules in this version:
    - placeholder-anchor-only generation must not be treated as real success
    - SVD pipeline call must stay compatible with installed diffusers version
    - negative prompt is tracked in metadata but NOT passed to SVD __call__
    """

    backend_kind = "svd"

    def __init__(self, cfg: Dict[str, Any]):
        cfg = cfg or {}

        self.model_id = cfg.get("model_id", "stabilityai/stable-video-diffusion-img2vid-xt")
        self.device = cfg.get("device", "cuda")
        self.dtype = cfg.get("dtype", "float16")

        self.default_width = int(cfg.get("width", 512))
        self.default_height = int(cfg.get("height", 288))
        self.default_num_frames = int(cfg.get("num_frames", 14))
        self.default_fps = int(cfg.get("fps", 8))
        self.default_motion_bucket_id = int(cfg.get("motion_bucket_id", 127))
        self.default_noise_aug_strength = float(cfg.get("noise_aug_strength", 0.02))
        self.default_num_inference_steps = int(cfg.get("num_inference_steps", 25))
        self.decode_chunk_size = int(cfg.get("decode_chunk_size", 8))
        self.seed = cfg.get("seed", None)

        self.enable_cpu_offload = bool(cfg.get("enable_cpu_offload", False))
        self.enable_model_cpu_offload = bool(cfg.get("enable_model_cpu_offload", False))
        self.enable_attention_slicing = bool(cfg.get("enable_attention_slicing", True))
        self.enable_vae_slicing = bool(cfg.get("enable_vae_slicing", True))
        self.enable_xformers = bool(cfg.get("enable_xformers", False))
        self.output_dir = cfg.get("output_dir", "outputs/generated/svd")

        self.pipeline = None

    # ------------------------------------------------------------------
    # public entry
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
        finally:
            self._safe_cuda_cleanup()

    def _generate_core(self, prompt_bundle: Dict[str, Any], repair_mode: bool) -> Dict[str, Any]:
        prompt_bundle = dict(prompt_bundle or {})
        scene_packet = dict(prompt_bundle.get("scene_packet", {}) or {})
        scene_id = str(prompt_bundle.get("scene_id", "") or scene_packet.get("scene_id", "") or "scene_unknown").strip()
        if not scene_id:
            scene_id = "scene_unknown"

        scene_dir = ensure_dir(os.path.join(self.config_output_dir(), scene_id))
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
                "num_frames": self.default_num_frames,
                "fps": self.default_fps,
                "width": self.default_width,
                "height": self.default_height,
                "num_inference_steps": self.default_num_inference_steps,
                "motion_bucket_id": self.default_motion_bucket_id,
                "noise_aug_strength": self.default_noise_aug_strength,
                "strength": self.default_noise_aug_strength,
                "reference_strength": 0.70,
            },
        )

        control_weights = normalize_control_weights(
            prompt_bundle.get("control_weights", {}) or reference_plan.get("control_weights", {}) or {}
        )

        retry_index = int(((prompt_bundle.get("prompt_metadata", {}) or {}).get("retry_index", 0)) or 0)
        if retry_index > 0:
            generation_params, control_weights = strengthen_params_for_retry(
                generation_params=generation_params,
                control_weights=control_weights,
                prompt_bundle=prompt_bundle,
                retry_index=retry_index,
            )
        else:
            control_weights = strengthen_controls_for_retry(
                control_weights=control_weights,
                retry_index=retry_index,
                repair_mode=repair_mode,
            )

        policy = dict(reference_plan.get("scene_policy", {}) or {})
        init_image_path = self._choose_init_image_path(reference_plan=reference_plan, policy=policy)

        paths = self._scene_output_paths(scene_dir=scene_dir, scene_id=scene_id)
        anchor_image_path = self._ensure_anchor_image(
            prompt_text=prompt_text,
            scene_dir=scene_dir,
            init_image_path=init_image_path,
            scene_id=scene_id,
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
        )
        secondary_reference_paths = self._build_secondary_reference_paths(
            reference_plan=reference_plan,
            anchor_image_path=anchor_image_path,
        )
        output_video_path = paths["video_path"]

        result = self._call_real_pipeline(
            prompt_text=prompt_text,
            negative_prompt=negative_prompt,
            anchor_image_path=anchor_image_path,
            secondary_reference_paths=secondary_reference_paths,
            output_video_path=output_video_path,
            generation_params=generation_params,
            control_weights=control_weights,
            policy=policy,
            prompt_bundle=prompt_bundle,
            paths=paths,
        )

        if not result.get("ok", False):
            result = self._fallback_generate_video(
                prompt_text=prompt_text,
                anchor_image_path=anchor_image_path,
                output_video_path=output_video_path,
                width=safe_int(generation_params.get("width", self.default_width), self.default_width),
                height=safe_int(generation_params.get("height", self.default_height), self.default_height),
                num_frames=safe_int(generation_params.get("num_frames", self.default_num_frames), self.default_num_frames),
                fps=safe_int(generation_params.get("fps", self.default_fps), self.default_fps),
                reason=result.get("error", "unknown_error"),
            )

        keyframe_paths = self._extract_and_store_keyframes(
            result=result,
            scene_dir=scene_dir,
            scene_id=scene_id,
            existing_paths=paths,
        )

        metadata = build_generated_metadata_from_constraints(
            prompt_bundle=prompt_bundle,
            reference_plan=reference_plan,
            generation_params=generation_params,
            backend_kind=self.backend_kind,
            video_path=result.get("video_path", output_video_path),
            keyframe_paths=keyframe_paths,
        )
        metadata["repair_mode"] = bool(repair_mode)
        metadata["scene_policy"] = policy
        metadata["control_weights"] = control_weights
        metadata["anchor_image_path"] = anchor_image_path
        metadata["secondary_reference_paths"] = secondary_reference_paths
        metadata["init_image_path"] = init_image_path
        metadata["negative_prompt"] = negative_prompt
        metadata["prompt_text"] = prompt_text
        metadata["used_init_reference"] = result.get("used_init_reference", init_image_path or "")
        metadata["used_fallback_anchor"] = result.get("used_fallback_anchor", False)
        metadata["quality_score"] = result.get("quality_score", 0.0)
        metadata["is_fallback_output"] = bool(result.get("is_fallback_output", False))
        metadata["fallback_reason"] = result.get("fallback_reason", "")
        metadata["generation_failed"] = bool(result.get("generation_failed", False))
        metadata["placeholder_conditioning_only"] = bool(result.get("placeholder_conditioning_only", False))
        metadata["has_real_init_image"] = bool(result.get("has_real_init_image", False))
        metadata["has_real_reference_assets"] = bool(result.get("has_real_reference_assets", False))
        metadata["reference_source_type"] = result.get("reference_source_type", "missing")
        metadata["semantic_evidence_status"] = result.get(
            "semantic_evidence_status",
            "missing" if bool(result.get("placeholder_conditioning_only", False)) else metadata.get("semantic_evidence_status", "missing")
        )

        if result.get("metadata"):
            for k, v in dict(result["metadata"]).items():
                metadata[k] = v

        out = {
            "ok": bool(result.get("ok", False)),
            "scene_id": scene_id,
            "video_path": result.get("video_path", output_video_path),
            "output_video_path": result.get("video_path", output_video_path),
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

    def config_output_dir(self) -> str:
        return ensure_dir(getattr(self, "output_dir", None) or "outputs/generated/svd")

    # ------------------------------------------------------------------
    # debug helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # pipeline load
    # ------------------------------------------------------------------

    def _torch_dtype(self):
        if torch is None:
            return None
        if str(self.dtype).lower() in {"fp16", "float16", "half"}:
            return torch.float16
        if str(self.dtype).lower() in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float32

    def _safe_cuda_cleanup(self):
        if torch is None:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_pipeline(self):
        if self.pipeline is not None:
            return
        if StableVideoDiffusionPipeline is None:
            raise RuntimeError(
                "diffusers StableVideoDiffusionPipeline is not available. "
                "Install a compatible diffusers version."
            )

        torch_dtype = self._torch_dtype()
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        if self.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        elif self.enable_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.to(self.device)

        if self.enable_attention_slicing and hasattr(self.pipeline, "enable_attention_slicing"):
            try:
                self.pipeline.enable_attention_slicing()
            except Exception:
                pass

        if self.enable_vae_slicing and hasattr(self.pipeline, "enable_vae_slicing"):
            try:
                self.pipeline.enable_vae_slicing()
            except Exception:
                pass

        if self.enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # generation params
    # ------------------------------------------------------------------

    def _choose_init_image_path(
        self,
        reference_plan: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str:
        ranked_refs = rank_reference_paths(reference_plan)
        prefer_prev_keyframe = bool(policy.get("prefer_previous_keyframe", True))

        if prefer_prev_keyframe:
            for path in ranked_refs:
                path = str(path or "")
                if path and os.path.isfile(path):
                    return path

        for path in ranked_refs:
            path = str(path or "")
            if path and os.path.isfile(path):
                return path

        return ""

    def _scene_output_paths(self, scene_dir: str, scene_id: str) -> Dict[str, str]:
        scene_name = os.path.basename(scene_id)
        out_dir = ensure_dir(scene_dir)
        return {
            "video_path": os.path.join(out_dir, f"{scene_name}.mp4"),
            "selected_keyframe": os.path.join(out_dir, f"{scene_name}_keyframe.png"),
            "first_frame": os.path.join(out_dir, f"{scene_name}_first.png"),
            "last_frame": os.path.join(out_dir, f"{scene_name}_last.png"),
        }

    def _build_secondary_reference_paths(
        self,
        reference_plan: Dict[str, Any],
        anchor_image_path: str,
    ) -> List[str]:
        ranked_refs = rank_reference_paths(reference_plan)
        out = []
        for p in ranked_refs:
            p = str(p or "")
            if not p or not os.path.isfile(p):
                continue
            if anchor_image_path and os.path.abspath(p) == os.path.abspath(anchor_image_path):
                continue
            out.append(p)
        return out[:4]

    # ------------------------------------------------------------------
    # placeholder anchor
    # ------------------------------------------------------------------

    def _ensure_anchor_image(
        self,
        prompt_text: str,
        scene_dir: str,
        init_image_path: str,
        scene_id: str,
        prompt_bundle: Dict[str, Any],
        repair_mode: bool,
    ) -> str:
        if init_image_path and os.path.isfile(init_image_path):
            return init_image_path

        anchor_path = os.path.join(scene_dir, f"{scene_id}_anchor.png")
        if os.path.isfile(anchor_path):
            return anchor_path

        self._create_placeholder_anchor(
            out_path=anchor_path,
            width=self.default_width,
            height=self.default_height,
            prompt_text=prompt_text,
            prompt_bundle=prompt_bundle,
            repair_mode=repair_mode,
        )
        return anchor_path

    def _create_placeholder_anchor(
        self,
        out_path: str,
        width: int,
        height: int,
        prompt_text: str,
        prompt_bundle: Dict[str, Any],
        repair_mode: bool,
    ) -> None:
        base = np.zeros((height, width, 3), dtype=np.uint8)
        contract = (prompt_bundle.get("prompt_contract", {}) or {})
        location = (contract.get("location_lock", {}) or {})
        style = (contract.get("style_lock", {}) or {})

        mood_text = " ".join(
            [
                str(location.get("name", "") or ""),
                str(style.get("palette", "") or ""),
                str(style.get("lighting", "") or ""),
                str(prompt_text or ""),
            ]
        ).lower()

        if any(k in mood_text for k in ["night", "dark", "shadow", "blue"]):
            bg = np.array([25, 35, 70], dtype=np.uint8)
        elif any(k in mood_text for k in ["sunset", "warm", "gold", "orange"]):
            bg = np.array([180, 110, 60], dtype=np.uint8)
        elif any(k in mood_text for k in ["forest", "green", "nature"]):
            bg = np.array([60, 120, 70], dtype=np.uint8)
        elif any(k in mood_text for k in ["snow", "ice", "white"]):
            bg = np.array([200, 210, 225], dtype=np.uint8)
        else:
            bg = np.array([90, 90, 105], dtype=np.uint8)

        base[:, :] = bg

        cx1, cy1 = int(width * 0.28), int(height * 0.18)
        cx2, cy2 = int(width * 0.72), int(height * 0.82)

        fg = np.array(
            [
                min(255, int(bg[0]) + 35),
                min(255, int(bg[1]) + 20),
                min(255, int(bg[2]) + 15),
            ],
            dtype=np.uint8,
        )
        base[cy1:cy2, cx1:cx2] = fg

        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        label = "repair anchor" if repair_mode else "scene anchor"
        text = f"{label}\n{prompt_text[:90]}"
        try:
            draw.rectangle([(8, 8), (width - 8, height - 8)], outline=(255, 255, 255), width=2)
            draw.text((16, 16), text, fill=(255, 255, 255), font=font)
        except Exception:
            pass

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)

    # ------------------------------------------------------------------
    # image utilities
    # ------------------------------------------------------------------

    def _load_reference_image(self, image_path: str) -> Optional[Image.Image]:
        if not image_path or not os.path.isfile(image_path):
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception:
            return None

    def _make_generator(self):
        if torch is None:
            return None
        if self.seed is None:
            return None
        try:
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                return torch.Generator(device="cuda").manual_seed(int(self.seed))
            return torch.Generator().manual_seed(int(self.seed))
        except Exception:
            return None

    def _build_svd_call_kwargs(
        self,
        image: Image.Image,
        generation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        IMPORTANT:
        Do NOT pass negative_prompt here.
        StableVideoDiffusionPipeline in many diffusers versions does not accept it.
        """
        kwargs = {
            "image": image,
            "height": safe_int(generation_params.get("height", self.default_height), self.default_height),
            "width": safe_int(generation_params.get("width", self.default_width), self.default_width),
            "num_frames": safe_int(generation_params.get("num_frames", self.default_num_frames), self.default_num_frames),
            "fps": safe_int(generation_params.get("fps", self.default_fps), self.default_fps),
            "decode_chunk_size": safe_int(self.decode_chunk_size, 8),
            "motion_bucket_id": safe_int(
                generation_params.get("motion_bucket_id", self.default_motion_bucket_id),
                self.default_motion_bucket_id,
            ),
            "noise_aug_strength": safe_float(
                generation_params.get("noise_aug_strength", self.default_noise_aug_strength),
                self.default_noise_aug_strength,
            ),
            "num_inference_steps": safe_int(
                generation_params.get("num_inference_steps", self.default_num_inference_steps),
                self.default_num_inference_steps,
            ),
            "generator": self._make_generator(),
        }

        return kwargs

    def _to_pil(self, frame: Any) -> Optional[Image.Image]:
        if frame is None:
            return None

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if hasattr(frame, "cpu") and hasattr(frame, "numpy"):
            try:
                arr = frame.detach().cpu().numpy()
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                return Image.fromarray(arr).convert("RGB")
            except Exception:
                return None

        if isinstance(frame, np.ndarray):
            try:
                arr = frame
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                return Image.fromarray(arr).convert("RGB")
            except Exception:
                return None

        return None

    # ------------------------------------------------------------------
    # real pipeline call
    # ------------------------------------------------------------------

    def _call_real_pipeline(
        self,
        prompt_text: str,
        negative_prompt: str,
        anchor_image_path: str,
        secondary_reference_paths: List[str],
        output_video_path: str,
        generation_params: Dict[str, Any],
        control_weights: Dict[str, Any],
        policy: Dict[str, Any],
        prompt_bundle: Dict[str, Any],
        paths: Dict[str, str],
    ) -> Dict[str, Any]:
        used_init_reference = anchor_image_path if os.path.isfile(anchor_image_path) else ""
        used_fallback_anchor = bool(used_init_reference and used_init_reference.endswith("_anchor.png"))
        placeholder_conditioning_only = bool(used_fallback_anchor)
        has_real_init_image = bool(used_init_reference and not used_fallback_anchor)
        has_real_reference_assets = bool(
            has_real_init_image or any(os.path.isfile(p) for p in (secondary_reference_paths or []))
        )

        try:
            self._load_pipeline()
            init_img = self._load_reference_image(anchor_image_path)
            if init_img is None:
                return self._attach_debug_error(
                    {
                        "ok": False,
                        "error": "No valid init / anchor image found for SVD.",
                        "video_path": "",
                        "selected_keyframe": "",
                        "first_frame": "",
                        "last_frame": "",
                        "quality_score": 0.0,
                        "used_init_reference": used_init_reference,
                        "used_fallback_anchor": used_fallback_anchor,
                        "generation_failed": True,
                        "is_fallback_output": False,
                    },
                    stage="load_anchor",
                    err="missing_or_invalid_anchor",
                    extra={
                        "anchor_image_path": anchor_image_path,
                        "model_id": self.model_id,
                        "device": self.device,
                    },
                )

            call_kwargs = self._build_svd_call_kwargs(
                image=init_img,
                generation_params=generation_params,
            )

            try:
                output = self.pipeline(**call_kwargs)
                frames = []
                if hasattr(output, "frames"):
                    frames = output.frames[0] if isinstance(output.frames, list) and len(output.frames) > 0 else output.frames
                elif isinstance(output, dict) and "frames" in output:
                    frames = output["frames"]
                else:
                    frames = []
            except Exception as e:
                return self._attach_debug_error(
                    {
                        "ok": False,
                        "error": f"SVD generation failed during pipeline call: {str(e)}",
                        "video_path": "",
                        "selected_keyframe": "",
                        "first_frame": "",
                        "last_frame": "",
                        "quality_score": 0.0,
                        "used_init_reference": used_init_reference,
                        "used_fallback_anchor": used_fallback_anchor,
                        "generation_failed": True,
                        "is_fallback_output": False,
                    },
                    stage="real_pipeline",
                    err=e,
                    extra={
                        "model_id": self.model_id,
                        "device": self.device,
                        "negative_prompt_tracked_but_not_passed": negative_prompt,
                    },
                )
        except Exception as e:
            return self._attach_debug_error(
                {
                    "ok": False,
                    "error": f"SVD generation failed: {str(e)}",
                    "video_path": "",
                    "selected_keyframe": "",
                    "first_frame": "",
                    "last_frame": "",
                    "quality_score": 0.0,
                    "used_init_reference": used_init_reference,
                    "used_fallback_anchor": used_fallback_anchor,
                    "generation_failed": True,
                    "is_fallback_output": False,
                },
                stage="real_pipeline",
                err=e,
                extra={
                    "model_id": self.model_id,
                    "device": self.device,
                    "negative_prompt_tracked_but_not_passed": negative_prompt,
                },
            )

        saved = self._save_frames_and_video(frames, paths)
        if not saved.get("ok", False):
            return self._attach_debug_error(
                {
                    **saved,
                    "quality_score": 0.0,
                    "used_init_reference": used_init_reference,
                    "used_fallback_anchor": used_fallback_anchor,
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

        if placeholder_conditioning_only:
            quality_score = 0.10
            reference_source_type = "placeholder_only"
            semantic_evidence_status = "missing"
        elif used_fallback_anchor:
            quality_score = 0.20
            reference_source_type = "mixed"
            semantic_evidence_status = "missing"
        elif used_init_reference:
            quality_score = 0.88
            reference_source_type = "real"
            semantic_evidence_status = "missing"
        else:
            quality_score = 0.45
            reference_source_type = "missing"
            semantic_evidence_status = "missing"

        return {
            **saved,
            "quality_score": quality_score,
            "used_init_reference": used_init_reference,
            "used_fallback_anchor": used_fallback_anchor,
            "placeholder_conditioning_only": placeholder_conditioning_only,
            "has_real_init_image": has_real_init_image,
            "has_real_reference_assets": has_real_reference_assets,
            "reference_source_type": reference_source_type,
            "semantic_evidence_status": semantic_evidence_status,
            "generation_failed": False,
            "is_fallback_output": False,
        }

    # ------------------------------------------------------------------
    # fallback debug artifact
    # ------------------------------------------------------------------

    def _fallback_generate_video(
        self,
        prompt_text: str,
        anchor_image_path: str,
        output_video_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        reason: str,
    ) -> Dict[str, Any]:
        try:
            img = Image.open(anchor_image_path).convert("RGB").resize((width, height))
            arr = np.array(img).astype(np.float32)

            frames: List[Image.Image] = []
            for i in range(max(1, num_frames)):
                delta = ((i % 5) - 2) * 2.0
                frame = np.clip(arr + delta, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(frame))

            self._save_frames_to_video(
                frames=frames,
                output_video_path=output_video_path,
                fps=fps,
            )

            return self._attach_debug_error(
                {
                    "ok": False,
                    "error": f"Fallback output used because real SVD generation failed: {reason}",
                    "video_path": output_video_path,
                    "frames": frames,
                    "fallback_reason": reason,
                    "quality_score": 0.05,
                    "is_fallback_output": True,
                    "generation_failed": True,
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
    ) -> Dict[str, str]:
        frames = result.get("frames", None)

        if not isinstance(frames, list) or len(frames) == 0:
            first_path = str(result.get("first_frame", "") or existing_paths.get("first_frame", ""))
            middle_path = str(result.get("selected_keyframe", "") or existing_paths.get("selected_keyframe", ""))
            last_path = str(result.get("last_frame", "") or existing_paths.get("last_frame", ""))
            best_path = middle_path
            return {
                "first_frame_path": first_path,
                "middle_frame_path": middle_path,
                "last_frame_path": last_path,
                "best_keyframe_path": best_path,
            }

        pil_frames = [self._to_pil(x) for x in frames if self._to_pil(x) is not None]
        if not pil_frames:
            return {
                "first_frame_path": "",
                "middle_frame_path": "",
                "last_frame_path": "",
                "best_keyframe_path": "",
            }

        first_idx = 0
        middle_idx = len(pil_frames) // 2
        last_idx = len(pil_frames) - 1

        first_path = existing_paths["first_frame"]
        middle_path = existing_paths["selected_keyframe"]
        last_path = existing_paths["last_frame"]
        best_path = middle_path

        pil_frames[first_idx].save(first_path)
        pil_frames[middle_idx].save(middle_path)
        pil_frames[last_idx].save(last_path)

        return {
            "first_frame_path": first_path,
            "middle_frame_path": middle_path,
            "last_frame_path": last_path,
            "best_keyframe_path": best_path,
        }

    def _save_frames_and_video(
        self,
        frames: List[Any],
        paths: Dict[str, str],
    ) -> Dict[str, Any]:
        pil_frames = [self._to_pil(f) for f in (frames or [])]
        pil_frames = [f for f in pil_frames if f is not None]

        if not pil_frames:
            return {
                "ok": False,
                "error": "No valid frames returned by SVD pipeline.",
                "video_path": "",
                "selected_keyframe": "",
                "first_frame": "",
                "last_frame": "",
                "frames": [],
            }

        width, height = pil_frames[0].size
        video_path = paths["video_path"]
        fps = self.default_fps

        try:
            self._save_frames_to_video(pil_frames, video_path, fps=fps)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to save video: {e}",
                "video_path": "",
                "selected_keyframe": "",
                "first_frame": "",
                "last_frame": "",
                "frames": pil_frames,
            }

        first_frame = paths["first_frame"]
        selected_keyframe = paths["selected_keyframe"]
        last_frame = paths["last_frame"]

        try:
            pil_frames[0].save(first_frame)
            pil_frames[len(pil_frames) // 2].save(selected_keyframe)
            pil_frames[-1].save(last_frame)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to save keyframes: {e}",
                "video_path": video_path,
                "selected_keyframe": "",
                "first_frame": "",
                "last_frame": "",
                "frames": pil_frames,
            }

        return {
            "ok": True,
            "video_path": video_path,
            "selected_keyframe": selected_keyframe,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames": pil_frames,
            "width": width,
            "height": height,
        }