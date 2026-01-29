# src/video/stitch.py
from __future__ import annotations

import os
from typing import Dict, Any, List


def _safe_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {}


def _safe_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return []


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _ensure_dir(path: str) -> str:
    path = _safe_text(path)
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def stitch_scene_clips(
    generation_output: Dict[str, Any],
    output_dir: str,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Stitch accepted scene clips into one final video.

    Compatible with both:
    - old generation pipeline outputs
    - rewritten generation pipeline outputs

    Expected modern keys:
    - accepted_scene_results
    - scene_results
    - generation_result.video_path / output_video_path

    Older fallback keys:
    - results
    - clip_path
    - output_path
    """
    generation_output = _safe_dict(generation_output)
    config = _safe_dict(config)

    output_dir = _ensure_dir(output_dir)
    final_video_path = os.path.join(output_dir, "final_video.mp4")

    clip_paths = _extract_clip_paths(generation_output)

    if not clip_paths:
        return {
            "ok": False,
            "video_path": "",
            "clips_used": [],
            "reason": "no_scene_clips_found",
        }

    # Try ffmpeg concat first
    ffmpeg_out = _stitch_with_ffmpeg(
        clip_paths=clip_paths,
        output_video_path=final_video_path,
        config=config,
    )
    if ffmpeg_out.get("ok", False):
        return ffmpeg_out

    # Fallback to imageio frame concatenation
    imageio_out = _stitch_with_imageio(
        clip_paths=clip_paths,
        output_video_path=final_video_path,
        config=config,
    )
    if imageio_out.get("ok", False):
        return imageio_out

    return {
        "ok": False,
        "video_path": "",
        "clips_used": clip_paths,
        "reason": "all_stitch_methods_failed",
        "ffmpeg_error": ffmpeg_out.get("reason", ""),
        "imageio_error": imageio_out.get("reason", ""),
    }


def _extract_clip_paths(generation_output: Dict[str, Any]) -> List[str]:
    """
    Support multiple possible output layouts.
    """
    clip_paths: List[str] = []

    # modern preferred path
    accepted_scene_results = _safe_list(generation_output.get("accepted_scene_results", []))
    if accepted_scene_results:
        for item in accepted_scene_results:
            item = _safe_dict(item)
            generation_result = _safe_dict(item.get("generation_result", {}))

            candidates = [
                _safe_text(generation_result.get("video_path", "")),
                _safe_text(generation_result.get("output_video_path", "")),
                _safe_text(item.get("clip_path", "")),
                _safe_text(item.get("output_path", "")),
            ]
            for c in candidates:
                if c:
                    clip_paths.append(c)
                    break

    # fallback: all scene_results but only accepted ones
    if not clip_paths:
        scene_results = _safe_list(generation_output.get("scene_results", []))
        for item in scene_results:
            item = _safe_dict(item)
            if not bool(item.get("accepted", False)):
                continue
            generation_result = _safe_dict(item.get("generation_result", {}))
            candidates = [
                _safe_text(generation_result.get("video_path", "")),
                _safe_text(generation_result.get("output_video_path", "")),
                _safe_text(item.get("clip_path", "")),
                _safe_text(item.get("output_path", "")),
            ]
            for c in candidates:
                if c:
                    clip_paths.append(c)
                    break

    # older structure fallback
    if not clip_paths:
        results = _safe_list(generation_output.get("results", []))
        for item in results:
            item = _safe_dict(item)
            candidates = [
                _safe_text(item.get("clip_path", "")),
                _safe_text(item.get("output_path", "")),
                _safe_text(_safe_dict(item.get("generation_result", {})).get("video_path", "")),
            ]
            for c in candidates:
                if c:
                    clip_paths.append(c)
                    break

    # keep only existing files, preserve order, dedupe
    final_paths: List[str] = []
    seen = set()
    for path in clip_paths:
        path = _safe_text(path)
        if not path:
            continue
        if not os.path.isfile(path):
            continue
        if path not in seen:
            seen.add(path)
            final_paths.append(path)

    return final_paths


def _stitch_with_ffmpeg(
    clip_paths: List[str],
    output_video_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import subprocess
        import tempfile

        concat_lines = []
        for path in clip_paths:
            # ffmpeg concat demuxer needs escaped single quotes
            safe_path = path.replace("'", r"'\''")
            concat_lines.append(f"file '{safe_path}'")

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            concat_file = f.name
            f.write("\n".join(concat_lines))

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            output_video_path,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            os.remove(concat_file)
        except Exception:
            pass

        if proc.returncode == 0 and os.path.isfile(output_video_path):
            return {
                "ok": True,
                "video_path": output_video_path,
                "clips_used": clip_paths,
                "method": "ffmpeg_concat_copy",
            }

        return {
            "ok": False,
            "video_path": "",
            "clips_used": clip_paths,
            "reason": proc.stderr[-1000:] if proc.stderr else "ffmpeg_failed",
        }
    except Exception as e:
        return {
            "ok": False,
            "video_path": "",
            "clips_used": clip_paths,
            "reason": f"ffmpeg_exception: {e}",
        }


def _stitch_with_imageio(
    clip_paths: List[str],
    output_video_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Slower fallback that re-encodes by reading all frames.
    """
    try:
        import imageio.v2 as imageio

        fps = int(_safe_dict(config.get("stitching", {})).get("fps", 8) or 8)

        writer = imageio.get_writer(output_video_path, fps=fps)

        for clip in clip_paths:
            try:
                reader = imageio.get_reader(clip)
                for frame in reader:
                    writer.append_data(frame)
                reader.close()
            except Exception:
                # skip broken clip but continue if possible
                continue

        writer.close()

        if os.path.isfile(output_video_path):
            return {
                "ok": True,
                "video_path": output_video_path,
                "clips_used": clip_paths,
                "method": "imageio_reencode",
            }

        return {
            "ok": False,
            "video_path": "",
            "clips_used": clip_paths,
            "reason": "imageio_no_output",
        }
    except Exception as e:
        return {
            "ok": False,
            "video_path": "",
            "clips_used": clip_paths,
            "reason": f"imageio_exception: {e}",
        }