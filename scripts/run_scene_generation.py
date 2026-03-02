# scripts/run_scene_generation.py
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import yaml
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.run_text_pipeline import run_text_planning_pipeline
from src.pipeline.run_generation_pipeline import run_generation_pipeline
from src.generation.backend.factory import build_backend
from src.continuity.consistency_scorer import ConsistencyScorer
from src.utils.logger import SimpleLogger
from src.utils.io import ensure_dir


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]

    cls_name = obj.__class__.__name__
    mod_name = getattr(obj.__class__, "__module__", "")

    if cls_name == "Image" or mod_name.startswith("PIL."):
        return f"<nonserializable:{cls_name}>"

    if mod_name.startswith("numpy"):
        return f"<nonserializable:{cls_name}>"

    return f"<nonserializable:{cls_name}>"


def dumps_pretty(obj) -> str:
    return json.dumps(_json_safe(obj), indent=2, ensure_ascii=False)


def build_run_dirs(base_out_dir: Path):
    logs_dir = base_out_dir / "logs"
    raw_dir = base_out_dir / "raw_llm"
    generation_dir = base_out_dir / "generation"
    generated_media_dir = base_out_dir / "generated"

    ensure_dir(str(base_out_dir))
    ensure_dir(str(logs_dir))
    ensure_dir(str(raw_dir))
    ensure_dir(str(generation_dir))
    ensure_dir(str(generated_media_dir))

    return logs_dir, raw_dir, generation_dir, generated_media_dir


def resolve_paths(settings: dict):
    paths_cfg = settings.get("paths", {})

    input_prompt_dir = paths_cfg.get("input_prompt_dir", "data/prompts")
    output_dir = paths_cfg.get("output_dir", "outputs/runs")
    backend_config_path = paths_cfg.get("backend_config", "configs/generation/backend.yaml")

    return {
        "input_prompt_dir": input_prompt_dir,
        "output_dir": output_dir,
        "backend_config_path": backend_config_path,
    }


def pick_prompt_file(input_prompt_dir: str, prompt_file: str = "") -> str:
    if prompt_file:
        if os.path.exists(prompt_file):
            return prompt_file
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    default_prompt = os.path.join(input_prompt_dir, "prompt_001.txt")
    if os.path.exists(default_prompt):
        return default_prompt

    prompt_dir_path = Path(input_prompt_dir)
    if prompt_dir_path.exists():
        txt_files = sorted(prompt_dir_path.glob("*.txt"))
        if txt_files:
            return str(txt_files[0])

    raise FileNotFoundError(
        f"No prompt file found. Checked: {default_prompt} and txt files inside {input_prompt_dir}"
    )


def inject_run_specific_backend_dirs(cfg: dict, run_root: Path) -> dict:
    """
    Force all generated media into the current run directory.

    Result:
    outputs/runs/scene_run_xxx/
        logs/
        raw_llm/
        generation/
        generated/
            cogvideox/
            svd/
    """
    cfg = copy.deepcopy(cfg or {})

    generated_root = run_root / "generated"
    cog_dir = generated_root / "cogvideox"
    svd_dir = generated_root / "svd"

    ensure_dir(str(generated_root))
    ensure_dir(str(cog_dir))
    ensure_dir(str(svd_dir))

    # top-level output_dir sometimes used by single backends
    cfg["output_dir"] = str(generated_root)

    # ensure nested backends section exists
    if "backends" not in cfg or not isinstance(cfg["backends"], dict):
        cfg["backends"] = {}

    if "cogvideox" not in cfg["backends"] or not isinstance(cfg["backends"]["cogvideox"], dict):
        cfg["backends"]["cogvideox"] = {}
    if "svd" not in cfg["backends"] or not isinstance(cfg["backends"]["svd"], dict):
        cfg["backends"]["svd"] = {}

    cfg["backends"]["cogvideox"]["output_dir"] = str(cog_dir)
    cfg["backends"]["svd"]["output_dir"] = str(svd_dir)

    # if the selected backend is directly one backend, support that too
    backend_name = str(cfg.get("backend", "") or "").strip().lower()
    if backend_name == "cogvideox":
        cfg["output_dir"] = str(cog_dir)
    elif backend_name == "svd":
        cfg["output_dir"] = str(svd_dir)
    else:
        cfg["output_dir"] = str(generated_root)

    return cfg


def main():
    settings_path = "configs/settings.yaml"
    settings = load_yaml(settings_path)
    resolved = resolve_paths(settings)

    input_prompt_dir = resolved["input_prompt_dir"]
    base_output_root = resolved["output_dir"]
    backend_config_path = resolved["backend_config_path"]

    prompt_file = sys.argv[1] if len(sys.argv) > 1 else ""
    prompt_path = pick_prompt_file(input_prompt_dir, prompt_file)

    prompt_text = read_text(prompt_path)
    backend_config = load_yaml(backend_config_path)

    run_id = datetime.now().strftime("scene_run_%Y%m%d_%H%M%S")
    base_out_dir = Path(base_output_root) / run_id
    logs_dir, raw_dir, generation_dir, generated_media_dir = build_run_dirs(base_out_dir)

    # inject run-scoped generated-media dirs
    backend_config = inject_run_specific_backend_dirs(backend_config, base_out_dir)

    log_file = logs_dir / "scene_generation.log"
    logger = SimpleLogger(log_file=str(log_file))

    logger.info("Starting scene generation pipeline")
    logger.info(f"Input prompt file: {prompt_path}")
    logger.info(f"Settings path: {settings_path}")
    logger.info(f"Backend config path: {backend_config_path}")
    logger.info(f"Run directory: {base_out_dir}")
    logger.info(f"Generated media directory: {generated_media_dir}")

    (base_out_dir / "input_prompt.txt").write_text(prompt_text, encoding="utf-8")
    (base_out_dir / "backend_config_used.yaml").write_text(
        yaml.safe_dump(backend_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )
    (base_out_dir / "settings_used.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # STEP 1: text planning
    text_out = run_text_planning_pipeline(
        prompt_text=prompt_text,
        config_path=settings_path,
        logger=logger,
        raw_save_dir=str(raw_dir)
    )

    (base_out_dir / "text_planning_output.json").write_text(
        dumps_pretty(text_out),
        encoding="utf-8"
    )

    if not text_out.get("ok", False):
        logger.error("Text planning failed")
        print("[ERROR] Text planning failed")
        print(f"Run directory: {base_out_dir}")
        return

    scene_packets = text_out.get("scene_packets", [])
    (base_out_dir / "scene_packets.json").write_text(
        dumps_pretty({"scene_packets": scene_packets}),
        encoding="utf-8"
    )

    logger.info(f"Scene packets built: {len(scene_packets)}")

    # STEP 2: backend build validation
    backend_runtime_cfg = dict(settings or {})
    backend_runtime_cfg.update(backend_config or {})
    if "backend" not in backend_runtime_cfg:
        backend_runtime_cfg["backend"] = (
            backend_config.get("backend")
            or backend_config.get("name")
            or "hybrid"
        )

    backend_runtime_cfg = inject_run_specific_backend_dirs(backend_runtime_cfg, base_out_dir)

    backend = build_backend(backend_runtime_cfg)
    logger.info(
        f"Backend built successfully: {backend_runtime_cfg.get('backend', 'unknown')}"
    )

    # STEP 3: scene-wise generation
    scorer = ConsistencyScorer(settings)

    generation_cfg = dict(settings or {})
    generation_cfg.update(backend_config or {})
    generation_cfg = inject_run_specific_backend_dirs(generation_cfg, base_out_dir)

    generation_cfg.setdefault("output", {})
    generation_cfg["output"]["generation_output_dir"] = str(generation_dir)

    generation_out = run_generation_pipeline(
        scene_packets=scene_packets,
        config=generation_cfg,
    )

    (base_out_dir / "scene_generation_output.json").write_text(
        dumps_pretty(generation_out),
        encoding="utf-8"
    )

    if generation_out.get("ok", False):
        logger.info("Scene generation completed successfully")
        print("[OK] Scene generation completed successfully")
        print(f"Run directory: {base_out_dir}")
    else:
        logger.error("Scene generation completed with errors")
        print("[ERROR] Scene generation completed with errors")
        print(f"Run directory: {base_out_dir}")

        failed = generation_out.get("failed_scene_results", [])
        if failed:
            print("Failed scenes:")
            for item in failed:
                item = item or {}
                scene_id = item.get("scene_id", "unknown")
                score_report = item.get("score_report", {}) or {}
                drift_report = item.get("drift_report", {}) or {}
                err = (
                    item.get("error")
                    or score_report.get("error")
                    or ", ".join(drift_report.get("drift_flags", []))
                    or "unknown error"
                )
                print(f"  - {scene_id}: {err}")

    pipeline_summary = generation_out.get("pipeline_summary", {}) or {}
    print("\nGeneration summary")
    print("------------------")
    print(f"Scenes: {pipeline_summary.get('num_scenes', 0)}")
    print(f"Accepted: {pipeline_summary.get('num_accepted', 0)}")
    print(f"Failed: {pipeline_summary.get('num_failed', 0)}")
    print(f"Average score: {pipeline_summary.get('avg_score', 0.0)}")
    print(f"Backends used: {pipeline_summary.get('backend_counts', {})}")
    print(f"Generated media root: {generated_media_dir}")


if __name__ == "__main__":
    main()