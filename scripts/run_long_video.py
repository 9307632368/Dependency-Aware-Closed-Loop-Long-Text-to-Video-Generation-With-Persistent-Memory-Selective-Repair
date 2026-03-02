# scripts/run_long_video.py
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.run_full_pipeline import run_full_pipeline
from src.generation.backend.factory import build_generation_backend
from src.utils.logger import SimpleLogger
from src.utils.io import ensure_dir


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def dumps_pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_run_dirs(base_out_dir: Path):
    logs_dir = base_out_dir / "logs"
    raw_dir = base_out_dir / "raw_llm"
    ensure_dir(str(base_out_dir))
    ensure_dir(str(logs_dir))
    ensure_dir(str(raw_dir))
    return logs_dir, raw_dir


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
    """
    Priority:
    1. explicit prompt file from CLI
    2. data/prompts/prompt_001.txt
    3. first .txt file inside input_prompt_dir
    """
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


def main():
    # -----------------------------
    # load settings
    # -----------------------------
    settings_path = "configs/settings.yaml"
    settings = load_yaml(settings_path)
    resolved = resolve_paths(settings)

    input_prompt_dir = resolved["input_prompt_dir"]
    base_output_root = resolved["output_dir"]
    backend_config_path = resolved["backend_config_path"]

    # optional CLI:
    # python scripts/run_long_video.py data/prompts/my_story.txt
    prompt_file = sys.argv[1] if len(sys.argv) > 1 else ""
    prompt_path = pick_prompt_file(input_prompt_dir, prompt_file)

    prompt_text = read_text(prompt_path)
    backend_config = load_yaml(backend_config_path)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_out_dir = Path(base_output_root) / run_id
    logs_dir, raw_dir = build_run_dirs(base_out_dir)

    log_file = logs_dir / "run.log"
    logger = SimpleLogger(log_file=str(log_file))

    logger.info("Starting long video pipeline")
    logger.info(f"Input prompt file: {prompt_path}")
    logger.info(f"Settings path: {settings_path}")
    logger.info(f"Backend config path: {backend_config_path}")

    # save run inputs for reproducibility
    (base_out_dir / "input_prompt.txt").write_text(prompt_text, encoding="utf-8")
    (base_out_dir / "backend_config_used.yaml").write_text(
        yaml.safe_dump(backend_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )
    (base_out_dir / "settings_used.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )

    # build backend
    backend = build_generation_backend(backend_config)
    logger.info(f"Backend built successfully: {backend_config.get('name', 'unknown')}")

    # run full pipeline
    out = run_full_pipeline(
        prompt_text=prompt_text,
        backend=backend,
        output_dir=str(base_out_dir),
        config_path=settings_path,
        logger=logger,
        raw_save_dir=str(raw_dir),
        scorer=None,
        max_retries_per_scene=2,
    )

    # save final summary
    (base_out_dir / "final_output.json").write_text(
        dumps_pretty(out),
        encoding="utf-8"
    )

    if out.get("ok", False):
        logger.info("Long video pipeline completed successfully")
        print("[OK] Long video pipeline completed successfully")
        print(f"Run directory: {base_out_dir}")
    else:
        logger.error("Long video pipeline finished with errors")
        print("[ERROR] Long video pipeline finished with errors")
        print(f"Run directory: {base_out_dir}")
        print(f"Stage: {out.get('stage', 'unknown')}")
        print(f"Error: {out.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()