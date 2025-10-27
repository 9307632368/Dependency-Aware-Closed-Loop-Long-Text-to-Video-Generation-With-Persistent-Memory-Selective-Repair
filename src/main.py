# src/main.py
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

from src.llm.client import LLMClient, load_llm_config_from_env
from src.llm.prompts import PromptPack
from src.text.segmentation import run_segmentation_with_verification
from src.text.dependency import run_dependency_detection
from src.utils.logger import SimpleLogger
from src.utils.io import ensure_dir


def load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def run_pipeline(
    prompt_text: str,
    config_path: str = "configs/settings.yaml",
    logger: SimpleLogger = None,
    raw_save_dir: str = None
) -> Dict[str, Any]:
    """
    LLM text pipeline only:
    1. segmentation
    2. dependency detection

    Returns:
    {
        "ok": True/False,
        "result": {
            "scenes": [...],
            "dependencies": [...],
            "global_notes": {...}
        }
    }
    """
    load_dotenv()

    if logger is None:
        logger = SimpleLogger()

    cfg = load_yaml(config_path)

    paths_cfg = cfg.get("paths", {})
    template_prompt_dir = paths_cfg.get("template_prompt_dir", "configs/prompts")
    out_dir = paths_cfg.get("output_dir", "outputs/runs")
    ensure_dir(out_dir)

    llm_defaults_model = cfg.get("llm", {}).get("model", "")
    llm_cfg = load_llm_config_from_env(default_model=llm_defaults_model)

    llm_cfg.temperature = float(cfg.get("llm", {}).get("temperature", getattr(llm_cfg, "temperature", 0.2)))
    llm_cfg.top_p = float(cfg.get("llm", {}).get("top_p", getattr(llm_cfg, "top_p", 1.0)))
    llm_cfg.max_tokens = int(cfg.get("llm", {}).get("max_tokens", getattr(llm_cfg, "max_tokens", 2048)))

    llm = LLMClient(llm_cfg)
    prompts = PromptPack(template_prompt_dir)

    logger.info("Starting segmentation stage")
    ok, seg_obj, err = run_segmentation_with_verification(
        llm=llm,
        prompts=prompts,
        long_prompt=prompt_text,
        max_fix_rounds=int(cfg.get("pipeline", {}).get("verify_rounds", 2)),
        raw_save_dir=raw_save_dir
    )
    if not ok:
        logger.error(f"Segmentation failed: {err}")
        return {
            "ok": False,
            "stage": "segmentation",
            "error": err,
            "partial": seg_obj
        }

    logger.info("Segmentation completed successfully")

    logger.info("Starting dependency detection stage")
    ok2, dep_obj, err2 = run_dependency_detection(
        llm=llm,
        prompts=prompts,
        scenes_json=seg_obj,
        max_retries=int(cfg.get("pipeline", {}).get("dependency_retries", 2)),
        raw_save_dir=raw_save_dir
    )
    if not ok2:
        logger.error(f"Dependency detection failed: {err2}")
        return {
            "ok": False,
            "stage": "dependency",
            "error": err2,
            "partial": dep_obj
        }

    logger.info("Dependency detection completed successfully")

    seg_obj.setdefault("global_notes", {})
    seg_obj["dependencies"] = dep_obj.get("dependencies", [])
    seg_obj["global_notes"]["num_scenes"] = len(seg_obj.get("scenes", []))

    return {
        "ok": True,
        "result": seg_obj
    }