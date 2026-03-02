# scripts/run_single_prompt.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from pathlib import Path
from datetime import datetime

from src.main import run_pipeline
from src.utils.logger import SimpleLogger
from src.utils.io import ensure_dir, write_json, write_text


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main():
    prompt_path = "data/prompts/prompt_001.txt"
    config_path = "configs/settings.yaml"

    prompt_text = read_text(prompt_path)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_out_dir = Path("outputs/runs") / run_id
    raw_dir = base_out_dir / "raw_llm"
    ensure_dir(str(base_out_dir))
    ensure_dir(str(raw_dir))

    logger = SimpleLogger(str(base_out_dir / "logs" / "run.log"))

    logger.info(f"Using prompt file: {prompt_path}")
    logger.info("Launching pipeline")

    out = run_pipeline(
        prompt_text,
        config_path=config_path,
        logger=logger,
        raw_save_dir=str(raw_dir)
    )

    write_text(str(base_out_dir / "input_prompt.txt"), prompt_text)

    if out["ok"]:
        result = out["result"]

        scenes_only = {
            "scenes": result.get("scenes", []),
            "global_notes": result.get("global_notes", {})
        }

        deps_only = {
            "dependencies": result.get("dependencies", [])
        }

        write_json(str(base_out_dir / "scenes.json"), scenes_only)
        write_json(str(base_out_dir / "dependencies.json"), deps_only)
        write_json(str(base_out_dir / "scenes_dependencies.json"), result)

        logger.info("Saved scenes.json, dependencies.json, scenes_dependencies.json")
        print(f"[OK] Saved outputs to: {base_out_dir}")
    else:
        write_json(str(base_out_dir / "error.json"), out)
        logger.error(f"Pipeline failed at stage={out.get('stage')}")
        print(f"[FAIL] Stage={out.get('stage')} Error={out.get('error')}")
        print(f"Saved: {base_out_dir / 'error.json'}")


if __name__ == "__main__":
    main()