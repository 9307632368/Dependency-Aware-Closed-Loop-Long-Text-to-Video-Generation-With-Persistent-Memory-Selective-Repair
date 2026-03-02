import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# scripts/run_batch_prompts.py
from pathlib import Path
from datetime import datetime
import json

from src.main import run_pipeline, ensure_dir
from src.utils.logger import SimpleLogger


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    prompt_dir = Path("data/prompts")
    config_path = "configs/settings.yaml"

    prompt_files = sorted(prompt_dir.glob("*.txt"))
    if not prompt_files:
        print("No .txt prompt files found in data/prompts/")
        return

    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_out_dir = Path("outputs/runs") / batch_id
    ensure_dir(str(batch_out_dir))

    logger = SimpleLogger(str(batch_out_dir / "logs" / "batch.log"))
    logger.info(f"Found {len(prompt_files)} prompt files")

    summary = []

    for prompt_file in prompt_files:
        logger.info(f"Processing {prompt_file.name}")

        prompt_text = read_text(prompt_file)

        run_name = prompt_file.stem
        run_dir = batch_out_dir / run_name
        ensure_dir(str(run_dir))

        out = run_pipeline(prompt_text, config_path=config_path, logger=logger)

        (run_dir / "input_prompt.txt").write_text(prompt_text, encoding="utf-8")

        if out["ok"]:
            result = out["result"]

            scenes_only = {
                "scenes": result.get("scenes", []),
                "global_notes": result.get("global_notes", {})
            }

            deps_only = {
                "dependencies": result.get("dependencies", [])
            }

            write_json(run_dir / "scenes.json", scenes_only)
            write_json(run_dir / "dependencies.json", deps_only)
            write_json(run_dir / "scenes_dependencies.json", result)

            summary.append({
                "file": prompt_file.name,
                "status": "ok",
                "num_scenes": len(result.get("scenes", []))
            })

            logger.info(f"Completed {prompt_file.name}")
        else:
            write_json(run_dir / "error.json", out)

            summary.append({
                "file": prompt_file.name,
                "status": "failed",
                "stage": out.get("stage"),
                "error": out.get("error")
            })

            logger.error(f"Failed {prompt_file.name}")

    write_json(batch_out_dir / "summary.json", {"runs": summary})
    logger.info("Batch run completed")
    print(f"[DONE] Batch results saved to: {batch_out_dir}")


if __name__ == "__main__":
    main()