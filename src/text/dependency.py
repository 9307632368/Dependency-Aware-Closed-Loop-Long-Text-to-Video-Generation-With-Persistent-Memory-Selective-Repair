# src/text/dependency.py
import json
from typing import Dict, Any, Tuple
from src.llm.client import LLMClient
from src.llm.prompts import PromptPack, render_template
from src.llm.parsing import extract_first_json_object


def repair_json_with_llm(llm: LLMClient, prompts: PromptPack, broken_output: str) -> Tuple[bool, Dict[str, Any], str]:
    repair_pair = prompts.load_pair("json_repair_system.txt", "json_repair_user.txt")
    repair_system = repair_pair["system"]
    repair_user = render_template(repair_pair["user"], {"BROKEN_OUTPUT": broken_output})

    ok, raw, err = llm.chat(repair_system, repair_user)
    if not ok:
        return False, {}, f"Repair LLM call failed: {err}"

    okj, obj, jerr = extract_first_json_object(raw)
    if not okj:
        return False, {}, f"Repair JSON parse failed: {jerr}"

    return True, obj, ""


def run_dependency_detection(
    llm: LLMClient,
    prompts: PromptPack,
    scenes_json: Dict[str, Any],
    max_retries: int = 2,
    raw_save_dir: str = None
) -> Tuple[bool, Dict[str, Any], str]:

    # NEW FIX
    scenes = scenes_json.get("scenes", [])
    if len(scenes) < 2:
        return True, {"dependencies": []}, ""

    dep_pair = prompts.load_pair("dep_system.txt", "dep_user.txt")
    dep_system = dep_pair["system"]
    dep_user = render_template(
        dep_pair["user"],
        {"SCENES_JSON": json.dumps(scenes_json, ensure_ascii=False, indent=2)}
    )

    last_error = ""

    for attempt in range(max_retries + 1):
        ok, raw, err = llm.chat(dep_system, dep_user)
        if not ok:
            last_error = f"Dependency LLM call failed: {err}"
            continue

        if raw_save_dir:
            from src.utils.io import write_text
            write_text(f"{raw_save_dir}/dep_raw_attempt_{attempt+1}.txt", raw)

        okj, obj, jerr = extract_first_json_object(raw)
        if not okj:
            okr, repaired_obj, repair_err = repair_json_with_llm(llm, prompts, raw)
            if not okr:
                last_error = f"Dependency JSON parse failed: {jerr}; Repair failed: {repair_err}"
                continue
            obj = repaired_obj

        if "dependencies" not in obj or not isinstance(obj["dependencies"], list):
            last_error = "Dependency output must contain { \"dependencies\": [ ... ] }"
            continue

        return True, obj, ""

    return False, {}, last_error