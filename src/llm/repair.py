# src/llm/repair.py
import json
from typing import Dict, Any, Tuple
from src.llm.client import LLMClient
from src.llm.prompts import PromptPack, render_template
from src.llm.parsing import extract_first_json_object


def repair_json_with_llm(
    llm: LLMClient,
    prompts: PromptPack,
    broken_text: str,
    schema_type: str = "generic"
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Ask LLM to repair malformed JSON output.
    schema_type: generic / segmentation / dependency
    """

    repair_pair = prompts.load_pair("repair_system.txt", "repair_user.txt")

    system = repair_pair["system"]
    user = render_template(
        repair_pair["user"],
        {
            "SCHEMA_TYPE": schema_type,
            "BROKEN_TEXT": broken_text
        }
    )

    ok, raw, err = llm.chat(system, user)
    if not ok:
        return False, {}, f"Repair call failed: {err}"

    okj, obj, jerr = extract_first_json_object(raw)
    if not okj:
        return False, {}, f"Repair parse failed: {jerr}"

    return True, obj, ""