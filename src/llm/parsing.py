# src/llm/parsing.py
import json
import re
from typing import Any, Dict, Tuple, List


def extract_first_json_object(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Extract first JSON object from raw LLM output.
    Handles:
    - plain JSON
    - markdown code blocks
    - extra text before/after JSON
    """
    t = text.strip()

    # remove markdown fences
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return True, obj, ""
    except Exception:
        pass

    # balanced braces extraction
    start = t.find("{")
    if start == -1:
        return False, {}, "Could not find opening '{' for JSON object."

    brace_count = 0
    end = -1
    for i in range(start, len(t)):
        if t[i] == "{":
            brace_count += 1
        elif t[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end == -1:
        return False, {}, "Could not find complete balanced JSON object."

    candidate = t[start:end + 1]

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return True, obj, ""
        return False, {}, "Extracted JSON is not an object."
    except Exception as e:
        return False, {}, f"JSON parse error: {e}"


def validate_schema_minimal(obj: Dict[str, Any]) -> List[str]:
    errors = []

    if "scenes" in obj:
        if not isinstance(obj["scenes"], list):
            errors.append("Invalid 'scenes' (must be list).")
        else:
            for i, s in enumerate(obj["scenes"]):
                if not isinstance(s, dict):
                    errors.append(f"Scene[{i}] must be object.")
                    continue
                if "scene_id" not in s or not isinstance(s["scene_id"], int):
                    errors.append(f"Scene[{i}] must contain int scene_id.")
                if "scene_text" not in s or not isinstance(s["scene_text"], str) or not s["scene_text"].strip():
                    errors.append(f"Scene[{i}] must contain non-empty scene_text.")

    if "dependencies" in obj:
        if not isinstance(obj["dependencies"], list):
            errors.append("Invalid 'dependencies' (must be list).")
        else:
            for i, d in enumerate(obj["dependencies"]):
                if not isinstance(d, dict):
                    errors.append(f"Dependency[{i}] must be object.")
                    continue
                required = ["from_scene_id", "to_scene_id", "dependent"]
                for k in required:
                    if k not in d:
                        errors.append(f"Dependency[{i}] missing '{k}'.")

    if "global_notes" in obj and not isinstance(obj["global_notes"], dict):
        errors.append("Invalid 'global_notes' (must be dict).")

    return errors


def dumps_pretty(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)