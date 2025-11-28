# src/text/segmentation.py
import json
from typing import Dict, Any, Tuple, List

from src.llm.client import LLMClient
from src.llm.prompts import PromptPack, render_template
from src.llm.parsing import extract_first_json_object
from src.text.sentence_utils import get_sentence_spans


def repair_json_with_llm(
    llm: LLMClient,
    prompts: PromptPack,
    broken_output: str
) -> Tuple[bool, Dict[str, Any], str]:
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


def build_numbered_sentence_block(sentence_spans: List[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(sentence_spans, start=1):
        lines.append(f"{i}. {item['sentence']}")
    return "\n".join(lines)


def validate_grouping_output(obj: Dict[str, Any], num_sentences: int) -> List[str]:
    errors = []

    if "scene_groups" not in obj or not isinstance(obj["scene_groups"], list):
        return ["Missing 'scene_groups' list."]

    used = []

    for i, group in enumerate(obj["scene_groups"]):
        if not isinstance(group, dict):
            errors.append(f"scene_groups[{i}] must be an object.")
            continue

        if "scene_id" not in group or not isinstance(group["scene_id"], int):
            errors.append(f"scene_groups[{i}] missing int 'scene_id'.")

        if "sentence_ids" not in group or not isinstance(group["sentence_ids"], list):
            errors.append(f"scene_groups[{i}] missing list 'sentence_ids'.")
            continue

        sent_ids = group["sentence_ids"]
        if not sent_ids:
            errors.append(f"scene_groups[{i}] has empty sentence_ids.")
            continue

        for sid in sent_ids:
            if not isinstance(sid, int):
                errors.append(f"scene_groups[{i}] sentence id must be int.")
            elif sid < 1 or sid > num_sentences:
                errors.append(f"scene_groups[{i}] sentence id {sid} out of range.")

        # contiguous check
        sorted_ids = sorted(sent_ids)
        for j in range(1, len(sorted_ids)):
            if sorted_ids[j] != sorted_ids[j - 1] + 1:
                errors.append(
                    f"scene_groups[{i}] sentence_ids must be contiguous, got {sent_ids}."
                )
                break

        used.extend(sent_ids)

    used_sorted = sorted(used)
    expected = list(range(1, num_sentences + 1))

    if used_sorted != expected:
        errors.append(
            f"Sentence coverage mismatch. Expected {expected}, got {used_sorted}."
        )

    return errors


def build_scenes_from_groups(
    original_text: str,
    sentence_spans: List[Dict[str, Any]],
    grouping_obj: Dict[str, Any]
) -> Dict[str, Any]:
    scenes = []

    for idx, group in enumerate(grouping_obj["scene_groups"], start=1):
        sentence_ids = sorted(group["sentence_ids"])

        first_idx = sentence_ids[0] - 1
        last_idx = sentence_ids[-1] - 1

        start_char = sentence_spans[first_idx]["start_char_index"]
        end_char = sentence_spans[last_idx]["end_char_index"]

        if start_char == -1 or end_char == -1:
            scene_text = " ".join(sentence_spans[i - 1]["sentence"] for i in sentence_ids)
            start_char = -1
            end_char = -1
        else:
            scene_text = original_text[start_char:end_char].strip()

        scenes.append({
            "scene_id": idx,
            "scene_text": scene_text,
            "start_char_index": start_char,
            "end_char_index": end_char,
            "characters": group.get("characters", []),
            "location": group.get("location", ""),
            "time_hint": group.get("time_hint", ""),
            "core_action": group.get("core_action", ""),
            "camera_style": group.get("camera_style", ""),
            "new_scene_reason": group.get("new_scene_reason", "")
        })

    return {
        "scenes": scenes,
        "dependencies": [],
        "global_notes": {
            "num_scenes": len(scenes),
            "warnings": [],
            "sentence_count": len(sentence_spans)
        }
    }


def run_segmentation_with_verification(
    llm: LLMClient,
    prompts: PromptPack,
    long_prompt: str,
    max_fix_rounds: int = 2,
    raw_save_dir: str = None
) -> Tuple[bool, Dict[str, Any], str]:
    sentence_spans = get_sentence_spans(long_prompt)
    if not sentence_spans:
        return False, {}, "No sentences found in input prompt."

    numbered_sentences = build_numbered_sentence_block(sentence_spans)

    seg_pair = prompts.load_pair("seg_system.txt", "seg_user.txt")
    seg_system = seg_pair["system"]
    seg_user = render_template(
        seg_pair["user"],
        {
            "PROMPT": long_prompt,
            "NUMBERED_SENTENCES": numbered_sentences,
            "NUM_SENTENCES": str(len(sentence_spans))
        }
    )

    ok, raw, err = llm.chat(seg_system, seg_user)
    if not ok:
        return False, {}, f"Segmentation LLM call failed: {err}"

    if raw_save_dir:
        from src.utils.io import write_text
        write_text(f"{raw_save_dir}/seg_raw.txt", raw)

    okj, obj, jerr = extract_first_json_object(raw)
    if not okj:
        okr, repaired_obj, repair_err = repair_json_with_llm(llm, prompts, raw)
        if not okr:
            return False, {}, f"Segmentation JSON parse failed: {jerr}; Repair failed: {repair_err}"
        obj = repaired_obj

    for round_idx in range(max_fix_rounds):
        errors = validate_grouping_output(obj, len(sentence_spans))
        if not errors:
            break

        verify_pair = prompts.load_pair("verify_system.txt", "verify_user.txt")
        verify_system = verify_pair["system"]
        verify_user = render_template(
            verify_pair["user"],
            {
                "PROMPT": long_prompt,
                "NUMBERED_SENTENCES": numbered_sentences,
                "NUM_SENTENCES": str(len(sentence_spans)),
                "PROPOSED_JSON": json.dumps(obj, ensure_ascii=False, indent=2),
                "VALIDATION_ERRORS": "\n".join(errors)
            }
        )

        ok2, raw2, err2 = llm.chat(verify_system, verify_user)
        if not ok2:
            return False, obj, f"Verification LLM call failed: {err2}"

        if raw_save_dir:
            from src.utils.io import write_text
            write_text(f"{raw_save_dir}/verify_raw_round_{round_idx + 1}.txt", raw2)

        okj2, obj2, jerr2 = extract_first_json_object(raw2)
        if not okj2:
            okr2, repaired_obj2, repair_err2 = repair_json_with_llm(llm, prompts, raw2)
            if not okr2:
                return False, obj, f"Verification JSON parse failed: {jerr2}; Repair failed: {repair_err2}"
            obj2 = repaired_obj2

        obj = obj2

    final_errors = validate_grouping_output(obj, len(sentence_spans))
    if final_errors:
        return False, obj, "Final grouping issues: " + "; ".join(final_errors)

    final_obj = build_scenes_from_groups(long_prompt, sentence_spans, obj)
    return True, final_obj, ""