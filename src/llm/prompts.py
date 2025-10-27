# src/llm/prompts.py
from pathlib import Path
from typing import Dict


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_template(template: str, variables: Dict[str, str]) -> str:
    """
    Very simple template renderer: replaces {VAR} with variables["VAR"].
    Avoids external templating libs.
    """
    out = template
    for k, v in variables.items():
        out = out.replace("{" + k + "}", v)
    return out


class PromptPack:
    """
    Loads system/user prompt templates from configs/prompts/
    """
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)

    def load_pair(self, system_file: str, user_file: str) -> Dict[str, str]:
        system = load_text(str(self.prompt_dir / system_file))
        user = load_text(str(self.prompt_dir / user_file))
        return {"system": system, "user": user}