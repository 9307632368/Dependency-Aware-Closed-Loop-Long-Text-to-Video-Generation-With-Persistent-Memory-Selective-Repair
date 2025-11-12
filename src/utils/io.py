# src/utils/io.py
import os
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def exists(path: str) -> bool:
    return os.path.exists(path)


def list_files(path: str, suffix: str = None):
    if not os.path.exists(path):
        return []

    p = Path(path)
    if suffix:
        return sorted([str(x) for x in p.glob(f"*{suffix}") if x.is_file()])
    return sorted([str(x) for x in p.iterdir() if x.is_file()])