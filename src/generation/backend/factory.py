# src/generation/backend/factory.py
from __future__ import annotations

from typing import Dict, Any

from src.generation.backend.svd_backend import SVDBackend
from src.generation.backend.cogvideox_backend import CogVideoXBackend
from src.generation.backend.hybrid_backend import HybridBackend


def build_backend(config: Dict[str, Any]):
    config = dict(config or {})
    backend_name = str(config.get("backend", "hybrid") or "hybrid").strip().lower()

    if backend_name == "svd":
        return SVDBackend(_merge_backend_specific_config(config, "svd"))

    if backend_name == "cogvideox":
        return CogVideoXBackend(_merge_backend_specific_config(config, "cogvideox"))

    if backend_name == "hybrid":
        return HybridBackend(config)

    raise ValueError(f"Unsupported backend: {backend_name}")


def _merge_backend_specific_config(config: Dict[str, Any], backend_name: str) -> Dict[str, Any]:
    """
    For direct single-backend mode:
    merge top-level config with config['backends'][backend_name].
    """
    config = dict(config or {})
    merged = dict(config)

    backends_cfg = config.get("backends", {}) or {}
    specific = backends_cfg.get(backend_name, {}) or {}

    merged.update(specific)
    merged["backend"] = backend_name
    return merged