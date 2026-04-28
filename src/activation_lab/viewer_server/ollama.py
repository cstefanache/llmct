"""Discover locally cached HuggingFace models (AutoModelForCausalLM-compatible)."""
from __future__ import annotations

import os
from pathlib import Path


def _hf_hub_cache() -> Path:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or os.path.join(hf_home, "hub")
    )


def _dir_to_model_id(name: str) -> str | None:
    """'models--Qwen--Qwen2.5-3B-Instruct' → 'Qwen/Qwen2.5-3B-Instruct'."""
    if not name.startswith("models--"):
        return None
    rest = name[len("models--"):]
    parts = rest.split("--", 1)
    return "/".join(parts)


def list_local_models() -> list[dict]:
    """Scan the HF hub cache and return all locally downloaded model ids.

    A model is included only if it has a completed snapshot (config.json present).
    Returns [] if the cache dir does not exist — never raises.
    """
    cache = _hf_hub_cache()
    if not cache.exists():
        return []

    result = []
    for entry in sorted(cache.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        model_id = _dir_to_model_id(entry.name)
        if not model_id:
            continue
        snapshots = entry / "snapshots"
        if not snapshots.is_dir():
            continue
        has_config = any(
            (snap / "config.json").exists()
            for snap in snapshots.iterdir()
            if snap.is_dir()
        )
        if has_config:
            result.append({"model_id": model_id})
    return result
