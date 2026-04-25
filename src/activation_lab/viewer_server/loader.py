"""NPZ and run-directory discovery + caching for the viewer server."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


class RunRegistry:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def list_runs(self) -> list[dict]:
        if not self.runs_dir.exists():
            return []
        out: list[dict] = []
        for p in sorted(self.runs_dir.iterdir(), key=lambda x: x.name):
            if not p.is_dir() or not (p / "run.json").exists():
                continue
            out.append({"id": p.name, "path": str(p)})
        return out

    def run_dir(self, run_id: str) -> Path:
        p = self.runs_dir / run_id
        if not p.is_dir() or not (p / "run.json").exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        return p

    def run_json(self, run_id: str) -> dict:
        return json.loads((self.run_dir(run_id) / "run.json").read_text())

    def snapshots_index(self, run_id: str) -> list[dict] | None:
        path = self.run_dir(run_id) / "conversation_snapshots" / "index.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("snapshots")

    def references_index(self, run_id: str) -> list[str] | None:
        path = self.run_dir(run_id) / "references" / "index.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("labels")

    def steps_index(self, run_id: str) -> list[dict] | None:
        path = self.run_dir(run_id) / "steps.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("steps")


@lru_cache(maxsize=8)
def _load_npz_cached(abs_path: str) -> dict[str, np.ndarray]:
    """Cache last N loaded NPZ files fully in memory. Small cache — tensors are heavy."""
    with np.load(abs_path) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"npz not found: {path}")
    return _load_npz_cached(str(path.resolve()))


def npz_inventory(path: Path) -> dict:
    """Lightweight metadata: keys and shapes (mmap, no full load)."""
    with np.load(path, mmap_mode="r") as npz:
        return {
            "file": path.name,
            "keys": list(npz.files),
            "shapes": {k: list(npz[k].shape) for k in npz.files},
            "dtypes": {k: str(npz[k].dtype) for k in npz.files},
        }


def resolve_npz(registry: RunRegistry, run_id: str, kind: str, name: str) -> Path:
    """Resolve a (kind, name) pair to an absolute NPZ path.

    kind ∈ {snapshot, reference, step}.
    name = file name without .npz extension (e.g. 'snapshot_00_system', 'ref_joy', 'step_000').
    """
    base = registry.run_dir(run_id)
    if kind == "snapshot":
        p = base / "conversation_snapshots" / f"{name}.npz"
    elif kind == "reference":
        p = base / "references" / f"{name}.npz"
    elif kind == "step":
        p = base / "tensors" / f"{name}.npz"
    else:
        raise ValueError(f"unknown npz kind: {kind}")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p
