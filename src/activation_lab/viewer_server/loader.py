"""NPZ and run-directory discovery + caching for the viewer server."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


RUN_ID_SEP = "__"


class RunRegistry:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir

    def list_runs(self) -> list[dict]:
        """Enumerate runs across both layouts.

        - Nested (current): ``runs_dir/<scenario>_<ts>/<promptLabel>/run.json``.
          Each child is emitted with ``id = "<scenario>__<promptLabel>"``,
          ``group = "<scenario>_<ts>"``, ``label = "<promptLabel>"``.
        - Legacy flat: ``runs_dir/<run_name>/run.json``. Emitted with
          ``id = "<run_name>"``, ``group = None``.
        """
        if not self.runs_dir.exists():
            return []
        out: list[dict] = []

        def _ctime(p: Path) -> float:
            st = p.stat()
            return getattr(st, "st_birthtime", st.st_ctime)

        for top in sorted(self.runs_dir.iterdir(), key=_ctime):
            if not top.is_dir():
                continue
            children = [
                c for c in sorted(top.iterdir(), key=lambda x: x.name)
                if c.is_dir() and (c / "run.json").exists()
            ]
            if children:
                for c in children:
                    out.append({
                        "id": f"{top.name}{RUN_ID_SEP}{c.name}",
                        "path": str(c),
                        "group": top.name,
                        "label": c.name,
                    })
            elif (top / "run.json").exists():
                out.append({
                    "id": top.name,
                    "path": str(top),
                    "group": None,
                    "label": top.name,
                })
        return out

    def run_dir(self, run_id: str) -> Path:
        if RUN_ID_SEP in run_id:
            group, label = run_id.split(RUN_ID_SEP, 1)
            p = self.runs_dir / group / label
        else:
            p = self.runs_dir / run_id
        if not p.is_dir() or not (p / "run.json").exists():
            raise FileNotFoundError(f"run not found: {run_id}")
        return p

    def scenario_dir(self, run_id: str) -> Path | None:
        """Parent scenario dir if this run is nested, else None."""
        if RUN_ID_SEP not in run_id:
            return None
        group = run_id.split(RUN_ID_SEP, 1)[0]
        p = self.runs_dir / group
        return p if p.is_dir() else None

    def run_json(self, run_id: str) -> dict:
        return json.loads((self.run_dir(run_id) / "run.json").read_text())

    def snapshots_index(self, run_id: str) -> list[dict] | None:
        path = self.run_dir(run_id) / "conversation_snapshots" / "index.json"
        if not path.exists():
            return None
        return json.loads(path.read_text()).get("snapshots")

    def references_dir(self, run_id: str) -> Path:
        """References live at the scenario level for nested runs, run level for legacy."""
        sdir = self.scenario_dir(run_id)
        base = sdir if sdir is not None else self.run_dir(run_id)
        return base / "references"

    def references_index(self, run_id: str) -> list[str] | None:
        path = self.references_dir(run_id) / "index.json"
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
    if kind == "snapshot":
        p = registry.run_dir(run_id) / "conversation_snapshots" / f"{name}.npz"
    elif kind == "reference":
        p = registry.references_dir(run_id) / f"{name}.npz"
    elif kind == "step":
        p = registry.run_dir(run_id) / "tensors" / f"{name}.npz"
    else:
        raise ValueError(f"unknown npz kind: {kind}")
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p
