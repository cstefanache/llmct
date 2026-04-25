from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from .generation import StepRecord
from .models import ArchInfo
from .scenario import Scenario


@dataclass
class RunPaths:
    root: Path
    tensors_dir: Path
    run_json: Path
    steps_json: Path


def make_run_dir(scenario: Scenario, label: str = "") -> RunPaths:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_name = f"{scenario.name}_{label}_{ts}" if label else f"{scenario.name}_{ts}"
    root = Path(scenario.output.dir) / run_name
    tensors_dir = root / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        tensors_dir=tensors_dir,
        run_json=root / "run.json",
        steps_json=root / "steps.json",
    )


def env_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": str(device),
    }


def write_run_manifest(
    paths: RunPaths,
    scenario: Scenario,
    arch: ArchInfo,
    tokenizer,
    device: torch.device,
    prompt_token_ids: list[int],
) -> None:
    manifest = {
        "scenario": scenario.model_dump(),
        "model": {
            "id": scenario.model.id,
            "arch": asdict(arch),
        },
        "tokenizer": {
            "id": scenario.model.id,
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template": bool(getattr(tokenizer, "chat_template", None)),
        },
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": [tokenizer.decode([i]) for i in prompt_token_ids],
        "env": env_info(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths.run_json.write_text(json.dumps(manifest, indent=2, default=str))


def _npz_path_for(paths: RunPaths, step: int) -> Path:
    return paths.tensors_dir / f"step_{step:03d}.npz"


def write_step(paths: RunPaths, record: StepRecord) -> dict[str, Any]:
    """Persist a step: writes the .npz sidecar and returns the steps.json entry dict."""
    npz_path = _npz_path_for(paths, record.step)
    np.savez_compressed(npz_path, **record.tensors)

    tensor_index = {
        key: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        for key, arr in record.tensors.items()
    }
    rel_npz = npz_path.relative_to(paths.root).as_posix()

    return {
        "step": record.step,
        "kind": record.kind,
        "input_token_ids": record.input_token_ids,
        "input_tokens": record.input_tokens,
        "generated_token_id": record.generated_token_id,
        "generated_token": record.generated_token,
        "logit_argmax_id": record.logit_argmax_id,
        "logit_argmax_logprob": record.logit_argmax_logprob,
        "top_k": record.top_k,
        "seq_len": record.seq_len,
        "tensors_file": rel_npz,
        "tensor_index": tensor_index,
    }


def write_steps_json(paths: RunPaths, entries: list[dict[str, Any]]) -> None:
    paths.steps_json.write_text(json.dumps({"steps": entries}, indent=2, default=str))


def write_reference_state(run_dir: Path, label: str, tensors: dict[str, np.ndarray]) -> None:
    """Save a reference prefill's full captured tensors to references/ref_{label}.npz.

    ``tensors`` is the raw dict produced by ``capture_reference_prefill`` — same key
    convention as the per-step NPZ files (layer_NN/hidden_in, layer_NN/attn_weights, ...).
    """
    ref_dir = run_dir / "references"
    ref_dir.mkdir(exist_ok=True)
    np.savez_compressed(ref_dir / f"ref_{label}.npz", **tensors)


def write_reference_index(run_dir: Path, labels: list[str]) -> None:
    """Write references/index.json listing captured reference labels."""
    (run_dir / "references" / "index.json").write_text(
        json.dumps({"labels": labels}), encoding="utf-8"
    )


def write_conversation_snapshot(
    run_dir: Path, idx: int, role: str, tensors: dict[str, np.ndarray]
) -> None:
    """Save one conversation prefix snapshot to conversation_snapshots/snapshot_NN_role.npz.

    ``tensors`` is the raw dict produced by ``capture_reference_prefill`` — same key
    convention as the per-step NPZ files.
    """
    snap_dir = run_dir / "conversation_snapshots"
    snap_dir.mkdir(exist_ok=True)
    np.savez_compressed(snap_dir / f"snapshot_{idx:02d}_{role}.npz", **tensors)


def write_conversation_snapshot_index(
    run_dir: Path, snapshots: list[dict[str, Any]]
) -> None:
    """Write conversation_snapshots/index.json with metadata for each captured snapshot."""
    (run_dir / "conversation_snapshots" / "index.json").write_text(
        json.dumps({"snapshots": snapshots}), encoding="utf-8"
    )
