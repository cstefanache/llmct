"""On-demand logit lens for the viewer server.

Reloads the model's ``final_norm`` + ``lm_head`` and projects each captured
intermediate-layer ``hidden_out`` through the unembedding to obtain per-layer
next-token probabilities (the classic *logit lens*). The projection math lives
in :func:`activation_lab.interp.logit_lens_batch`; this module only handles
model loading/caching and shaping the result into a JSON-friendly dict.

The model is loaded once per model-id and cached for the server's lifetime —
the first call for a given model pays the full HF load cost, subsequent calls
are fast.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from ..interp import logit_lens_batch
from ..models import load_model
from ..scenario import ModelConfig
from .loader import RunRegistry, load_npz, resolve_npz

# model_id -> (model, tokenizer)
_MODELS: dict[str, tuple[Any, Any]] = {}
_LOCK = threading.Lock()


def _get_model(run_json: dict) -> tuple[Any, Any]:
    """Return a cached ``(model, tokenizer)`` for this run's model, loading once."""
    model_id = run_json["model"]["id"]
    with _LOCK:
        cached = _MODELS.get(model_id)
        if cached is not None:
            return cached
        cfg = ModelConfig(**run_json["scenario"]["model"])
        model, tokenizer, _arch, _device = load_model(cfg)
        _MODELS[model_id] = (model, tokenizer)
        return model, tokenizer


def compute_logit_lens(
    registry: RunRegistry,
    run_id: str,
    kind: str,
    name: str,
    position: int = -1,
    top_k: int = 10,
) -> dict:
    """Per-layer top-k next-token predictions for one snapshot at one position.

    Raises ``ValueError`` if the snapshot has no per-layer ``hidden_out`` tensors
    (the caller turns this into an HTTP 400).
    """
    run_json = registry.run_json(run_id)
    num_layers = int(run_json["model"]["arch"]["num_layers"])

    tensors = load_npz(resolve_npz(registry, run_id, kind, name))
    hiddens: dict[int, np.ndarray] = {}
    seq_len = 0
    for li in range(num_layers):
        key = f"layer_{li:02d}/hidden_out"
        if key in tensors:
            arr = tensors[key]
            hiddens[li] = arr
            seq_len = int(arr.shape[-2]) if arr.ndim >= 2 else int(arr.shape[0])
    if not hiddens:
        raise ValueError("snapshot has no per-layer hidden_out tensors")

    model, tokenizer = _get_model(run_json)
    frames = logit_lens_batch(
        hiddens, model, position=position, top_k=top_k, tokenizer=tokenizer
    )

    layers_out: list[dict] = []
    for fr in frames:
        predicted = [
            {
                "id": t["id"],
                "token": t["token"],
                "logprob": t["logprob"],
                "prob": math.exp(t["logprob"]),
            }
            for t in fr.top_k
        ]
        layers_out.append(
            {
                "layer": fr.layer,
                "argmax_id": fr.argmax_id,
                "argmax_token": (predicted[0]["token"] if predicted else None),
                "argmax_prob": math.exp(fr.argmax_logprob),
                "kl_from_final": fr.kl_from_final,
                "predicted": predicted,
            }
        )

    return {
        "run_id": run_id,
        "kind": kind,
        "name": name,
        "model_id": run_json["model"]["id"],
        "num_layers": num_layers,
        "seq_len": seq_len,
        "position": position,
        "layers": layers_out,
    }
