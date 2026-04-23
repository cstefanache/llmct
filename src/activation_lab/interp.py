"""Post-hoc analyses over captured runs.

All functions operate on NumPy arrays loaded from the per-step NPZ files. Logit-lens-style
operations that require the model's final norm + lm_head reload the model on demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import get_module


# --------------------------------------------------------------------------- residual stream

@dataclass
class ResidualNorms:
    layer: int
    step: int
    position: int  # which token position (-1 for "last")
    norm_hidden_in: float
    norm_attn_out: float
    norm_mlp_down_out: float
    norm_hidden_out: float
    # How much of hidden_out is explained by attn + mlp contributions (cosine)
    attn_contribution_cos: float
    mlp_contribution_cos: float


def _last_position(arr: np.ndarray) -> np.ndarray:
    """Pick the last token position from a (B, T, D) tensor."""
    if arr.ndim == 3:
        return arr[0, -1, :]
    if arr.ndim == 2:
        return arr[-1, :]
    raise ValueError(f"unexpected shape {arr.shape}")


def residual_norms_for_step(npz: np.lib.npyio.NpzFile, num_layers: int, step: int) -> list[ResidualNorms]:
    """For each layer in this step, compute residual-stream magnitudes at the final position."""
    out: list[ResidualNorms] = []
    for li in range(num_layers):
        k = lambda name: f"layer_{li:02d}/{name}"  # noqa: E731
        needed = ("hidden_in", "attn_out", "mlp_down_out", "hidden_out")
        if not all(k(n) in npz.files for n in needed):
            continue
        h_in = _last_position(npz[k("hidden_in")]).astype(np.float32)
        a = _last_position(npz[k("attn_out")]).astype(np.float32)
        m = _last_position(npz[k("mlp_down_out")]).astype(np.float32)
        h_out = _last_position(npz[k("hidden_out")]).astype(np.float32)

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

        out.append(
            ResidualNorms(
                layer=li,
                step=step,
                position=-1,
                norm_hidden_in=float(np.linalg.norm(h_in)),
                norm_attn_out=float(np.linalg.norm(a)),
                norm_mlp_down_out=float(np.linalg.norm(m)),
                norm_hidden_out=float(np.linalg.norm(h_out)),
                attn_contribution_cos=_cos(a, h_out),
                mlp_contribution_cos=_cos(m, h_out),
            )
        )
    return out


# --------------------------------------------------------------------------- neuron trajectory

def neuron_trajectory(
    run_dir: Path,
    source: str,
    layer: int,
    neuron: int,
    position: int = -1,
) -> np.ndarray:
    """Return a 1D array of this neuron's activation at `position` across all steps.

    `source` is a per-layer key like 'hidden_out', 'mlp_act'. `position` is absolute within the
    tensor's seq axis (-1 for last, works for both prefill and decode).
    """
    import json
    steps = json.loads((run_dir / "steps.json").read_text())["steps"]
    key = f"layer_{layer:02d}/{source}"
    vals: list[float] = []
    for step in steps:
        npz_path = run_dir / step["tensors_file"]
        if not npz_path.exists():
            continue
        with np.load(npz_path) as npz:
            if key not in npz.files:
                vals.append(float("nan"))
                continue
            t = npz[key]
            if t.ndim == 3:
                vals.append(float(t[0, position, neuron]))
            elif t.ndim == 2:
                vals.append(float(t[position, neuron]))
            else:
                vals.append(float("nan"))
    return np.array(vals)


# --------------------------------------------------------------------------- cross-layer

def cross_layer_cosine(npz: np.lib.npyio.NpzFile, num_layers: int, source: str = "hidden_out") -> np.ndarray:
    """Cosine-similarity matrix over layers, using the last-position vector as the layer signature."""
    vecs: list[np.ndarray] = []
    present: list[int] = []
    for li in range(num_layers):
        k = f"layer_{li:02d}/{source}"
        if k in npz.files:
            vecs.append(_last_position(npz[k]).astype(np.float32))
            present.append(li)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    M = np.stack(vecs, axis=0)  # (L, H)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Mn = M / norms
    return Mn @ Mn.T  # (L, L)


# --------------------------------------------------------------------------- logit lens

@dataclass
class LogitLensFrame:
    step: int
    layer: int
    position: int
    top_k: list[dict[str, Any]]          # [{id, token, logprob}]
    argmax_id: int
    argmax_logprob: float
    kl_from_final: float | None          # KL(this_layer || final_layer) — only if both present


def _discover_final_norm(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("model.norm", "model.final_layer_norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        try:
            return get_module(model, path)
        except AttributeError:
            continue
    raise RuntimeError("could not locate final norm module on this model")


def _discover_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("lm_head", "embed_out"):
        try:
            return get_module(model, path)
        except AttributeError:
            continue
    raise RuntimeError("could not locate lm_head module on this model")


@torch.no_grad()
def logit_lens_batch(
    hiddens: dict[int, np.ndarray],  # layer_idx -> (T, H) or (B, T, H)
    model: torch.nn.Module,
    position: int = -1,
    top_k: int = 10,
    tokenizer=None,
) -> list[LogitLensFrame]:
    """Project each layer's hidden state through final_norm + lm_head at one position.

    Returns one frame per layer with top-k next-token predictions and KL from the last layer.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    final_norm = _discover_final_norm(model)
    lm_head = _discover_lm_head(model)

    frames: list[LogitLensFrame] = []
    all_layer_logits: dict[int, torch.Tensor] = {}

    for li, h in sorted(hiddens.items()):
        arr = h
        if arr.ndim == 3:
            arr = arr[0]
        v = arr[position]  # (H,)
        x = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device=device, dtype=dtype)
        x = x.view(1, 1, -1)  # (1, 1, H)
        h_norm = final_norm(x)
        logits = lm_head(h_norm)  # (1, 1, V)
        logits_f = logits.float().view(-1)  # (V,)
        all_layer_logits[li] = logits_f

    # KL from the deepest available layer's distribution.
    if all_layer_logits:
        final_logprobs = torch.log_softmax(all_layer_logits[max(all_layer_logits)], dim=-1)
    else:
        final_logprobs = None

    for li, lg in all_layer_logits.items():
        lp = torch.log_softmax(lg, dim=-1)
        argmax_id = int(lp.argmax().item())
        argmax_lp = float(lp[argmax_id].item())
        vals, ids = torch.topk(lp, k=min(top_k, lp.shape[-1]))
        payload = [
            {
                "id": int(i),
                "token": tokenizer.decode([int(i)]) if tokenizer is not None else None,
                "logprob": float(v),
            }
            for v, i in zip(vals.tolist(), ids.tolist())
        ]
        kl: float | None = None
        if final_logprobs is not None and li != max(all_layer_logits):
            p = torch.exp(lp)
            kl = float((p * (lp - final_logprobs)).sum().item())
        frames.append(
            LogitLensFrame(
                step=-1,  # filled by caller
                layer=li,
                position=position,
                top_k=payload,
                argmax_id=argmax_id,
                argmax_logprob=argmax_lp,
                kl_from_final=kl,
            )
        )
    return frames


def logit_lens_for_step(
    npz: np.lib.npyio.NpzFile,
    num_layers: int,
    model: torch.nn.Module,
    tokenizer,
    step: int,
    position: int = -1,
    top_k: int = 10,
) -> list[LogitLensFrame]:
    hiddens: dict[int, np.ndarray] = {}
    for li in range(num_layers):
        k = f"layer_{li:02d}/hidden_out"
        if k in npz.files:
            hiddens[li] = npz[k]
    frames = logit_lens_batch(hiddens, model, position=position, top_k=top_k, tokenizer=tokenizer)
    for fr in frames:
        fr.step = step
    return frames
