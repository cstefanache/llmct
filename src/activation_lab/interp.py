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


# --------------------------------------------------------------------------- steering

@torch.no_grad()
def project_residual_to_logits(
    vec: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Project a single residual vector (H,) through final_norm + lm_head to full-vocab logits (V,).

    This is the *exact* projection the logit lens uses (see :func:`logit_lens_batch`), so
    ``softmax(project_residual_to_logits(h))[X]`` is the same "prob(X) at this layer" the viewer
    displays. Steering solves against this objective so the number the user sets is the number the
    lens reads. The live ``nn.Module`` is always called (never a hand-rolled norm) so RMSNorm
    quirks — Gemma's ``(1+w)`` gain, eps, dtype — are handled correctly.
    """
    x = vec.to(device=device, dtype=dtype).view(1, 1, -1)
    logits = lm_head(final_norm(x))
    return logits.float().view(-1)


@dataclass
class SteerSolution:
    alpha: float          # scalar applied to the (unit-less) steering direction
    p0: float             # prob(X) at this layer/position before steering (alpha = 0)
    target_prob: float    # the clamped target we aimed for
    achieved_prob: float  # prob(X) actually reached at the returned alpha
    saturated: bool       # True if the target was unreachable and alpha was capped


@torch.no_grad()
def solve_steer_alpha(
    h: torch.Tensor,               # (H,) residual at the target position/layer
    direction: torch.Tensor,       # (H,) steering direction, e.g. lm_head.weight[X]
    token_id: int,                 # X — the token whose prob we drive
    target_prob: float,            # desired prob(X) over the full vocab
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    tol: float = 1e-4,
    max_bracket: int = 40,
    max_bisect: int = 60,
    alpha_cap_mult: float = 1e4,
) -> SteerSolution:
    """Find ``alpha`` so that ``softmax(project(h + alpha*direction))[X] == target_prob``.

    The objective is empirically monotone in ``alpha`` (increasing for a boost, decreasing for a
    suppression) but RMSNorm bounds the residual norm, so the reachable prob **saturates** below 1
    and above 0 — ``target = 1.0`` is generally unreachable. When the target lies outside the
    reachable range we cap ``alpha`` and return ``saturated=True`` (best effort) rather than looping.
    """
    h = h.to(device=device, dtype=torch.float32)
    d = direction.to(device=device, dtype=torch.float32)
    eps = 1e-6

    def prob_at(alpha: float) -> float:
        logits = project_residual_to_logits(h + alpha * d, final_norm, lm_head, device, dtype)
        lp = torch.log_softmax(logits, dim=-1)
        return float(lp[token_id].exp().item())

    p0 = prob_at(0.0)
    tgt = min(max(float(target_prob), eps), 1.0 - eps)
    if abs(tgt - p0) <= tol:
        return SteerSolution(0.0, p0, tgt, p0, False)

    sign = 1.0 if tgt > p0 else -1.0
    hn = float(torch.linalg.norm(h).item())
    dn = float(torch.linalg.norm(d).item())
    s0 = (hn / dn) if dn > eps else 1.0
    alpha_cap = alpha_cap_mult * s0

    # exponential bracketing from a scale-matched seed
    lo_a, lo_p = 0.0, p0
    hi_a = sign * s0
    hi_p = prob_at(hi_a)
    crossed = False
    stagnant = 0
    for _ in range(max_bracket):
        if (sign > 0 and hi_p >= tgt) or (sign < 0 and hi_p <= tgt):
            crossed = True
            break
        if abs(hi_p - lo_p) < 1e-7:          # prob stopped moving → saturation
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        lo_a, lo_p = hi_a, hi_p
        hi_a *= 2.0
        if abs(hi_a) >= alpha_cap:
            hi_a = sign * alpha_cap
            hi_p = prob_at(hi_a)
            crossed = (sign > 0 and hi_p >= tgt) or (sign < 0 and hi_p <= tgt)
            break
        hi_p = prob_at(hi_a)

    if not crossed:
        return SteerSolution(hi_a, p0, tgt, hi_p, True)

    # bisection within [lo_a, hi_a]
    a_lo, a_hi = lo_a, hi_a
    a_best, p_best = hi_a, hi_p
    for _ in range(max_bisect):
        mid = 0.5 * (a_lo + a_hi)
        pm = prob_at(mid)
        a_best, p_best = mid, pm
        if abs(pm - tgt) <= tol:
            break
        if (sign > 0 and pm < tgt) or (sign < 0 and pm > tgt):
            a_lo = mid
        else:
            a_hi = mid
    return SteerSolution(a_best, p0, tgt, p_best, False)


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
