"""Interactive activation *steering* for the viewer server.

A steering session keeps a model resident in the server process (sharing the logit-lens model
cache) and lets the UI:

1. run a base prefill+generation over a prompt,
2. read the per-layer logit lens at any token position,
3. apply a single **steer edit** — add ``alpha · W_U[X]`` to a layer's ``hidden_out`` at the
   clicked position so token *X* reaches a target probability — and re-run generation,
4. save the base + steered runs as ordinary run directories so they drop into the compare views.

The intervention edits the residual stream (``hidden_out``, pre-final-norm) and lets the rest of the
network compute forward from the edited state. The scalar ``alpha`` is solved against the *exact*
logit-lens projection (:func:`activation_lab.interp.solve_steer_alpha`) so the probability the user
sets is the probability the lens reads. See that module for the mechanism's limitations (RMSNorm
saturation makes ``p=1`` generally unreachable; all-layers steering compounds and can push the
residual off-manifold).
"""
from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, model_validator

from ..capture import CaptureContext
from ..generation import (
    StepRecord,
    _build_input_ids,
    _pick_next_token,
    _top_k_payload,
    capture_reference_prefill,
)
from ..interp import (
    _discover_final_norm,
    _discover_lm_head,
    logit_lens_batch,
    project_residual_to_logits,
    solve_steer_alpha,
)
from ..models import get_module
from ..scenario import (
    CaptureConfig,
    GenerationConfig,
    Message,
    ModelConfig,
    OutputConfig,
    PromptConfig,
    Scenario,
)
from ..serialize import (
    make_run_dir,
    make_scenario_dir,
    write_conversation_snapshot,
    write_conversation_snapshot_index,
    write_run_manifest,
    write_scenario_manifest,
    write_step,
    write_steps_json,
)
from .lens import get_model_by_cfg

# All forward passes that (de)register steering hooks run under this lock. FastAPI dispatches sync
# handlers in a threadpool, and the hooks mutate shared state on a single resident model, so steering
# must be single-flight.
INFERENCE_LOCK = threading.Lock()

# Light capture config used for lens forwards — we only need hidden_out.
_HIDDEN_ONLY = CaptureConfig(
    hidden_states=True, attention_weights=False, qkv=False, mlp=False, logits=False, top_k_probs=0
)


# --------------------------------------------------------------------------- request models

class SteerEdit(BaseModel):
    token_id: int                              # X — the token to boost/suppress/swap-to
    position: int                              # ABSOLUTE index into full = prompt + generated
    scope: Literal["layer", "all"]
    layers: list[int]                          # concrete target layers (single-element for "layer")
    mode: Literal["relative", "absolute"]
    value: float                               # t∈[-1,1] (relative) | p∈[0,1] (absolute) | w∈[0,1] (swap)
    # Intervention family:
    #   "point"     — edit the residual at THIS position only (default; local next-token nudge).
    #   "broadcast" — add the same residual direction at EVERY position + every generated token
    #                 (classic activation steering; strong, global push toward the target token).
    #   "swap"      — interpolate the INPUT EMBEDDING at this position toward the target token by
    #                 weight `value` (a soft token swap; the model genuinely "reads" a new token).
    kind: Literal["point", "broadcast", "swap"] = "point"

    @model_validator(mode="after")
    def _check(self) -> SteerEdit:
        if self.kind == "swap":
            if not (0.0 <= self.value <= 1.0):
                raise ValueError("swap weight must be in [0, 1]")
            return self
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if self.mode == "relative" and not (-1.0 <= self.value <= 1.0):
            raise ValueError("relative value must be in [-1, 1]")
        if self.mode == "absolute" and not (0.0 <= self.value <= 1.0):
            raise ValueError("absolute value must be in [0, 1]")
        return self


# --------------------------------------------------------------------------- session

@dataclass
class SteeringSession:
    session_id: str
    model_cfg: ModelConfig
    messages: list[Message]
    gen_cfg: GenerationConfig
    cap_cfg: CaptureConfig
    model: Any
    tokenizer: Any
    arch: Any
    device: Any
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    # base state (filled by run_base)
    prompt_ids: list[int] = field(default_factory=list)
    base_generated_ids: list[int] = field(default_factory=list)
    full_ids: list[int] = field(default_factory=list)
    base_hiddens: dict[int, Any] = field(default_factory=dict)  # layer -> (1, T, H) np.ndarray
    # last steered result (for save)
    last_edit: SteerEdit | None = None
    last_steered_ids: list[int] = field(default_factory=list)
    last_solved: list[dict] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)


_SESSIONS: dict[str, SteeringSession] = {}


def create_session(
    model: dict,
    messages: list[dict],
    generation: dict | None = None,
    capture: dict | None = None,
) -> str:
    model_cfg = ModelConfig(**model)
    msgs = [Message(**m) for m in messages]
    gen_cfg = GenerationConfig(**(generation or {}))
    # Rich capture by default so saved snapshots carry attention etc.
    cap_cfg = CaptureConfig(**capture) if capture else CaptureConfig()
    m, tok, arch, device = get_model_by_cfg(model_cfg)
    final_norm = _discover_final_norm(m)
    lm_head = _discover_lm_head(m)
    sid = uuid.uuid4().hex
    _SESSIONS[sid] = SteeringSession(
        session_id=sid, model_cfg=model_cfg, messages=msgs, gen_cfg=gen_cfg, cap_cfg=cap_cfg,
        model=m, tokenizer=tok, arch=arch, device=device, final_norm=final_norm, lm_head=lm_head,
    )
    return sid


def get_session(session_id: str) -> SteeringSession:
    s = _SESSIONS.get(session_id)
    if s is None:
        raise KeyError(session_id)
    return s


# --------------------------------------------------------------------------- steering hooks

class _SteerState:
    """Mutable state shared with the steering forward-hooks for one forward/generation run."""

    def __init__(self, target_pos: int, token_id: int, direction: torch.Tensor,
                 target_prob_by_layer: dict[int, float], session: SteeringSession,
                 *, broadcast: bool = False, fixed_alpha: dict[int, float] | None = None):
        self.target_pos = target_pos
        self.token_id = token_id
        self.direction = direction                 # (H,) float32
        self.target_prob_by_layer = target_prob_by_layer
        self.layers = set(target_prob_by_layer)
        self.session = session
        self.cur_offset = 0                        # abs index of local position 0 in the current pass
        self.solved: dict[int, dict] = {}
        # broadcast: add `fixed_alpha[L]·direction` to EVERY position at layer L (prefill + each
        # generated token), instead of solving at a single target position.
        self.broadcast = broadcast
        self.fixed_alpha = fixed_alpha or {}


def _make_steer_hook(state: _SteerState, layer_idx: int):
    s = state.session

    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if not isinstance(t, torch.Tensor):
            return output

        if state.broadcast:
            a = state.fixed_alpha.get(layer_idx)
            if a is None:
                return output
            t = t.clone()
            t[0, :, :] = (t[0, :, :].to(torch.float32) + a * state.direction).to(t.dtype)
            return (t, *output[1:]) if isinstance(output, tuple) else t

        seq = t.shape[1]
        local = state.target_pos - state.cur_offset
        if not (0 <= local < seq):
            return output
        h = t[0, local, :]
        sol = solve_steer_alpha(
            h, state.direction, state.token_id, state.target_prob_by_layer[layer_idx],
            s.final_norm, s.lm_head, s.device, next(s.model.parameters()).dtype,
        )
        state.solved[layer_idx] = {
            "layer": layer_idx, "p0": sol.p0, "target_prob": sol.target_prob,
            "alpha": sol.alpha, "achieved_prob": sol.achieved_prob, "saturated": sol.saturated,
        }
        edited = (h.to(torch.float32) + sol.alpha * state.direction).to(t.dtype)
        t = t.clone()
        t[0, local, :] = edited
        if isinstance(output, tuple):
            return (t, *output[1:])
        return t

    return hook


def _register_hooks(session: SteeringSession, state: _SteerState) -> list:
    layers = get_module(session.model, session.arch.layer_module_path)
    return [layers[li].register_forward_hook(_make_steer_hook(state, li)) for li in sorted(state.layers)]


# --------------------------------------------------------------------------- forward helpers

def _prompt_cfg(session: SteeringSession) -> PromptConfig:
    return PromptConfig(messages=session.messages, run_at_each_message=False)


# A soft token swap: (absolute prefill position, delta vector to add to that position's input embedding).
EmbedOverride = tuple[int, torch.Tensor]


def _forward_kwargs(session: SteeringSession, cur_input: torch.Tensor, is_prefill: bool,
                    embed_override: EmbedOverride | None) -> dict:
    """Return the model() input kwargs. For a swap on the prefill pass, feed interpolated
    `inputs_embeds` instead of `input_ids` so the model reads a blended token at one position."""
    if embed_override is not None and is_prefill:
        pos, delta = embed_override
        embed = get_module(session.model, session.arch.embed_module_path)
        ie = embed(cur_input).clone()
        if 0 <= pos < ie.shape[1]:
            ie[0, pos, :] = ie[0, pos, :] + delta.to(dtype=ie.dtype, device=ie.device)
        return {"inputs_embeds": ie}
    return {"input_ids": cur_input}


def _capture_hiddens(session: SteeringSession, ids: list[int], state: _SteerState | None = None,
                     embed_override: EmbedOverride | None = None) -> dict[int, Any]:
    """One use_cache=False forward over `ids`; return {layer: (1, T, H)} hidden_out (optionally steered)."""
    input_ids = torch.tensor([ids], device=session.device, dtype=torch.long)
    handles = _register_hooks(session, state) if state is not None else []
    try:
        if state is not None:
            state.cur_offset = 0
        kwargs = _forward_kwargs(session, input_ids, True, embed_override)
        with torch.no_grad(), CaptureContext(session.model, session.arch, _HIDDEN_ONLY) as ctx:
            session.model(**kwargs, past_key_values=None, use_cache=False, return_dict=True)
            tensors = ctx.drain()
    finally:
        for h in handles:
            h.remove()
    out: dict[int, Any] = {}
    for li in range(session.arch.num_layers):
        k = f"layer_{li:02d}/hidden_out"
        if k in tensors:
            out[li] = tensors[k]
    return out


def _run_loop(session: SteeringSession, prompt_cfg: PromptConfig, cap_cfg: CaptureConfig,
              state: _SteerState | None = None, embed_override: EmbedOverride | None = None):
    """Autoregressive prefill+decode loop (mirrors generation.run_generation) with cur_offset tracking.

    Hooks, if any, must be registered by the caller. `embed_override` applies a soft token swap on
    the prefill pass. Yields one StepRecord per pass.
    """
    model, tokenizer, arch, device = session.model, session.tokenizer, session.arch, session.device
    gen = session.gen_cfg
    torch.manual_seed(gen.seed)
    generator = torch.Generator(device="cpu").manual_seed(gen.seed) if gen.do_sample else None

    input_ids = _build_input_ids(tokenizer, prompt_cfg, device)
    eos_id = tokenizer.eos_token_id
    past_kv = None
    cur_input = input_ids
    token_history = list(input_ids[0].tolist())
    prompt_len = len(token_history)

    for step in range(gen.max_new_tokens):
        is_prefill = step == 0
        if state is not None:
            state.cur_offset = 0 if is_prefill else (prompt_len + step - 1)
        fwd = _forward_kwargs(session, cur_input, is_prefill, embed_override)
        with torch.no_grad(), CaptureContext(model, arch, cap_cfg) as ctx:
            out = model(
                **fwd, past_key_values=past_kv, use_cache=True,
                output_attentions=cap_cfg.attention_weights, return_dict=True,
            )
            ctx.record_attentions(getattr(out, "attentions", None))
            ctx.record_logits(out.logits)
            tensors = ctx.drain()
        past_kv = out.past_key_values
        logits_last = out.logits[:, -1, :]
        top_k, argmax_id, argmax_lp, logit_stats = _top_k_payload(logits_last, tokenizer, cap_cfg.top_k_probs)
        next_id = _pick_next_token(logits_last, gen, generator)
        input_ids_list = cur_input[0].tolist()
        seq_len = len(token_history) if is_prefill else len(token_history) + 1
        yield StepRecord(
            step=step, kind="prefill" if is_prefill else "decode",
            input_token_ids=input_ids_list, input_tokens=[tokenizer.decode([i]) for i in input_ids_list],
            generated_token_id=next_id, generated_token=tokenizer.decode([next_id]),
            logit_argmax_id=argmax_id, logit_argmax_logprob=argmax_lp, top_k=top_k,
            seq_len=seq_len, tensors=tensors, logit_stats=logit_stats,
        )
        token_history.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        if step + 1 >= gen.max_new_tokens:
            break
        cur_input = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)


def _generate_ids(session: SteeringSession, state: _SteerState | None = None,
                  embed_override: EmbedOverride | None = None) -> list[int]:
    """Run the loop with a light capture config and return only the generated token ids."""
    light = CaptureConfig(hidden_states=False, attention_weights=False, qkv=False, mlp=False,
                          logits=False, top_k_probs=0)
    handles = _register_hooks(session, state) if state is not None else []
    ids: list[int] = []
    try:
        for rec in _run_loop(session, _prompt_cfg(session), light, state, embed_override):
            if rec.generated_token_id is not None:
                ids.append(rec.generated_token_id)
    finally:
        for h in handles:
            h.remove()
    return ids


# --------------------------------------------------------------------------- lens shaping

def _lens_payload(session: SteeringSession, hiddens: dict[int, Any], position: int, top_k: int) -> list[dict]:
    frames = logit_lens_batch(hiddens, session.model, position=position, top_k=top_k, tokenizer=session.tokenizer)
    out: list[dict] = []
    for fr in frames:
        predicted = [
            {"id": t["id"], "token": t["token"], "logprob": t["logprob"], "prob": math.exp(t["logprob"])}
            for t in fr.top_k
        ]
        out.append({
            "layer": fr.layer,
            "argmax_id": fr.argmax_id,
            "argmax_token": (predicted[0]["token"] if predicted else None),
            "argmax_prob": math.exp(fr.argmax_logprob),
            "kl_from_final": fr.kl_from_final,
            "predicted": predicted,
        })
    return out


def _positions(session: SteeringSession) -> list[dict]:
    prompt_len = session.prompt_len
    out = []
    for i, tid in enumerate(session.full_ids):
        out.append({
            "abs": i,
            "section": "prompt" if i < prompt_len else "generated",
            "rel": i if i < prompt_len else i - prompt_len,
            "token_id": tid,
            "token": session.tokenizer.decode([tid]),
        })
    return out


# --------------------------------------------------------------------------- public operations

def run_base(session: SteeringSession) -> dict:
    with INFERENCE_LOCK:
        prompt_ids = _build_input_ids(session.tokenizer, _prompt_cfg(session), session.device)[0].tolist()
        generated_ids = _generate_ids(session)
        session.prompt_ids = prompt_ids
        session.base_generated_ids = generated_ids
        session.full_ids = prompt_ids + generated_ids
        session.base_hiddens = _capture_hiddens(session, session.full_ids)
    eos_id = session.tokenizer.eos_token_id
    return {
        "prompt_len": len(prompt_ids),
        "full_len": len(session.full_ids),
        "positions": _positions(session),
        "generated_text": session.tokenizer.decode(generated_ids),
        "eos_hit": bool(generated_ids and generated_ids[-1] == eos_id),
        "num_layers": session.arch.num_layers,
        "model_id": session.model_cfg.id,
    }


def get_lens(session: SteeringSession, position: int, top_k: int) -> dict:
    if not session.base_hiddens:
        raise ValueError("run base first")
    pos = _norm_pos(session, position)
    with INFERENCE_LOCK:
        layers = _lens_payload(session, session.base_hiddens, pos, top_k)
    return {
        "model_id": session.model_cfg.id,
        "num_layers": session.arch.num_layers,
        "seq_len": len(session.full_ids),
        "position": pos,
        "layers": layers,
    }


def _norm_pos(session: SteeringSession, position: int) -> int:
    n = len(session.full_ids)
    pos = position if position >= 0 else n + position
    if not (0 <= pos < n):
        raise ValueError(f"position {position} out of range [0,{n})")
    return pos


def _relative_target(p0: float, t: float) -> float:
    return p0 + t * (1.0 - p0) if t >= 0 else p0 * (1.0 + t)


def _direction(session: SteeringSession, token_id: int) -> torch.Tensor:
    if not (0 <= token_id < session.lm_head.weight.shape[0]):
        raise ValueError(f"token_id {token_id} out of vocab range")
    return session.lm_head.weight[token_id].detach().to(device=session.device, dtype=torch.float32)


def _prob_of_token(session: SteeringSession, hiddens: dict[int, Any], pos: int,
                   layers: list[int], token_id: int, direction: torch.Tensor) -> dict[int, float]:
    """Baseline prob(token_id) at `pos` for each requested layer, via the exact lens projection."""
    import numpy as np
    dtype = next(session.model.parameters()).dtype
    out: dict[int, float] = {}
    for li in layers:
        arr = hiddens.get(li)
        if arr is None:
            continue
        a = arr[0] if arr.ndim == 3 else arr
        vec = torch.from_numpy(np.asarray(a[pos], dtype=np.float32))
        logits = project_residual_to_logits(vec, session.final_norm, session.lm_head, session.device, dtype)
        out[li] = float(torch.log_softmax(logits, dim=-1)[token_id].exp().item())
    return out


def _resolve_targets(session: SteeringSession, edit: SteerEdit, at_pos: int,
                     direction: torch.Tensor) -> dict[int, float]:
    """Per-layer target prob for token X at `at_pos` (absolute vs relative-to-baseline)."""
    if edit.mode == "absolute":
        return {li: float(edit.value) for li in edit.layers}
    p0 = _prob_of_token(session, session.base_hiddens, at_pos, edit.layers, edit.token_id, direction)
    return {li: _relative_target(p0.get(li, 0.0), edit.value) for li in edit.layers}


def _solve_fixed_alpha(session: SteeringSession, at_pos: int, targets: dict[int, float],
                       token_id: int, direction: torch.Tensor) -> tuple[dict[int, float], list[dict]]:
    """Solve one alpha per layer at the unsteered `at_pos` residual (for broadcast steering)."""
    import numpy as np
    dtype = next(session.model.parameters()).dtype
    fixed: dict[int, float] = {}
    solved: list[dict] = []
    for li in sorted(targets):
        arr = session.base_hiddens.get(li)
        if arr is None:
            continue
        a = arr[0] if arr.ndim == 3 else arr
        h = torch.from_numpy(np.asarray(a[at_pos], dtype=np.float32)).to(session.device)
        sol = solve_steer_alpha(h, direction, token_id, targets[li],
                                session.final_norm, session.lm_head, session.device, dtype)
        fixed[li] = sol.alpha
        solved.append({
            "layer": li, "p0": sol.p0, "target_prob": sol.target_prob,
            "alpha": sol.alpha, "achieved_prob": sol.achieved_prob, "saturated": sol.saturated,
        })
    return fixed, solved


def _swap_delta(session: SteeringSession, edit: SteerEdit, pos: int) -> torch.Tensor:
    """Embedding delta for a soft token swap: w·(emb[X] − emb[original]) at `pos`."""
    if pos >= session.prompt_len:
        raise ValueError("soft token swap only applies to prompt tokens (they feed the model as input)")
    embed = get_module(session.model, session.arch.embed_module_path)
    vocab = embed.weight.shape[0]
    if not (0 <= edit.token_id < vocab):
        raise ValueError(f"token_id {edit.token_id} out of vocab range")
    orig_id = session.full_ids[pos]
    return (float(edit.value) * (embed.weight[edit.token_id] - embed.weight[orig_id])).detach()


def _make_intervention(session: SteeringSession, edit: SteerEdit, pos: int):
    """Build the intervention for an edit. Returns (state, embed_override, solved, lens_pos).

    - point:     (state, None, None, pos)            — solved is filled by the hook during capture.
    - broadcast: (state, None, solved_list, frontier) — alpha solved once at the answer frontier.
    - swap:      (None, (pos, delta), [], pos)        — embedding interpolation at prefill.
    """
    if edit.kind == "swap":
        return None, (pos, _swap_delta(session, edit, pos)), [], pos

    bad = [li for li in edit.layers if not (0 <= li < session.arch.num_layers)]
    if bad:
        raise ValueError(f"layers out of range: {bad}")
    direction = _direction(session, edit.token_id)

    if edit.kind == "broadcast":
        frontier = session.prompt_len - 1          # the position that produces the first answer word
        targets = _resolve_targets(session, edit, frontier, direction)
        fixed_alpha, solved = _solve_fixed_alpha(session, frontier, targets, edit.token_id, direction)
        state = _SteerState(pos, edit.token_id, direction, targets, session,
                            broadcast=True, fixed_alpha=fixed_alpha)
        return state, None, solved, frontier

    targets = _resolve_targets(session, edit, pos, direction)
    return _SteerState(pos, edit.token_id, direction, targets, session), None, None, pos


def run_steer(session: SteeringSession, edit: SteerEdit, top_k: int) -> dict:
    if not session.base_hiddens:
        raise ValueError("run base first")
    pos = _norm_pos(session, edit.position)

    with INFERENCE_LOCK:
        state, override, solved_pre, lens_pos = _make_intervention(session, edit, pos)
        # steered lens at `lens_pos` — a forward over full_ids[:lens_pos+1] is enough because
        # positions <= lens_pos are causally unaffected by anything after them.
        steered_hiddens = _capture_hiddens(session, session.full_ids[: lens_pos + 1], state, override)
        lens_layers = _lens_payload(session, steered_hiddens, -1, top_k)
        if solved_pre is not None:
            solved = solved_pre
        elif state is not None:
            solved = [state.solved[li] for li in sorted(state.solved)]
        else:
            solved = []
        steered_ids = _generate_ids(session, state, override)

    diverged = None
    for i, tid in enumerate(steered_ids):
        base_tid = session.base_generated_ids[i] if i < len(session.base_generated_ids) else None
        if base_tid != tid:
            diverged = session.prompt_len + i
            break

    session.last_edit = edit
    session.last_steered_ids = steered_ids
    session.last_solved = solved

    swap_info = None
    if edit.kind == "swap":
        swap_info = {
            "weight": float(edit.value),
            "from_token": session.tokenizer.decode([session.full_ids[pos]]),
            "to_token": session.tokenizer.decode([edit.token_id]),
        }

    return {
        "kind": edit.kind,
        "position": pos,
        "lens_position": lens_pos,
        "solved": solved,
        "swap": swap_info,
        "generated_text": session.tokenizer.decode(steered_ids),
        "steered_token_ids": steered_ids,
        "full_len": session.prompt_len + len(steered_ids),
        "eos_hit": bool(steered_ids and steered_ids[-1] == session.tokenizer.eos_token_id),
        "diverged_from_base_at": diverged,
        "lens": {
            "model_id": session.model_cfg.id,
            "num_layers": session.arch.num_layers,
            "seq_len": session.prompt_len + len(steered_ids),
            "position": lens_pos,
            "layers": lens_layers,
        },
    }


# --------------------------------------------------------------------------- save

def _tensor_index(tensors: dict) -> dict:
    return {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in tensors.items()}


def _safe_slug(text: str, fallback: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")
    return cleaned or fallback


def _compact_layers(layers: list[int], num_layers: int) -> tuple[str, str]:
    """Return (human, slug) for a layer set: half-open ranges (e.g. '0:22-23:48'). Slug uses '.'."""
    ls = sorted(set(layers))
    if not ls:
        return ("none", "none")
    if ls == list(range(num_layers)):
        return (f"0:{num_layers}", f"0.{num_layers}")
    ranges: list[tuple[int, int]] = []
    start = prev = ls[0]
    for v in ls[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append((start, prev))
        start = prev = v
    ranges.append((start, prev))
    human = "-".join(f"{a}:{b + 1}" for a, b in ranges)  # half-open end
    slug = human.replace(":", ".")
    return (human, slug)


def _steer_labels(session: SteeringSession, edit: SteerEdit) -> tuple[str, str]:
    """Build (human_label, dir_slug). point: token-[steer|swap]-with-layers-prob;
    broadcast: token-broadcast-…; swap: token-softswap-to-w{weight}."""
    pos_tok = session.tokenizer.decode([session.full_ids[edit.position]]).strip() or "tok"
    to_tok = session.tokenizer.decode([edit.token_id]).strip() or f"id{edit.token_id}"

    if edit.kind == "swap":
        parts = [pos_tok, "softswap", to_tok, f"w{edit.value:.2f}"]
        human = "-".join(parts)
        return human, _safe_slug(human, "steer")

    # residual kinds: base next-token argmax at this position/final layer decides steer vs swap
    base_layer = session.base_hiddens.get(max(session.base_hiddens)) if session.base_hiddens else None
    swap = True
    if base_layer is not None:
        import numpy as np
        a = base_layer[0] if base_layer.ndim == 3 else base_layer
        vec = torch.from_numpy(np.asarray(a[edit.position], dtype=np.float32))
        logits = project_residual_to_logits(vec, session.final_norm, session.lm_head,
                                             session.device, next(session.model.parameters()).dtype)
        swap = int(logits.argmax().item()) != edit.token_id
    verb = "swap" if swap else "steer"
    layers_h, layers_s = _compact_layers(edit.layers, session.arch.num_layers)
    prob = f"t{edit.value:+.2f}" if edit.mode == "relative" else f"p{edit.value:.2f}"
    prefix = ["broadcast"] if edit.kind == "broadcast" else []
    tail = ([to_tok] if swap else []) + [layers_h, prob]
    human = "-".join([pos_tok] + prefix + [verb] + tail)
    slug_tail = ([to_tok] if swap else []) + [layers_s, prob]
    slug = _safe_slug("-".join([pos_tok] + prefix + [verb] + slug_tail), "steer")
    return human, slug


def _write_run(session: SteeringSession, scenario: Scenario, parent: Path, label: str,
               state: _SteerState | None, sidecar: dict | None,
               embed_override: EmbedOverride | None = None) -> tuple[str, str]:
    """Run generation (optionally steered) with full capture and write a run directory. Returns (group, label)."""
    paths = make_run_dir(parent, label)
    prompt_ids = _build_input_ids(session.tokenizer, _prompt_cfg(session), session.device)[0].tolist()
    write_run_manifest(paths, scenario, session.arch, session.tokenizer, session.device, prompt_ids)

    handles = _register_hooks(session, state) if state is not None else []
    entries: list[dict] = []
    gen_ids: list[int] = []
    try:
        for rec in _run_loop(session, _prompt_cfg(session), session.cap_cfg, state, embed_override):
            entries.append(write_step(paths, rec))
            if rec.generated_token_id is not None:
                gen_ids.append(rec.generated_token_id)
    finally:
        for h in handles:
            h.remove()
    write_steps_json(paths, entries)

    # conversation snapshots (each message prefix + the generated assistant turn)
    generated_text = session.tokenizer.decode(gen_ids)
    _write_conversation_snapshots(session, paths.root, generated_text)

    if sidecar is not None:
        (paths.root / "steer.json").write_text(json.dumps(sidecar, indent=2, default=str))
    return parent.name, label


def _write_conversation_snapshots(session: SteeringSession, run_dir: Path, generated_text: str) -> None:
    messages = list(session.messages)
    if generated_text:
        messages.append(Message(role="assistant", content=generated_text))
    if not messages:
        return
    snap_info = []
    for i, msg in enumerate(messages):
        prefix = messages[: i + 1]
        tensors, input_ids = capture_reference_prefill(
            session.model, session.tokenizer, session.arch, prefix, session.cap_cfg, session.device
        )
        write_conversation_snapshot(run_dir, i, msg.role, tensors)
        snap_info.append({
            "index": i, "role": msg.role, "content_preview": msg.content[:80],
            "input_token_ids": input_ids, "input_tokens": [session.tokenizer.decode([t]) for t in input_ids],
            "seq_len": len(input_ids), "tensors_file": f"snapshot_{i:02d}_{msg.role}.npz",
            "tensor_index": _tensor_index(tensors),
            "generated": bool(generated_text) and i == len(messages) - 1,
        })
    write_conversation_snapshot_index(run_dir, snap_info)


def save_run(session: SteeringSession, runs_dir: Path, label: str | None = None) -> dict:
    if session.last_edit is None:
        raise ValueError("no steered run to save; run a steer first")
    edit = session.last_edit
    human, slug = _steer_labels(session, edit)
    name = _safe_slug(f"steer-{slug}", "steer")

    scenario = Scenario(
        name=name,
        model=session.model_cfg,
        prompt=[PromptConfig(messages=session.messages, run_at_each_message=False)],
        generation=session.gen_cfg,
        capture=session.cap_cfg,
        output=OutputConfig(dir=str(runs_dir)),
    )

    with INFERENCE_LOCK:
        parent = make_scenario_dir(scenario)
        write_scenario_manifest(parent, scenario)
        # base run
        _write_run(session, scenario, parent, "base", None, None)
        # steered run — rebuild the same intervention (point / broadcast / swap) used at run time
        pos = _norm_pos(session, edit.position)
        state, override, _, _ = _make_intervention(session, edit, pos)
        sidecar = {
            "edit": edit.model_dump(),
            "kind": edit.kind,
            "human_label": human,
            "position_token": session.tokenizer.decode([session.full_ids[pos]]),
            "solved": session.last_solved,
            "base_text": session.tokenizer.decode(session.base_generated_ids),
            "steered_text": session.tokenizer.decode(session.last_steered_ids),
        }
        _write_run(session, scenario, parent, "steered", state, sidecar, override)

    from .loader import RUN_ID_SEP
    return {
        "group": parent.name,
        "base_run_id": f"{parent.name}{RUN_ID_SEP}base",
        "steered_run_id": f"{parent.name}{RUN_ID_SEP}steered",
        "human_label": human,
    }
