from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .capture import CaptureContext
from .models import ArchInfo
from .scenario import CaptureConfig, GenerationConfig, Message, PromptConfig


@dataclass
class StepRecord:
    step: int
    kind: str  # "prefill" | "decode"
    input_token_ids: list[int]
    input_tokens: list[str]
    generated_token_id: int | None
    generated_token: str | None
    logit_argmax_id: int | None
    logit_argmax_logprob: float | None
    top_k: list[dict[str, Any]]
    seq_len: int
    tensors: dict[str, np.ndarray] = field(default_factory=dict)


def _build_input_ids(
    tokenizer, prompt: PromptConfig, device: torch.device
) -> torch.Tensor:
    messages = [m.model_dump() for m in prompt.messages]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Tokenizers without a chat template (e.g. tiny test models) get a flat concat.
        text = "\n".join(m["content"] for m in messages)
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return ids.to(device)


def _pick_next_token(
    logits_last: torch.Tensor, gen: GenerationConfig, generator: torch.Generator | None
) -> int:
    if not gen.do_sample:
        return int(logits_last.argmax(dim=-1).item())

    logits = logits_last / max(gen.temperature, 1e-6)

    if gen.top_k is not None and gen.top_k > 0:
        vals, _ = torch.topk(logits, gen.top_k)
        cutoff = vals[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

    if gen.top_p is not None and 0.0 < gen.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > gen.top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)

    # Move to CPU so that a CPU-pinned generator (for reproducibility) matches the tensor device.
    probs = torch.softmax(logits.float().cpu(), dim=-1)
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_id.item())


def _top_k_payload(
    logits_last: torch.Tensor, tokenizer, k: int
) -> tuple[list[dict[str, Any]], int, float]:
    logprobs = torch.log_softmax(logits_last.float(), dim=-1)
    argmax_id = int(logprobs.argmax(dim=-1).item())
    argmax_lp = float(logprobs[..., argmax_id].item())

    payload: list[dict[str, Any]] = []
    if k > 0:
        top_vals, top_ids = torch.topk(logprobs, k=min(k, logprobs.shape[-1]))
        top_vals = top_vals.squeeze().tolist()
        top_ids = top_ids.squeeze().tolist()
        if isinstance(top_ids, int):
            top_ids = [top_ids]
            top_vals = [top_vals]
        for tid, lp in zip(top_ids, top_vals):
            payload.append(
                {
                    "id": int(tid),
                    "token": tokenizer.decode([int(tid)]),
                    "logprob": float(lp),
                }
            )
    return payload, argmax_id, argmax_lp


def run_generation(
    model: torch.nn.Module,
    tokenizer,
    arch: ArchInfo,
    prompt_cfg: PromptConfig,
    gen_cfg: GenerationConfig,
    cap_cfg: CaptureConfig,
    device: torch.device,
) -> Iterator[StepRecord]:
    """Yields one StepRecord per forward pass (prefill + each decode)."""
    torch.manual_seed(gen_cfg.seed)
    generator: torch.Generator | None = None
    if gen_cfg.do_sample:
        # The mps generator on Apple Silicon does not accept a seed directly for multinomial;
        # CPU generator keeps sampling reproducible regardless of model device.
        generator = torch.Generator(device="cpu").manual_seed(gen_cfg.seed)

    input_ids = _build_input_ids(tokenizer, prompt_cfg, device)
    eos_id = tokenizer.eos_token_id

    past_kv = None
    cur_input = input_ids
    token_history: list[int] = list(input_ids[0].tolist())
    # max_new_tokens forward passes total. Step 0 is prefill (ingests full prompt and emits the
    # first generated token). Steps 1..N-1 are decode (ingest the previous generated token and
    # emit the next). Result: exactly max_new_tokens generated tokens.
    total_steps = gen_cfg.max_new_tokens

    for step in range(total_steps):
        is_prefill = step == 0
        with torch.no_grad(), CaptureContext(model, arch, cap_cfg) as ctx:
            out = model(
                input_ids=cur_input,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=cap_cfg.attention_weights,
                return_dict=True,
            )
            ctx.record_attentions(getattr(out, "attentions", None))
            logits = out.logits  # (B, T, V)
            ctx.record_logits(logits)
            tensors = ctx.drain()

        past_kv = out.past_key_values

        logits_last = logits[:, -1, :]
        top_k, argmax_id, argmax_lp = _top_k_payload(
            logits_last, tokenizer, cap_cfg.top_k_probs
        )
        next_id = _pick_next_token(logits_last, gen_cfg, generator)

        input_ids_list = cur_input[0].tolist()
        seq_len = len(token_history) if is_prefill else len(token_history) + 1

        yield StepRecord(
            step=step,
            kind="prefill" if is_prefill else "decode",
            input_token_ids=input_ids_list,
            input_tokens=[tokenizer.decode([i]) for i in input_ids_list],
            generated_token_id=next_id,
            generated_token=tokenizer.decode([next_id]),
            logit_argmax_id=argmax_id,
            logit_argmax_logprob=argmax_lp,
            top_k=top_k,
            seq_len=seq_len,
            tensors=tensors,
        )

        token_history.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        if step + 1 >= total_steps:
            break
        cur_input = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)


REFERENCE_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"]


def capture_reference_prefill(
    model: torch.nn.Module,
    tokenizer,
    arch: ArchInfo,
    messages: list[Message],
    cap_cfg: CaptureConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run a single prefill pass and return last-token activations per layer.

    Returns {source: (num_layers, hidden_size)} for hidden_in/hidden_out/attn_out/mlp_down_out.
    """
    prompt_cfg = PromptConfig(messages=messages, run_at_each_message=False)
    input_ids = _build_input_ids(tokenizer, prompt_cfg, device)

    with torch.no_grad(), CaptureContext(model, arch, cap_cfg) as ctx:
        model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        tensors = ctx.drain()

    result: dict[str, np.ndarray] = {}
    for src in REFERENCE_SOURCES:
        layers_arr = []
        for layer in range(arch.num_layers):
            key = f"layer_{layer:02d}/{src}"
            if key in tensors:
                arr = np.asarray(tensors[key], dtype=np.float32)
                # shape: (B, T, H) — take last token position
                vec = arr[0, -1, :] if arr.ndim == 3 else arr.flatten()[:arch.hidden_size]
            else:
                vec = np.zeros(arch.hidden_size, dtype=np.float32)
            layers_arr.append(vec)
        result[src] = np.stack(layers_arr)  # (num_layers, hidden_size)
    return result
