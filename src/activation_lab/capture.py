from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .models import ArchInfo, get_module
from .scenario import CaptureConfig


def _to_numpy(t: torch.Tensor, store_dtype: str) -> np.ndarray:
    # torch.bfloat16 has no numpy equivalent; upcast to float32 first.
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    t = t.detach().to("cpu")
    if store_dtype == "float16":
        t = t.to(torch.float16)
    elif store_dtype == "float32":
        t = t.to(torch.float32)
    # bfloat16 storage round-trip: numpy can't hold bf16, so we store as float32.
    elif store_dtype == "bfloat16":
        t = t.to(torch.float32)
    return t.numpy()


def _resolve_layer_indices(num_layers: int, spec: list[int] | str) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    return sorted(set(i for i in spec if 0 <= i < num_layers))


class CaptureContext:
    """Registers forward hooks and collects activations into a per-step buffer.

    Usage:
        with CaptureContext(model, arch, capture_cfg) as ctx:
            out = model(input_ids, use_cache=True, output_attentions=True)
            ctx.record_attentions(out.attentions)   # from HF output
            tensors = ctx.drain()
    """

    def __init__(self, model: torch.nn.Module, arch: ArchInfo, cfg: CaptureConfig):
        self.model = model
        self.arch = arch
        self.cfg = cfg
        self.layer_indices = _resolve_layer_indices(arch.num_layers, cfg.layers)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._buffer: dict[str, np.ndarray] = {}

    # --- lifecycle ---

    def __enter__(self) -> CaptureContext:
        self._register_hooks()
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # --- public ---

    def drain(self) -> dict[str, np.ndarray]:
        out = self._buffer
        self._buffer = {}
        return out

    def record_attentions(self, attentions: tuple[torch.Tensor, ...] | None) -> None:
        if not self.cfg.attention_weights or attentions is None:
            return
        for idx in self.layer_indices:
            if idx < len(attentions):
                self._store(f"layer_{idx:02d}/attn_weights", attentions[idx])

    def record_logits(self, logits: torch.Tensor) -> None:
        if self.cfg.logits:
            self._store("logits", logits)

    # --- internals ---

    def _store(self, key: str, tensor: torch.Tensor) -> None:
        self._buffer[key] = _to_numpy(tensor, self.cfg.store_dtype)

    def _hook_output(self, key: str) -> Callable:
        def _h(_mod, _inp, out):
            # Some modules return a tuple; take the first tensor element.
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                self._store(key, t)
        return _h

    def _hook_pre_input(self, key: str) -> Callable:
        def _h(_mod, inp):
            t = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
            if isinstance(t, torch.Tensor):
                self._store(key, t)
        return _h

    def _register_hooks(self) -> None:
        layers = get_module(self.model, self.arch.layer_module_path)

        if self.cfg.hidden_states:
            embed = get_module(self.model, self.arch.embed_module_path)
            self._handles.append(
                embed.register_forward_hook(self._hook_output("embeddings"))
            )

        for idx in self.layer_indices:
            layer = layers[idx]
            tag = f"layer_{idx:02d}"

            if self.cfg.hidden_states:
                self._handles.append(
                    layer.register_forward_pre_hook(self._hook_pre_input(f"{tag}/hidden_in"))
                )
                self._handles.append(
                    layer.register_forward_hook(self._hook_output(f"{tag}/hidden_out"))
                )

            attn = getattr(layer, self.arch.attn_sublayer)
            if self.cfg.qkv:
                for name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(attn, name):
                        short = name[0]  # q / k / v
                        self._handles.append(
                            getattr(attn, name).register_forward_hook(
                                self._hook_output(f"{tag}/{short}")
                            )
                        )
                if hasattr(attn, "o_proj"):
                    self._handles.append(
                        attn.o_proj.register_forward_hook(
                            self._hook_output(f"{tag}/attn_out")
                        )
                    )

            mlp = getattr(layer, self.arch.mlp_sublayer, None)
            if self.cfg.mlp and mlp is not None:
                for name, out_key in (
                    ("gate_proj", "mlp_gate"),
                    ("up_proj", "mlp_up"),
                    ("act_fn", "mlp_act"),
                    ("down_proj", "mlp_down_out"),
                ):
                    mod = getattr(mlp, name, None)
                    if mod is not None:
                        self._handles.append(
                            mod.register_forward_hook(self._hook_output(f"{tag}/{out_key}"))
                        )
                if hasattr(mlp, "down_proj"):
                    self._handles.append(
                        mlp.down_proj.register_forward_pre_hook(
                            self._hook_pre_input(f"{tag}/mlp_down_in")
                        )
                    )
