from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import resolve_device, resolve_dtype
from .scenario import ModelConfig


@dataclass
class ArchInfo:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_module_path: str  # e.g. "model.layers"
    embed_module_path: str  # e.g. "model.embed_tokens"
    lm_head_module_path: str  # e.g. "lm_head"
    attn_sublayer: str  # e.g. "self_attn"
    mlp_sublayer: str  # e.g. "mlp"


def _get_attr_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    obj: object = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _discover_paths(model: torch.nn.Module) -> tuple[str, str, str]:
    """Find (layer_list_path, embed_path, lm_head_path) for common causal LMs.

    Handles Llama / Qwen2 / Mistral / Gemma families where the inner decoder lives under
    `model.model` and the head under `lm_head`.
    """
    # layer list
    candidates_layers = ["model.layers", "transformer.h", "gpt_neox.layers"]
    layer_path = next((p for p in candidates_layers if _try_path(model, p)), None)
    if layer_path is None:
        raise RuntimeError("could not locate decoder layer list on this model")

    candidates_embed = ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"]
    embed_path = next((p for p in candidates_embed if _try_path(model, p)), None)
    if embed_path is None:
        raise RuntimeError("could not locate token embedding module")

    lm_head_path = next(
        (p for p in ["lm_head", "embed_out"] if _try_path(model, p)),
        None,
    )
    if lm_head_path is None:
        raise RuntimeError("could not locate LM head module")

    return layer_path, embed_path, lm_head_path


def _try_path(model: torch.nn.Module, path: str) -> bool:
    try:
        _get_attr_path(model, path)
        return True
    except AttributeError:
        return False


def load_model(
    cfg: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ArchInfo, torch.device]:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.id,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    extra = {"gguf_file": cfg.gguf_file} if cfg.gguf_file else {}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.id,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation="eager",  # required for clean output_attentions
        **extra,
    )
    model = model.to(device).eval()

    layer_path, embed_path, head_path = _discover_paths(model)

    layer_list = _get_attr_path(model, layer_path)
    first_layer = layer_list[0]
    attn_sublayer = "self_attn" if hasattr(first_layer, "self_attn") else "attention"
    mlp_sublayer = "mlp"

    cfg_hf = model.config
    arch = ArchInfo(
        num_layers=cfg_hf.num_hidden_layers,
        hidden_size=cfg_hf.hidden_size,
        num_attention_heads=cfg_hf.num_attention_heads,
        num_key_value_heads=getattr(cfg_hf, "num_key_value_heads", cfg_hf.num_attention_heads),
        vocab_size=cfg_hf.vocab_size,
        layer_module_path=layer_path,
        embed_module_path=embed_path,
        lm_head_module_path=head_path,
        attn_sublayer=attn_sublayer,
        mlp_sublayer=mlp_sublayer,
    )
    return model, tokenizer, arch, device


def get_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    return _get_attr_path(model, path)
