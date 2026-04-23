from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PromptConfig(BaseModel):
    messages: list[Message]
    run_at_each_message: bool = False


class ModelConfig(BaseModel):
    id: str
    gguf_file: str | None = None
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: Literal["auto", "mps", "cpu"] = "auto"
    trust_remote_code: bool = False


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=8, ge=1)
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 42


class CaptureConfig(BaseModel):
    hidden_states: bool = True
    attention_weights: bool = True
    qkv: bool = True
    mlp: bool = True
    logits: bool = True
    top_k_probs: int = Field(default=20, ge=0)
    layers: list[int] | Literal["all"] = "all"
    store_dtype: Literal["float16", "bfloat16", "float32"] = "float16"


class OutputConfig(BaseModel):
    dir: str = "./runs"
    format: Literal["json+npz"] = "json+npz"


class Scenario(BaseModel):
    name: str
    model: ModelConfig
    prompt: PromptConfig
    generation: GenerationConfig = GenerationConfig()
    capture: CaptureConfig = CaptureConfig()
    output: OutputConfig = OutputConfig()

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not v or any(c in v for c in "/\\ \t\n"):
            raise ValueError("name must be non-empty and contain no whitespace or slashes")
        return v


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    return Scenario.model_validate(data)
