"""End-to-end smoke test using a tiny random Llama.

Downloads ~2MB on first run; everything after is cached.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from activation_lab.generation import run_generation
from activation_lab.models import load_model
from activation_lab.scenario import (
    CaptureConfig,
    GenerationConfig,
    ModelConfig,
    Message,
    OutputConfig,
    PromptConfig,
    Scenario,
)
from activation_lab.serialize import (
    make_run_dir,
    write_run_manifest,
    write_step,
    write_steps_json,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    return Scenario(
        name="smoke",
        model=ModelConfig(id=TINY_MODEL, dtype="float32", device="cpu"),
        prompt=PromptConfig(messages=[Message(role="user", content="hi")]),
        generation=GenerationConfig(max_new_tokens=3, do_sample=False, seed=0),
        capture=CaptureConfig(top_k_probs=5, store_dtype="float32"),
        output=OutputConfig(dir=str(tmp_path)),
    )


def test_end_to_end(scenario: Scenario, tmp_path: Path) -> None:
    model, tok, arch, device = load_model(scenario.model)
    paths = make_run_dir(scenario)

    from activation_lab.generation import _build_input_ids
    prompt_ids = _build_input_ids(tok, scenario.prompt, device)[0].tolist()
    write_run_manifest(paths, scenario, arch, tok, device, prompt_ids)

    entries = []
    for rec in run_generation(
        model=model,
        tokenizer=tok,
        arch=arch,
        prompt_cfg=scenario.prompt,
        gen_cfg=scenario.generation,
        cap_cfg=scenario.capture,
        device=device,
    ):
        entries.append(write_step(paths, rec))
    write_steps_json(paths, entries)

    # Run dir exists with expected files.
    assert paths.run_json.exists()
    assert paths.steps_json.exists()
    assert len(entries) == scenario.generation.max_new_tokens

    # Manifest is valid JSON with expected keys.
    manifest = json.loads(paths.run_json.read_text())
    for key in ("scenario", "model", "tokenizer", "prompt_token_ids", "env"):
        assert key in manifest
    assert manifest["model"]["arch"]["num_layers"] == arch.num_layers

    # Each step has an NPZ with at least hidden_out for every captured layer.
    for entry in entries:
        npz_path = paths.root / entry["tensors_file"]
        assert npz_path.exists()
        data = np.load(npz_path)
        for li in range(arch.num_layers):
            assert f"layer_{li:02d}/hidden_out" in data.files
            assert f"layer_{li:02d}/q" in data.files
            assert f"layer_{li:02d}/k" in data.files
            assert f"layer_{li:02d}/v" in data.files
            assert f"layer_{li:02d}/attn_weights" in data.files
            assert f"layer_{li:02d}/mlp_gate" in data.files
            assert f"layer_{li:02d}/mlp_act" in data.files
            assert f"layer_{li:02d}/mlp_down_out" in data.files
        assert "logits" in data.files
        assert "embeddings" in data.files

    # Prefill hidden_out spans full prompt seq_len; decode has seq 1.
    prefill = np.load(paths.root / entries[0]["tensors_file"])
    decode1 = np.load(paths.root / entries[1]["tensors_file"])
    h_prefill = prefill["layer_00/hidden_out"]
    h_decode = decode1["layer_00/hidden_out"]
    assert h_prefill.shape[1] == len(prompt_ids)
    assert h_decode.shape[1] == 1
    assert h_prefill.shape[2] == arch.hidden_size


def test_layer_subset(scenario: Scenario) -> None:
    """layers=[0] should only capture layer 0 activations."""
    scenario.capture.layers = [0]
    model, tok, arch, device = load_model(scenario.model)
    paths = make_run_dir(scenario)

    entries = []
    for rec in run_generation(
        model=model,
        tokenizer=tok,
        arch=arch,
        prompt_cfg=scenario.prompt,
        gen_cfg=scenario.generation,
        cap_cfg=scenario.capture,
        device=device,
    ):
        entries.append(write_step(paths, rec))

    data = np.load(paths.root / entries[0]["tensors_file"])
    assert "layer_00/hidden_out" in data.files
    assert "layer_01/hidden_out" not in data.files
