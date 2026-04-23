# Activation Lab

Local research harness that loads a HuggingFace causal LM, runs a scenario (prompt + generation settings), and emits a full per-step log of the model's internal activations: hidden states, Q/K/V projections, attention weights, MLP intermediates, and logits with top-k probabilities.

Every forward pass produces one record:
- Metadata in `run.json` + `steps.json` (human-readable).
- Tensors in `tensors/step_NNN.npz` (compressed NumPy).

## Install

Requires Python 3.11+. On Apple Silicon, MPS is used by default; falls back to CPU.

```bash
# with uv (recommended on macOS)
uv venv
uv pip install -e .

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
activation-lab run scenarios/example_qwen_basic.yaml
```

```bash
uv run activation-lab run scenarios/example_qwen_basic.yaml
```

First run downloads `Qwen/Qwen2.5-3B-Instruct` (~6 GB) into `~/.cache/huggingface`.

Output lands in `runs/qwen_basic_smoke_<utc-timestamp>/`:

```
run.json               # scenario echo, model arch, tokenizer info, env
steps.json             # per-step metadata + tensor index
tensors/
  step_000.npz         # prefill: activations over full prompt
  step_001.npz         # first decode token
  ...
```

Read a tensor back:

```python
import numpy as np
d = np.load("runs/.../tensors/step_001.npz")
print(sorted(d.files)[:5])
print(d["layer_00/hidden_out"].shape)   # (1, 1, hidden_size) at decode
```

## Scenario format

A scenario YAML has five sections: `model`, `prompt`, `generation`, `capture`, `output`. See [`scenarios/example_qwen_basic.yaml`](scenarios/example_qwen_basic.yaml) for the canonical example.

Key toggles in `capture`:
- `hidden_states`, `attention_weights`, `qkv`, `mlp`, `logits` — turn sections on/off.
- `layers: [0, 5, 10, 35]` — capture a subset instead of `all` to save disk.
- `top_k_probs` — how many top tokens to log per step.
- `store_dtype: float16` — round tensors before writing (big size saving; default).

## Captured tensors per layer

For each captured layer `i`, the NPZ contains keys under `layer_{i:02d}/`:

| Key | Shape | Source |
|---|---|---|
| `hidden_in` | `(B, T, H)` | pre-hook on decoder block input |
| `hidden_out` | `(B, T, H)` | block output (residual stream out) |
| `q`, `k`, `v` | `(B, T, H_kv)` | projections, pre-RoPE, pre-head-split |
| `attn_out` | `(B, T, H)` | attention block output after `o_proj` |
| `attn_weights` | `(B, heads, T, T)` | softmax output, from `output_attentions=True` |
| `mlp_gate` | `(B, T, I)` | `gate_proj` output (pre-activation) |
| `mlp_up` | `(B, T, I)` | `up_proj` output |
| `mlp_act` | `(B, T, I)` | post-activation (`silu(gate)` for llama-family) |
| `mlp_down_in` | `(B, T, I)` | input to `down_proj` (= `act * up`) |
| `mlp_down_out` | `(B, T, H)` | MLP block output |

Global keys:
- `embeddings` — token embedding lookup output.
- `logits` — final `lm_head` output, `(B, T, V)`.

Prefill (step 0) has `T = prompt_length`. Decode steps have `T = 1`.

## Generation semantics

`max_new_tokens = N` means exactly **N forward passes** and **N generated tokens**:
- Step 0 (prefill): ingests the full prompt, emits the first generated token.
- Steps 1..N-1 (decode): each ingests the previous generated token, emits the next.

Greedy decoding is the default for reproducibility. `do_sample: true` plus `top_k` / `top_p` / `temperature` enables sampling; sampling uses a CPU generator seeded by `generation.seed` so results are reproducible regardless of device.

## Interpretability

All analyses operate on a completed run directory and use the captured tensors (some also reload the model for projection steps).

### Logit lens

Projects every layer's `hidden_out` through the model's final norm + `lm_head` so you can see what the model *would* predict from each layer's residual stream, and how that distribution converges to the real output.

```bash
activation-lab logit-lens runs/qwen_basic_smoke_<ts>
# → runs/.../logit_lens.json
```

### Residual-stream decomposition

For each layer at each step: `||hidden_in||`, `||attn_out||`, `||mlp_out||`, `||hidden_out||`, and the cosine alignment of the attn/mlp contributions with the layer's output.

### Neuron trajectory

For a chosen `(layer, neuron, source)`, read the activation at the last captured position across every step — useful for spotting neurons that fire on specific generated tokens.

### Cross-layer cosine similarity

Cosine matrix over layers using each layer's `hidden_out` (at the last position) as a signature. Shows representation drift vs redundancy across depth.

## UI — Streamlit

```bash
activation-lab view runs/qwen_basic_smoke_<ts>
# opens http://localhost:8501
```

Tabs: **Heatmap** (any per-layer tensor), **Residual stream**, **Logit lens** (reloads the model on first use — ~60s), **Neuron** (slider over neuron index; generated tokens annotated on the trajectory), **Cross-layer** cosine.

## Notebook

```bash
uv pip install -e '.[notebook]'
jupyter lab notebooks/explore.ipynb
```

Or just use the `Run` API directly:

```python
from activation_lab.view import Run
run = Run("runs/qwen_basic_smoke_<ts>")

run.summary()
run.heatmap(layer=15, step=0, source="hidden_out")
run.plot_residual_norms(step=0)
run.plot_neuron(layer=15, neuron=42, source="mlp_act")
run.plot_cross_layer_cosine(step=0)
run.logit_lens_table(step=0, top_k=5)    # reloads model lazily
```

## Other commands

```bash
activation-lab layers Qwen/Qwen2.5-3B-Instruct   # print module tree
activation-lab inspect runs/.../run.json         # pretty summary of a completed run
activation-lab heatmap runs/.../                 # batch PNG heatmaps per (layer, step)
```

## Notes & caveats

- **MPS**: `attn_implementation="eager"` is forced so `output_attentions=True` returns usable tensors. This disables the fused SDPA kernel → slower than generate-only inference, but that's the price of capture.
- **Memory**: activations are moved to CPU inside each hook immediately. For a 3B model over a 100-token prompt, one prefill NPZ can easily be 100+ MB — use `layers: [...]` to narrow if disk is tight.
- **Reproducibility**: greedy + fixed dtype + CPU sampling generator are stable across runs on the same machine. MPS is not bit-identical across Mac generations.
- **Custom architectures**: the module-path discovery handles llama/qwen/mistral/gemma/gpt-neox families. Unusual models may need `trust_remote_code: true` in the `model` block.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

Uses `hf-internal-testing/tiny-random-LlamaForCausalLM` (~2 MB) — runs in seconds without internet after the first download.
