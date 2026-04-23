"""Notebook-friendly wrapper around a run directory.

Typical usage in Jupyter:

    from activation_lab.view import Run
    run = Run("runs/qwen_basic_smoke_<ts>")
    run.summary()
    run.heatmap(layer=10, step=0, source="hidden_out")
    run.logit_lens(step=0)
    run.neuron_trajectory(layer=10, neuron=42, source="mlp_act")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..interp import (
    LogitLensFrame,
    cross_layer_cosine,
    logit_lens_for_step,
    neuron_trajectory,
    residual_norms_for_step,
)


@dataclass
class StepView:
    step: int
    kind: str
    generated_token: str | None
    seq_len: int
    npz_path: Path


class Run:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.manifest: dict[str, Any] = json.loads((self.path / "run.json").read_text())
        self.steps_raw: list[dict] = json.loads((self.path / "steps.json").read_text())["steps"]
        self.num_layers: int = self.manifest["model"]["arch"]["num_layers"]
        self.hidden_size: int = self.manifest["model"]["arch"]["hidden_size"]
        self.model_id: str = self.manifest["model"]["id"]
        self._model = None  # lazy
        self._tokenizer = None

    # ---------------------------------------------------------- accessors

    @property
    def steps(self) -> list[StepView]:
        return [
            StepView(
                step=s["step"],
                kind=s["kind"],
                generated_token=s["generated_token"],
                seq_len=s["seq_len"],
                npz_path=self.path / s["tensors_file"],
            )
            for s in self.steps_raw
        ]

    def step_meta(self, step: int) -> dict:
        return next(s for s in self.steps_raw if s["step"] == step)

    def load_npz(self, step: int) -> np.lib.npyio.NpzFile:
        meta = self.step_meta(step)
        return np.load(self.path / meta["tensors_file"])

    def tensor(self, step: int, layer: int, key: str) -> np.ndarray:
        with self.load_npz(step) as npz:
            full_key = key if "/" in key else f"layer_{layer:02d}/{key}"
            return np.asarray(npz[full_key])

    def summary(self) -> dict[str, Any]:
        info = {
            "model": self.model_id,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "device": self.manifest["env"]["device"],
            "num_steps": len(self.steps_raw),
            "generated": "".join(s.get("generated_token") or "" for s in self.steps_raw),
            "prompt": "".join(self.manifest.get("prompt_tokens", [])),
        }
        return info

    # ---------------------------------------------------------- model (lazy)

    def load_model(self):
        """Reload the HF model (used for logit lens). Cached on the instance."""
        if self._model is None:
            from ..models import load_model
            from ..scenario import ModelConfig
            cfg = ModelConfig(**self.manifest["scenario"]["model"])
            model, tok, _arch, _dev = load_model(cfg)
            self._model = model
            self._tokenizer = tok
        return self._model, self._tokenizer

    # ---------------------------------------------------------- plots

    def heatmap(
        self,
        layer: int,
        step: int,
        source: str = "hidden_out",
        cmap: str = "RdBu_r",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        data = self.tensor(step, layer, source)
        if data.ndim == 3:
            data = data[0]
        elif data.ndim == 4:
            # attn_weights: (B, heads, T, T) -> head-mean
            data = data[0].mean(axis=0)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        vmax = float(np.abs(data).max()) or 1.0
        ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{source} layer={layer} step={step} shape={tuple(data.shape)}")
        ax.set_xlabel("channel")
        ax.set_ylabel("pos")
        return ax

    def residual_norms(self, step: int) -> list[dict]:
        with self.load_npz(step) as npz:
            rows = residual_norms_for_step(npz, self.num_layers, step)
        return [asdict(r) for r in rows]

    def plot_residual_norms(self, step: int, ax: plt.Axes | None = None) -> plt.Axes:
        rows = self.residual_norms(step)
        layers = [r["layer"] for r in rows]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, [r["norm_hidden_in"] for r in rows], label="||hidden_in||")
        ax.plot(layers, [r["norm_attn_out"] for r in rows], label="||attn_out||")
        ax.plot(layers, [r["norm_mlp_down_out"] for r in rows], label="||mlp_out||")
        ax.plot(layers, [r["norm_hidden_out"] for r in rows], label="||hidden_out||")
        ax.set_xlabel("layer")
        ax.set_ylabel("L2 norm")
        ax.set_title(f"Residual stream magnitudes — step {step}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def neuron_trajectory(
        self, layer: int, neuron: int, source: str = "hidden_out", position: int = -1
    ) -> np.ndarray:
        return neuron_trajectory(self.path, source, layer, neuron, position=position)

    def plot_neuron(
        self,
        layer: int,
        neuron: int,
        source: str = "hidden_out",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        vals = self.neuron_trajectory(layer, neuron, source)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        xs = np.arange(len(vals))
        ax.plot(xs, vals, marker="o")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("activation")
        ax.set_title(f"Neuron L{layer}#{neuron} trajectory ({source})")
        ax.grid(True, alpha=0.3)
        return ax

    def cross_layer_cosine(self, step: int, source: str = "hidden_out") -> np.ndarray:
        with self.load_npz(step) as npz:
            return cross_layer_cosine(npz, self.num_layers, source=source)

    def plot_cross_layer_cosine(
        self, step: int, source: str = "hidden_out", ax: plt.Axes | None = None
    ) -> plt.Axes:
        M = self.cross_layer_cosine(step, source=source)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"Cross-layer cosine ({source}) — step {step}")
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, shrink=0.8)
        return ax

    # ---------------------------------------------------------- logit lens

    def logit_lens(
        self, step: int, position: int = -1, top_k: int = 10
    ) -> list[LogitLensFrame]:
        model, tok = self.load_model()
        with self.load_npz(step) as npz:
            frames = logit_lens_for_step(
                npz=npz,
                num_layers=self.num_layers,
                model=model,
                tokenizer=tok,
                step=step,
                position=position,
                top_k=top_k,
            )
        return frames

    def logit_lens_table(self, step: int, position: int = -1, top_k: int = 5) -> list[dict]:
        frames = self.logit_lens(step, position=position, top_k=top_k)
        return [
            {
                "layer": f.layer,
                "argmax_token": (f.top_k[0]["token"] if f.top_k else None),
                "argmax_logprob": round(f.argmax_logprob, 3),
                "kl_from_final": None if f.kl_from_final is None else round(f.kl_from_final, 3),
                "top": [(t["token"], round(t["logprob"], 2)) for t in f.top_k],
            }
            for f in frames
        ]
