"""Compute and render activation heatmaps from a run directory.

For each (layer, step) pair, extracts a 2D slice from a chosen tensor source and writes:
  - a PNG heatmap
  - a sibling .npy file with the raw 2D array used

Supported sources:
  hidden_in, hidden_out, q, k, v, attn_out, attn_weights,
  mlp_gate, mlp_up, mlp_act, mlp_down_in, mlp_down_out

All sources except `attn_weights` expect NPZ tensors shaped (B, T, D) → rendered as (T, D).
`attn_weights` has shape (B, heads, T, T). By default heads are averaged → (T, T).
With per_head=True, each head is rendered separately into a head-indexed subfolder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Normalization = Literal["none", "per_image", "global", "signed"]


@dataclass
class HeatmapRequest:
    run_dir: Path
    source: str = "hidden_out"
    layers: list[int] | None = None  # None = all captured layers
    steps: list[int] | None = None  # None = all steps
    per_head: bool = False  # only affects attn_weights
    normalize: Normalization = "per_image"
    reduce: Literal["abs", "signed"] = "signed"
    cmap: str = "viridis"
    dpi: int = 100
    out_subdir: str = "heatmaps"


def _load_steps_index(run_dir: Path) -> list[dict]:
    steps_path = run_dir / "steps.json"
    data = json.loads(steps_path.read_text())
    return data["steps"]


def _discover_layers(npz_files: list[str]) -> list[int]:
    idx: set[int] = set()
    for name in npz_files:
        if name.startswith("layer_") and "/" in name:
            head = name.split("/", 1)[0]
            try:
                idx.add(int(head.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(idx)


def _extract_2d(arr: np.ndarray, source: str) -> np.ndarray:
    """Collapse a captured tensor into a 2D array suitable for imshow."""
    if source == "attn_weights":
        # (B, heads, T, T) → mean over batch, head-axis handled by caller
        if arr.ndim != 4:
            raise ValueError(f"attn_weights expected 4D, got shape {arr.shape}")
        return arr[0]  # (heads, T, T)
    # Typical (B, T, D)
    if arr.ndim == 3:
        return arr[0]  # (T, D)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unsupported shape {arr.shape} for source {source!r}")


def _normalize(arr: np.ndarray, mode: Normalization, reduce: str) -> tuple[np.ndarray, float, float]:
    data = arr
    if reduce == "abs":
        data = np.abs(data)

    if mode == "none":
        return data, float(data.min()), float(data.max())
    if mode == "per_image":
        lo, hi = float(data.min()), float(data.max())
        return data, lo, hi
    if mode == "signed":
        m = float(np.abs(data).max())
        if m == 0.0:
            m = 1.0
        return data, -m, m
    if mode == "global":
        # Caller supplies global range via plot kwargs; here we just pass through.
        return data, float(data.min()), float(data.max())
    raise ValueError(f"unknown normalize mode: {mode}")


def _render_png(
    data: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
    xlabel: str,
    ylabel: str,
) -> None:
    h, w = data.shape
    # Aspect: make images roughly square-ish when w >> h (typical for hidden × seq=1).
    fig_w = max(4.0, min(16.0, w / 200.0 + 4.0))
    fig_h = max(2.0, min(10.0, h / 10.0 + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def generate(req: HeatmapRequest) -> list[Path]:
    """Produce heatmaps for (layer, step) pairs. Returns list of written PNG paths."""
    run_dir = req.run_dir
    steps = _load_steps_index(run_dir)
    if req.steps is not None:
        wanted = set(req.steps)
        steps = [s for s in steps if s["step"] in wanted]

    out_root = run_dir / req.out_subdir / req.source
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for step_meta in steps:
        step_idx = step_meta["step"]
        npz_path = run_dir / step_meta["tensors_file"]
        if not npz_path.exists():
            continue
        with np.load(npz_path) as npz:
            layers = req.layers if req.layers is not None else _discover_layers(list(npz.files))
            for layer in layers:
                key = f"layer_{layer:02d}/{req.source}"
                if key not in npz.files:
                    continue
                raw = npz[key]
                try:
                    slab = _extract_2d(raw, req.source)
                except ValueError:
                    continue

                if req.source == "attn_weights":
                    # slab: (heads, T, T)
                    if req.per_head:
                        for h in range(slab.shape[0]):
                            written.append(
                                _write_one(
                                    slab[h],
                                    out_root / f"layer_{layer:02d}" / f"step_{step_idx:03d}_head_{h:02d}",
                                    req,
                                    title=f"{req.source} L{layer} step{step_idx} head{h}",
                                    xlabel="key pos",
                                    ylabel="query pos",
                                )
                            )
                    else:
                        merged = slab.mean(axis=0)
                        written.append(
                            _write_one(
                                merged,
                                out_root / f"layer_{layer:02d}_step_{step_idx:03d}",
                                req,
                                title=f"{req.source} (head-mean) L{layer} step{step_idx}",
                                xlabel="key pos",
                                ylabel="query pos",
                            )
                        )
                else:
                    # slab: (T, D)
                    written.append(
                        _write_one(
                            slab,
                            out_root / f"layer_{layer:02d}_step_{step_idx:03d}",
                            req,
                            title=f"{req.source} L{layer} step{step_idx} shape={tuple(slab.shape)}",
                            xlabel="channel",
                            ylabel="token pos",
                        )
                    )
    return written


def _write_one(
    data: np.ndarray,
    stem: Path,
    req: HeatmapRequest,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    npy_path = stem.with_suffix(".npy")

    data_norm, vmin, vmax = _normalize(data, req.normalize, req.reduce)
    np.save(npy_path, data_norm)
    _render_png(
        data_norm,
        png_path,
        title=title,
        cmap=req.cmap,
        vmin=vmin,
        vmax=vmax,
        dpi=req.dpi,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return png_path
