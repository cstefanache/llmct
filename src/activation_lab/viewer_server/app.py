"""FastAPI app for the Research report viewer.

Routes:
  GET  /api/runs                                   → list runs
  GET  /api/runs/{run_id}                          → run.json
  GET  /api/runs/{run_id}/tree                     → sidebar tree
  GET  /api/runs/{run_id}/npz/{kind}/{name}/meta   → keys+shapes for a single npz
  GET  /api/runs/{run_id}/npz/{kind}/{name}/attention.png?layer=all|N   → attn-softmax
  GET  /api/runs/{run_id}/npz/{kind}/{name}/qk.png?layer=all|N          → raw q·k/√dk
  GET  /api/runs/{run_id}/npz/{kind}/{name}/qkv.png                     → stacked ⟨q,k⟩·v (L, D)
  GET  /api/runs/{run_id}/npz/{kind}/{name}/entropy                     → per-layer entropy
  POST /api/compare/metrics                        → per-layer cos/MAE/STD/Jaccard
  POST /api/compare/heatmap.png                    → pair heatmap (side/diff/sq/hadamard/ratio)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import compute
from .loader import RunRegistry, load_npz, npz_inventory, resolve_npz
from .render import matrix_to_png


Kind = Literal["snapshot", "reference", "step"]


class NpzRef(BaseModel):
    run_id: str
    kind: Kind
    name: str


class MetricsRequest(BaseModel):
    refs: list[NpzRef]
    sources: list[str]


class AdvancedMetricsRequest(BaseModel):
    refs: list[NpzRef]
    sources: list[str]


class HeatmapPairRequest(BaseModel):
    a: NpzRef
    b: NpzRef
    source: Literal["attention", "qk", "qkv"]
    layer: str = "all"
    variant: Literal["a", "b", "abs_diff", "sq_err", "hadamard", "ratio"]


def create_app(runs_dir: Path | None = None) -> FastAPI:
    runs_dir = runs_dir or Path(os.environ.get("ACTIVATION_LAB_RUNS_DIR", "runs")).resolve()
    registry = RunRegistry(runs_dir)

    app = FastAPI(title="Activation Lab Viewer", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------- runs

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return registry.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return registry.run_json(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/runs/{run_id}/tree")
    def get_tree(run_id: str) -> dict:
        try:
            base = registry.run_dir(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

        tree: dict = {"run_id": run_id, "children": []}
        _ = base  # base dir not needed now that the tree no longer probes NPZs

        # conversation snapshots — has_attention is unknown until the user clicks
        # the node and /meta is fetched. Kept as null to avoid opening every NPZ.
        snaps = registry.snapshots_index(run_id)
        if snaps:
            children = []
            for s in snaps:
                fname = s.get("tensors_file") or f"snapshot_{s['index']:02d}_{s['role']}.npz"
                name = fname.removesuffix(".npz")
                children.append({
                    "kind": "snapshot",
                    "name": name,
                    "label": f"{s['index']:02d} {s['role']}: {s.get('content_preview', '')[:50]}",
                    "role": s["role"],
                    "index": s["index"],
                    "has_attention": None,
                })
            tree["children"].append({"type": "conversation_snapshots", "children": children})

        # references — same policy as snapshots
        refs = registry.references_index(run_id)
        if refs:
            children = []
            for label in refs:
                fname = f"ref_{label}.npz"
                name = fname.removesuffix(".npz")
                children.append({
                    "kind": "reference",
                    "name": name,
                    "label": label,
                    "has_attention": None,
                })
            tree["children"].append({"type": "references", "children": children})

        # tensors / steps
        steps = registry.steps_index(run_id)
        if steps:
            children = []
            for s in steps:
                fname = s.get("tensors_file") or f"tensors/step_{s['step']:03d}.npz"
                name = Path(fname).name.removesuffix(".npz")
                children.append({
                    "kind": "step",
                    "name": name,
                    "label": f"step {s['step']:>3} {s.get('kind','')} → {s.get('generated_token') or ''}",
                    "step": s["step"],
                    "has_attention": True,  # steps always capture attention if enabled
                })
            tree["children"].append({"type": "tensors", "children": children})

        return tree

    # ----------------------------------------------------------------- npz meta

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/meta")
    def get_npz_meta(run_id: str, kind: Kind, name: str) -> dict:
        path = _resolve(registry, run_id, kind, name)
        inv = npz_inventory(path)
        inv["layers"] = compute._discover_layers(inv["keys"])
        inv["has_attention"] = any(k.endswith("/attn_weights") for k in inv["keys"])
        inv["has_qk"] = any(k.endswith("/q") for k in inv["keys"]) and any(
            k.endswith("/k") for k in inv["keys"]
        )
        inv["has_qkv_last"] = "qkv_last" in inv["keys"]
        if inv["has_qkv_last"] and not inv["layers"]:
            # snapshot/reference with only qkv_last: derive layer count from its first axis
            shape = inv["shapes"].get("qkv_last") or []
            if len(shape) == 2:
                inv["layers"] = list(range(shape[0]))
        return inv

    # ----------------------------------------------------------------- attention & qk

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/attention.png")
    def attn_png(
        run_id: str,
        kind: Kind,
        name: str,
        layer: str = Query("all"),
        cmap: str = Query("viridis"),
    ) -> Response:
        tensors = load_npz(_resolve(registry, run_id, kind, name))
        if not compute.has_attention(tensors):
            raise HTTPException(400, "this npz has no attention weights")
        mat = _stacked_or_single(tensors, layer, compute.attention_matrix)
        png = matrix_to_png(mat, cmap=cmap)
        return Response(content=png, media_type="image/png")

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/qk.png")
    def qk_png(
        run_id: str,
        kind: Kind,
        name: str,
        layer: str = Query("all"),
        cmap: str = Query("RdBu_r"),
    ) -> Response:
        tensors = load_npz(_resolve(registry, run_id, kind, name))
        if not compute.has_qk(tensors):
            raise HTTPException(400, "this npz has no q/k tensors")
        mat = _stacked_or_single(tensors, layer, compute.qk_scores)
        png = matrix_to_png(mat, cmap=cmap, signed=True)
        return Response(content=png, media_type="image/png")

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/qkv.png")
    def qkv_png(
        run_id: str,
        kind: Kind,
        name: str,
        cmap: str = Query("RdBu_r"),
    ) -> Response:
        tensors = load_npz(_resolve(registry, run_id, kind, name))
        if not compute.has_qkv_last(tensors):
            raise HTTPException(400, "this npz has no qkv_last tensor")
        mat = compute.qkv_last_matrix(tensors)
        png = matrix_to_png(mat, cmap=cmap, signed=True)
        return Response(content=png, media_type="image/png")

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/entropy")
    def entropy(run_id: str, kind: Kind, name: str) -> dict:
        tensors = load_npz(_resolve(registry, run_id, kind, name))
        if not compute.has_attention(tensors):
            raise HTTPException(400, "this npz has no attention weights")
        vals = compute.attention_entropy(tensors)
        return {"layers": compute.layers_available(tensors), "entropy": vals}

    # ----------------------------------------------------------------- compare

    @app.post("/api/compare/metrics")
    def compare_metrics(req: MetricsRequest = Body(...)) -> dict:
        if len(req.refs) < 2:
            raise HTTPException(400, "need at least 2 refs to compare")
        loaded = [load_npz(_resolve(registry, r.run_id, r.kind, r.name)) for r in req.refs]
        return compute.multi_metrics(loaded, req.sources)

    @app.post("/api/compare/advanced")
    def compare_advanced(req: AdvancedMetricsRequest = Body(...)) -> dict:
        if len(req.refs) < 2:
            raise HTTPException(400, "need at least 2 refs")
        loaded = [load_npz(_resolve(registry, r.run_id, r.kind, r.name)) for r in req.refs]
        return {
            "svd": compute.svd_analysis(loaded, req.sources),
            "cka": compute.cka_analysis(loaded, req.sources),
            "pca": compute.pca_analysis(loaded, req.sources),
        }

    @app.post("/api/compare/heatmap.png")
    def compare_heatmap(req: HeatmapPairRequest = Body(...)) -> Response:
        if req.source == "attention":
            extract = compute.attention_matrix
            need = compute.has_attention
            cmap = "viridis"
            signed = False
        elif req.source == "qk":
            extract = compute.qk_scores
            need = compute.has_qk
            cmap = "RdBu_r"
            signed = True
        else:  # qkv — full stacked (L, D); layer selector is ignored.
            extract = None
            need = compute.has_qkv_last
            cmap = "RdBu_r"
            signed = True

        derivative = req.variant not in ("a", "b")

        def _extract_full(tensors):
            if req.source == "qkv":
                return compute.qkv_last_matrix(tensors)
            return _stacked_or_single(tensors, req.layer, extract)

        # Only load the side(s) we actually need.
        ma = mb = None
        if req.variant == "a" or derivative:
            ta = load_npz(_resolve(registry, req.a.run_id, req.a.kind, req.a.name))
            if not need(ta):
                raise HTTPException(400, f"npz A lacks {req.source} data")
            ma = _extract_full(ta)
        if req.variant == "b" or derivative:
            tb = load_npz(_resolve(registry, req.b.run_id, req.b.kind, req.b.name))
            if not need(tb):
                raise HTTPException(400, f"npz B lacks {req.source} data")
            mb = _extract_full(tb)

        if req.variant == "a":
            out = ma
        elif req.variant == "b":
            out = mb
        else:
            if ma.shape != mb.shape:
                raise HTTPException(
                    400,
                    f"shape mismatch for diff variants: A={ma.shape} vs B={mb.shape} "
                    f"(pairing steps with different KV-cache lengths is not comparable)",
                )
            diffs = compute.diff_matrices(ma, mb)
            out = diffs[req.variant]
            if req.variant in ("abs_diff", "sq_err"):
                signed = False
                cmap = "magma"
            elif req.variant == "ratio":
                signed = True
                cmap = "RdBu_r"
        png = matrix_to_png(out, cmap=cmap, signed=signed)
        return Response(content=png, media_type="image/png")

    return app


# ----------------------------------------------------------------- helpers


def _resolve(registry: RunRegistry, run_id: str, kind: str, name: str) -> Path:
    try:
        return resolve_npz(registry, run_id, kind, name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


def _stacked_or_single(tensors, layer: str, extract_fn):
    import numpy as np
    layers = compute.layers_available(tensors)
    if layer == "all":
        mats = [extract_fn(tensors, li) for li in layers]
        return np.vstack(mats)
    try:
        li = int(layer)
    except ValueError as e:
        raise HTTPException(400, f"invalid layer: {layer}") from e
    if li not in layers:
        raise HTTPException(404, f"layer {li} not captured")
    return extract_fn(tensors, li)
