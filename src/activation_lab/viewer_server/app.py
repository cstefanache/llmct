"""FastAPI app for the Research report viewer.

Routes:
  GET  /api/runs                                   → list runs
  GET  /api/runs/{run_id}                          → run.json
  GET  /api/runs/{run_id}/tree                     → sidebar tree
  GET  /api/runs/{run_id}/scenario.yaml            → reconstruct scenario YAML from run.json
  GET  /api/runs/{run_id}/npz/{kind}/{name}/meta   → keys+shapes for a single npz
  GET  /api/runs/{run_id}/npz/{kind}/{name}/attention.png?layer=all|N   → attn-softmax
  GET  /api/runs/{run_id}/npz/{kind}/{name}/qk.png?layer=all|N          → raw q·k/√dk
  GET  /api/runs/{run_id}/npz/{kind}/{name}/qkv.png                     → stacked ⟨q,k⟩·v (L, D)
  GET  /api/runs/{run_id}/npz/{kind}/{name}/entropy                     → per-layer entropy
  POST /api/compare/metrics                        → per-layer cos/MAE/STD/Jaccard
  POST /api/compare/heatmap.png                    → pair heatmap (side/diff/sq/hadamard/ratio)
  GET  /api/scenarios                              → list scenario YAML files
  GET  /api/scenarios/file?path=<rel>             → raw YAML text
  POST /api/scenarios/validate                     → validate YAML against Scenario model
  POST /api/scenarios/save                         → write YAML to scenarios/
  GET  /api/models/local                           → locally cached HF models
  POST /api/runs/launch                            → launch a scenario as a subprocess
  GET  /api/jobs/{job_id}                          → job status + log tail
"""
from __future__ import annotations

import base64
import json as json_mod
import os
from pathlib import Path
from typing import Literal

import yaml
from fastapi import Body, FastAPI, HTTPException, Query, Response  # Body kept for other routes
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from ..scenario import Scenario
from . import compute, jobs
from . import ollama as ollama_mod
from .loader import RunRegistry, load_npz, npz_inventory, resolve_npz
from .render import matrix_to_png


Kind = Literal["snapshot", "reference", "step"]


class NpzRef(BaseModel):
    run_id: str
    kind: Kind
    name: str


class ValidateRequest(BaseModel):
    yaml: str


class SaveRequest(BaseModel):
    path: str
    yaml: str
    overwrite: bool = False


class LaunchRequest(BaseModel):
    path: str


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


class ReportRequest(BaseModel):
    kind: Literal["run", "npz", "pair", "multi"]
    run_id: str | None = None
    ref: NpzRef | None = None
    a: NpzRef | None = None
    b: NpzRef | None = None
    refs: list[NpzRef] | None = None
    sources: list[str] = []


def create_app(runs_dir: Path | None = None) -> FastAPI:
    runs_dir = runs_dir or Path(os.environ.get("ACTIVATION_LAB_RUNS_DIR", "runs")).resolve()
    scenarios_dir = runs_dir.parent / "scenarios"
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

    @app.get("/api/runs/{run_id}/npz/{kind}/{name}/convergence")
    def get_convergence(
        run_id: str, kind: Kind, name: str,
        source: str = Query("hidden_out"),
    ) -> dict:
        tensors = load_npz(_resolve(registry, run_id, kind, name))
        return compute.residual_convergence(tensors, source=source)

    @app.get("/api/runs/{run_id}/logit_stats")
    def get_logit_stats(run_id: str) -> dict:
        steps = registry.steps_index(run_id)
        if not steps:
            raise HTTPException(404, "no steps found for this run")
        result: dict = {"steps": [], "entropy": [], "effective_vocab": [], "top1_prob": []}
        for s in steps:
            ls = s.get("logit_stats")
            if ls:
                result["steps"].append(s["step"])
                result["entropy"].append(ls.get("entropy"))
                result["effective_vocab"].append(ls.get("effective_vocab"))
                result["top1_prob"].append(ls.get("top1_prob"))
        if not result["steps"]:
            raise HTTPException(404, "no logit_stats in steps (run predates this feature)")
        return result

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

    @app.post("/api/compare/group")
    def compare_group(req: AdvancedMetricsRequest = Body(...)) -> dict:
        if len(req.refs) < 2:
            raise HTTPException(400, "need at least 2 refs")
        loaded = [load_npz(_resolve(registry, r.run_id, r.kind, r.name)) for r in req.refs]
        return compute.group_analysis(loaded, req.sources)

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

    # ----------------------------------------------------------------- scenarios

    def _safe_scenario_path(rel: str) -> Path:
        """Resolve a relative path under scenarios_dir, rejecting traversals."""
        try:
            p = (scenarios_dir / rel).resolve()
        except Exception as exc:
            raise HTTPException(400, f"invalid path: {rel}") from exc
        if not p.is_relative_to(scenarios_dir.resolve()):
            raise HTTPException(400, "path traversal not allowed")
        return p

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        if not scenarios_dir.exists():
            return []
        result = []
        for p in sorted(scenarios_dir.rglob("*.yaml")):
            rel = str(p.relative_to(scenarios_dir))
            result.append({"path": rel, "name": p.stem, "mtime": p.stat().st_mtime})
        return result

    @app.get("/api/scenarios/file")
    def get_scenario_file(path: str = Query(...)) -> dict:
        p = _safe_scenario_path(path)
        if not p.exists():
            raise HTTPException(404, f"scenario not found: {path}")
        return {"yaml": p.read_text()}

    @app.get("/api/runs/{run_id}/scenario.yaml")
    def get_run_scenario_yaml(run_id: str) -> Response:
        try:
            data = registry.run_json(run_id)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        scenario_dict = data.get("scenario", {})
        text = yaml.safe_dump(scenario_dict, sort_keys=False, allow_unicode=True)
        return Response(content=text, media_type="text/yaml")

    @app.post("/api/scenarios/validate")
    def validate_scenario(req: ValidateRequest = Body(...)) -> dict:
        try:
            raw = yaml.safe_load(req.yaml)
            Scenario.model_validate(raw)
            return {"ok": True, "errors": []}
        except yaml.YAMLError as e:
            return {"ok": False, "errors": [{"loc": [], "msg": f"YAML parse error: {e}"}]}
        except ValidationError as e:
            errs = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
            return {"ok": False, "errors": errs}

    @app.post("/api/scenarios/save")
    def save_scenario(req: SaveRequest = Body(...)) -> dict:
        p = _safe_scenario_path(req.path)
        if p.exists() and not req.overwrite:
            raise HTTPException(409, f"file already exists: {req.path}")
        # Validate before writing
        try:
            raw = yaml.safe_load(req.yaml)
            Scenario.model_validate(raw)
        except yaml.YAMLError as e:
            raise HTTPException(400, f"YAML parse error: {e}")
        except ValidationError as e:
            raise HTTPException(422, str(e))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.yaml)
        return {"saved": req.path}

    # ----------------------------------------------------------------- local models

    @app.get("/api/models/local")
    def get_local_models() -> list[dict]:
        return ollama_mod.list_local_models()

    # ----------------------------------------------------------------- launch / jobs

    @app.post("/api/runs/launch")
    def launch_run(req: LaunchRequest = Body(...)) -> dict:
        p = _safe_scenario_path(req.path)
        if not p.exists():
            raise HTTPException(404, f"scenario file not found: {req.path}")
        try:
            raw = yaml.safe_load(p.read_text())
            scenario = Scenario.model_validate(raw)
        except Exception as e:
            raise HTTPException(400, f"invalid scenario: {e}")
        info = jobs.launch_scenario(p, scenario.name)
        return {"job_id": info.job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        info = jobs.get(job_id)
        if info is None:
            raise HTTPException(404, "job not found")
        run_id = None
        if runs_dir.exists():
            candidates = [
                p for p in runs_dir.iterdir()
                if p.is_dir()
                and p.name.startswith(info.expected_run_prefix)
                and (p / "run.json").exists()
            ]
            if candidates:
                run_id = max(candidates, key=lambda p: p.stat().st_mtime).name
        return {
            "status": info.status,
            "returncode": info.returncode,
            "started_at": info.started_at,
            "finished_at": info.finished_at,
            "log_tail": info.log_tail(),
            "run_id": run_id,
        }

    # ----------------------------------------------------------------- report
    # Parameters are base64-encoded JSON in the query string so the browser can
    # open the report as a plain GET navigation (new tab, bookmarkable URL).

    @app.get("/api/report/view")
    def view_report(params: str = Query(...)) -> Response:
        from . import report as report_mod
        try:
            req_data = json_mod.loads(base64.b64decode(params.encode()).decode())
        except Exception as exc:
            raise HTTPException(400, f"invalid params encoding: {exc}") from exc

        kind = req_data.get("kind")
        sources = req_data.get("sources", [])
        try:
            if kind == "run":
                html = report_mod.generate_run_report(registry, req_data["run_id"])
            elif kind == "npz":
                html = report_mod.generate_npz_report(registry, req_data["ref"], sources)
            elif kind == "pair":
                html = report_mod.generate_pair_report(
                    registry, req_data["a"], req_data["b"], sources,
                )
            elif kind == "multi":
                html = report_mod.generate_multi_report(registry, req_data["refs"], sources)
            else:
                raise HTTPException(400, f"unknown kind: {kind}")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"report generation failed: {exc}") from exc
        return Response(content=html, media_type="text/html")

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
