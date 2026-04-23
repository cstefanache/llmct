#!/usr/bin/env python3
"""Compare runs against each other (inter-run) and compare initial vs final state within each run (intra-run).

Usage:
    python compare_runs.py [runs_dir]

Outputs:
  - Console: per-pair / per-run summary tables
  - comparison_report.csv       — inter-run summary stats per pair × state × source
  - comparison_cosines.csv      — inter-run per-layer metrics per pair × state × source
  - intra_run_report.csv        — intra-run summary stats per run × source
  - intra_run_cosines.csv       — intra-run per-layer metrics per run × source
"""
from __future__ import annotations

import contextlib
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm

_here = Path(__file__).parent
_project_root = _here if (_here / "runs").exists() else _here.parent
sys.path.insert(0, str(_project_root / "src"))

from activation_lab.view import Run

RESIDUAL_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"]


def initial_state(run: Run, sources: list[str] = RESIDUAL_SOURCES) -> dict[str, np.ndarray]:
    """Load prefill step (step 0), last prompt position → {source: (num_layers, hidden_size)}."""
    with run.load_npz(0) as npz:
        return {
            src: np.stack([
                np.asarray(npz[f"layer_{layer:02d}/{src}"], dtype=np.float32)[0, -1, :]
                for layer in range(run.num_layers)
            ])
            for src in sources
        }


def final_state(run: Run, sources: list[str] = RESIDUAL_SOURCES) -> dict[str, np.ndarray]:
    """Load last generation step, last token position → {source: (num_layers, hidden_size)}."""
    last_step = run.steps_raw[-1]["step"]
    with run.load_npz(last_step) as npz:
        return {
            src: np.stack([
                np.asarray(npz[f"layer_{layer:02d}/{src}"], dtype=np.float32)[0, -1, :]
                for layer in range(run.num_layers)
            ])
            for src in sources
        }


def cosine_per_layer(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    dots = (s1 * s2).sum(axis=-1)
    return dots / (norm(s1, axis=-1) * norm(s2, axis=-1) + 1e-8)


def overlap_per_layer(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Jaccard similarity of top-1% most-active dimensions per layer."""
    n_layers, hidden_size = s1.shape
    k = max(1, hidden_size // 100)
    top1 = np.argsort(np.abs(s1), axis=-1)[:, -k:]
    top2 = np.argsort(np.abs(s2), axis=-1)[:, -k:]
    result = np.zeros(n_layers)
    for layer in range(n_layers):
        a, b = set(top1[layer].tolist()), set(top2[layer].tolist())
        inter = len(a & b)
        union = len(a | b)
        result[layer] = inter / union if union else 1.0
    return result


def _compute_metrics(
    states_a: dict[str, np.ndarray],
    states_b: dict[str, np.ndarray],
    num_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared metric computation given two state dicts. Returns (summary_df, cosines_df)."""
    summary_rows: list[dict] = []
    cosine_rows: list[dict] = []

    for src in RESIDUAL_SOURCES:
        s1, s2 = states_a[src], states_b[src]
        cos_vals = cosine_per_layer(s1, s2)
        mae_vals = np.abs(s1 - s2).mean(axis=-1)
        ovl_vals = overlap_per_layer(s1, s2)

        summary_rows.append({
            "source": src,
            "mean_cosine":       round(float(cos_vals.mean()), 4),
            "std_cosine":        round(float(cos_vals.std()), 4),
            "median_cosine":     round(float(np.median(cos_vals)), 4),
            "min_cosine":        round(float(cos_vals.min()), 4),
            "most_divergent_layer": int(cos_vals.argmin()),
            "mean_MAE":          round(float(mae_vals.mean()), 4),
            "std_MAE":           round(float(mae_vals.std()), 4),
            "median_MAE":        round(float(np.median(mae_vals)), 4),
            "max_MAE_layer":     int(mae_vals.argmax()),
            "mean_overlap":      round(float(ovl_vals.mean()), 4),
            "std_overlap":       round(float(ovl_vals.std()), 4),
            "median_overlap":    round(float(np.median(ovl_vals)), 4),
            "min_overlap":       round(float(ovl_vals.min()), 4),
            "min_overlap_layer": int(ovl_vals.argmin()),
        })

        for layer in range(num_layers):
            cosine_rows.append({
                "source":  src,
                "layer":   layer,
                "cosine":  round(float(cos_vals[layer]), 4),
                "MAE":     round(float(mae_vals[layer]), 4),
                "overlap": round(float(ovl_vals[layer]), 4),
            })

    summary_df = pd.DataFrame(summary_rows).set_index("source")
    cosines_df = pd.DataFrame(cosine_rows)
    return summary_df, cosines_df


def compare_pair(
    run_a: Run, run_b: Run, state_fn=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, cosines_df) for one inter-run pair and state."""
    if state_fn is None:
        state_fn = initial_state
    return _compute_metrics(state_fn(run_a), state_fn(run_b), run_a.num_layers)


def compare_run_states(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare initial state vs final state within the same run."""
    return _compute_metrics(initial_state(run), final_state(run), run.num_layers)


def get_step_topk(run: Run) -> list[dict] | None:
    """Read pre-captured top-k next-token predictions from steps.json.

    Each step already has a `top_k` list with {id, token, logprob} entries captured during
    generation (controlled by CaptureConfig.top_k_probs). Returns None if not captured.
    """
    results = []
    for step_info in run.steps_raw:
        raw_top = step_info.get("top_k") or []
        if not raw_top:
            return None
        entries = [
            {"token": t["token"], "prob": round(float(np.exp(t["logprob"])), 5)}
            for t in raw_top
        ]
        results.append({
            "step":            step_info["step"],
            "kind":            step_info.get("kind", ""),
            "generated_token": step_info.get("generated_token", "") or "",
            "top":             entries,
        })
    return results or None


def run_label(run: Run, short: bool = False) -> str:
    info = run.summary()
    generated = (info.get("generated", "") or "").replace("\n", " ").strip()
    if short:
        return generated[:40] or run.path.name
    return f"{run.path.name}  →  '{generated[:50]}'"


def discover_runs(runs_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        with contextlib.suppress(FileNotFoundError, KeyError):
            runs.append(Run(p))
    return runs


def generate_html_report(
    report_df: pd.DataFrame,
    cosines_df: pd.DataFrame,
    intra_report_df: pd.DataFrame,
    intra_cosines_df: pd.DataFrame,
    run_topk: dict[str, list[dict] | None],
    out_path: Path,
) -> None:
    """Write a self-contained HTML report from the comparison DataFrames."""
    import html as html_module
    import json

    SOURCE_COLORS = {
        "hidden_in":    "#3b82f6",
        "hidden_out":   "#10b981",
        "attn_out":     "#f59e0b",
        "mlp_down_out": "#ef4444",
    }

    STATE_LABELS = {
        "initial": "Initial state — prefill, last prompt token",
        "final":   "Final state — last generation step, last token",
    }

    def cosine_bg(val: float) -> str:
        t = max(0.0, min(1.0, (float(val) - 0.5) / 0.5))
        r = int(220 * (1 - t))
        g = int(180 * t)
        return f"rgba({r},{g},60,0.35)"

    def _val_color(val: float, lo: float, hi: float, reverse: bool = False) -> str:
        t = max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9)))
        if reverse:
            t = 1.0 - t
        if t < 0.5:
            r, g = 220, int(180 * t * 2)
        else:
            r, g = int(220 * (1.0 - (t - 0.5) * 2)), 160
        return f"rgba({r},{g},50,0.8)"

    def _heatmap_table(
        matrix: list,
        row_labels: list,
        col_labels: list,
        lo: float,
        hi: float,
        reverse: bool = False,
        fmt: str = ".3f",
        title: str = "",
        clickable=None,
    ) -> str:
        def _lbl(s: str) -> str:
            e = html_module.escape(s)
            if clickable and s in clickable:
                return f'<span class="run-lnk" data-run="{e}">{e}</span>'
            return e

        col_hdrs = "".join(f"<th class='hm-col-hdr'>{_lbl(str(c))}</th>" for c in col_labels)
        rows_html = ""
        for r_lbl, row in zip(row_labels, matrix, strict=False):
            cells = ""
            for val in row:
                if val is None:
                    cells += "<td class='hm-na'></td>"
                else:
                    bg = _val_color(val, lo, hi, reverse)
                    display = f"{val:{fmt}}"
                    cells += f'<td class="hm-cell" style="background:{bg}" title="{val}">{display}</td>'
            rows_html += f"<tr><th class='hm-row-hdr'>{_lbl(str(r_lbl))}</th>{cells}</tr>"
        ttl = f"<div class='hm-title'>{html_module.escape(title)}</div>" if title else ""
        return (
            f"{ttl}<div class='hm-wrap'><table class='hm'>"
            f"<thead><tr><th></th>{col_hdrs}</tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    pairs = report_df["pair"].unique().tolist()

    # ── run×run legend ────────────────────────────────────────────────────────
    runs_ordered: list[str] = []
    run_gen: dict[str, str] = {}
    for _, row in report_df.iterrows():
        for key, gen_key in (("run_a", "generated_a"), ("run_b", "generated_b")):
            if row[key] not in run_gen:
                runs_ordered.append(row[key])
                run_gen[row[key]] = str(row[gen_key])[:20]

    n_runs  = len(runs_ordered)
    run_idx = {r: i for i, r in enumerate(runs_ordered)}
    run_short = [f"R{i + 1}" for i in range(n_runs)]

    run_legend_html = "".join(
        f"<tr><td><b class='run-lnk' data-run='R{i + 1}'>R{i + 1}</b></td>"
        f"<td><code>{html_module.escape(runs_ordered[i])}</code></td>"
        f"<td><em>{html_module.escape(run_gen[runs_ordered[i]])}</em></td></tr>"
        for i in range(n_runs)
    )
    run_row_labels = run_short

    _run_detail: dict = {}
    for _, row in report_df.iterrows():
        ra, rb = row["run_a"], row["run_b"]
        if ra not in _run_detail:
            _run_detail[ra] = {
                "prompt":    str(row.get("prompt_a", "") or ""),
                "generated": str(row.get("generated_a", "") or ""),
            }
        if rb not in _run_detail:
            _run_detail[rb] = {
                "prompt":    str(row.get("prompt_b", "") or ""),
                "generated": str(row.get("generated_b", "") or ""),
            }
    runs_js_data = {
        f"R{i + 1}": {
            "name":      runs_ordered[i],
            "prompt":    _run_detail.get(runs_ordered[i], {}).get("prompt", ""),
            "generated": _run_detail.get(runs_ordered[i], {}).get("generated", ""),
        }
        for i in range(n_runs)
    }
    runs_js_data_json = json.dumps(runs_js_data)

    RR_METRICS = [
        ("mean_cosine",   False, ".3f", "Mean cosine"),
        ("min_cosine",    False, ".3f", "Min cosine"),
        ("mean_MAE",      True,  ".3f", "Mean MAE"),
        ("mean_overlap",  False, ".3f", "Mean overlap"),
    ]

    # ── shared overview/aggregate/heatmap builders ────────────────────────────
    overview_header = (
        "".join(f"<th>{s}</th>" for s in RESIDUAL_SOURCES)
        + "<th>Mean</th><th>Std</th><th>Median</th>"
    )

    def _build_overview_rows(df_s: pd.DataFrame, index_col: str) -> str:
        if df_s.empty:
            return "<tr><td colspan='8'>No data</td></tr>"
        pivot = (
            df_s.pivot_table(index=index_col, columns="source", values="mean_cosine")
            .reindex(columns=RESIDUAL_SOURCES)
        )
        rows = ""
        for label, row in pivot.iterrows():
            vals = [row[s] for s in RESIDUAL_SOURCES]
            row_mean   = float(np.mean(vals))
            row_std    = float(np.std(vals))
            row_median = float(np.median(vals))
            source_cells = "".join(
                f'<td style="background:{cosine_bg(row[s])}">{row[s]:.4f}</td>'
                for s in RESIDUAL_SOURCES
            )
            stat_cells = (
                f'<td style="background:{cosine_bg(row_mean)}">{row_mean:.4f}</td>'
                f'<td class="stat-cell">{row_std:.4f}</td>'
                f'<td style="background:{cosine_bg(row_median)}">{row_median:.4f}</td>'
            )
            rows += f"<tr><td class='pair-cell'>{label}</td>{source_cells}{stat_cells}</tr>"
        return rows

    def _build_agg_rows(df_s: pd.DataFrame) -> str:
        agg_rows_html = ""
        for src in RESIDUAL_SOURCES:
            src_df = df_s[df_s["source"] == src]
            def _agg(col: str, _df=src_df) -> str:
                s = _df[col].dropna()
                if s.empty:
                    return "—"
                return f"{s.mean():.4f} ± {s.std():.4f}"
            agg_rows_html += (
                f"<tr><td>{src}</td>"
                f"<td>{_agg('mean_cosine')}</td><td>{_agg('std_cosine')}</td><td>{_agg('median_cosine')}</td>"
                f"<td>{_agg('mean_MAE')}</td><td>{_agg('std_MAE')}</td><td>{_agg('median_MAE')}</td>"
                f"<td>{_agg('mean_overlap')}</td><td>{_agg('std_overlap')}</td><td>{_agg('median_overlap')}</td>"
                f"</tr>"
            )
        return agg_rows_html

    def _build_rr_sections(df_s: pd.DataFrame) -> str:
        rr_html = ""
        for src in RESIDUAL_SOURCES:
            src_df  = df_s[df_s["source"] == src]
            hms_html = ""
            for col, reverse, fmt, label in RR_METRICS:
                mat = [[None] * n_runs for _ in range(n_runs)]
                for _, row in src_df.iterrows():
                    i = run_idx.get(row["run_a"])
                    j = run_idx.get(row["run_b"])
                    if i is not None and j is not None:
                        v = float(row[col])
                        mat[i][j] = v
                        mat[j][i] = v
                flat = [mat[i][j] for i in range(n_runs) for j in range(n_runs) if mat[i][j] is not None]
                lo = min(flat) if flat else 0.0
                hi = max(flat) if flat else 1.0
                if col in ("mean_cosine", "min_cosine"):
                    lo, hi = min(lo, 0.5), 1.0
                hms_html += f"<div class='hm-item'>{_heatmap_table(mat, run_row_labels, run_short, lo, hi, reverse, fmt, label, clickable=set(run_short))}</div>"
            rr_html += f"<h3 class='src-heading'>{src}</h3><div class='hm-grid'>{hms_html}</div>"
        return rr_html

    pair_short_labels = []
    for pair in pairs:
        pr = report_df[report_df["pair"] == pair]
        ga = str(pr["generated_a"].iloc[0])[:16]
        gb = str(pr["generated_b"].iloc[0])[:16]
        pair_short_labels.append(f"{ga} / {gb}")

    def _build_layer_sections(df_cosines: pd.DataFrame, row_keys: list[str], row_labels: list[str], key_col: str) -> str:
        layer_html = ""
        for src in RESIDUAL_SOURCES:
            src_cos    = df_cosines[df_cosines["source"] == src]
            all_layers = sorted(src_cos["layer"].unique().tolist())
            n_layers   = len(all_layers)

            cos_mat, mae_mat, ovl_mat = [], [], []
            for key in row_keys:
                pc = src_cos[src_cos[key_col] == key].sort_values("layer")
                if len(pc) == n_layers:
                    cos_mat.append(pc["cosine"].tolist())
                    mae_mat.append(pc["MAE"].tolist())
                    ovl_mat.append(pc["overlap"].tolist() if "overlap" in pc.columns else [None] * n_layers)
                else:
                    cos_mat.append([None] * n_layers)
                    mae_mat.append([None] * n_layers)
                    ovl_mat.append([None] * n_layers)

            flat_cos = [v for row in cos_mat for v in row if v is not None]
            flat_mae = [v for row in mae_mat for v in row if v is not None]
            flat_ovl = [v for row in ovl_mat for v in row if v is not None]
            cos_lo  = min(flat_cos) if flat_cos else 0.0
            mae_hi  = max(flat_mae) if flat_mae else 1.0
            ovl_lo  = min(flat_ovl) if flat_ovl else 0.0

            layer_labels = [str(li) for li in all_layers]
            hms_html = (
                f"<div class='hm-item hm-full'>"
                f"{_heatmap_table(cos_mat, row_labels, layer_labels, cos_lo, 1.0, False, '.3f', 'Cosine per layer')}</div>"
                f"<div class='hm-item hm-full'>"
                f"{_heatmap_table(mae_mat, row_labels, layer_labels, 0.0, mae_hi, True, '.3f', 'MAE per layer')}</div>"
                f"<div class='hm-item hm-full'>"
                f"{_heatmap_table(ovl_mat, row_labels, layer_labels, ovl_lo, 1.0, False, '.3f', 'Top-1% overlap per layer')}</div>"
            )
            layer_html += f"<h3 class='src-heading'>{src}</h3><div class='hm-grid hm-grid-2'>{hms_html}</div>"
        return layer_html

    # ── inter-run sections ────────────────────────────────────────────────────
    overview_rows_initial = _build_overview_rows(report_df[report_df["state"] == "initial"], "pair")
    overview_rows_final   = _build_overview_rows(report_df[report_df["state"] == "final"],   "pair")
    agg_rows_initial      = _build_agg_rows(report_df[report_df["state"] == "initial"])
    agg_rows_final        = _build_agg_rows(report_df[report_df["state"] == "final"])
    rr_sections_initial   = _build_rr_sections(report_df[report_df["state"] == "initial"])
    rr_sections_final     = _build_rr_sections(report_df[report_df["state"] == "final"])
    layer_sections_initial = _build_layer_sections(
        cosines_df[cosines_df["state"] == "initial"], pairs, pair_short_labels, "pair"
    )
    layer_sections_final = _build_layer_sections(
        cosines_df[cosines_df["state"] == "final"], pairs, pair_short_labels, "pair"
    )

    # ── intra-run sections ────────────────────────────────────────────────────
    intra_runs = intra_report_df["run"].unique().tolist() if not intra_report_df.empty else []
    intra_run_short_labels = [r[:32] for r in intra_runs]

    overview_rows_intra  = _build_overview_rows(intra_report_df, "run")
    agg_rows_intra       = _build_agg_rows(intra_report_df)
    layer_sections_intra = _build_layer_sections(
        intra_cosines_df, intra_runs, intra_run_short_labels, "run"
    )

    # ── shared chart/table helpers ────────────────────────────────────────────
    def _chart_datasets(pc_s: pd.DataFrame, metric: str) -> list:
        out = []
        for src in RESIDUAL_SOURCES:
            vals = pc_s[pc_s["source"] == src].sort_values("layer")[metric].tolist()
            c = SOURCE_COLORS.get(src, "#888")
            out.append({
                "label": src, "data": vals,
                "borderColor": c, "backgroundColor": c + "22",
                "tension": 0.3, "pointRadius": 3,
            })
        return out

    def _summary_table_rows(pr_s: pd.DataFrame) -> str:
        def _fmt(r, col):
            v = r.get(col, float("nan"))
            return f"{float(v):.4f}" if v == v else "—"
        return "".join(
            f'<tr style="background:{cosine_bg(r["mean_cosine"])}">'
            f'<td>{r["source"]}</td>'
            f'<td>{r["mean_cosine"]:.4f}</td><td>{_fmt(r, "std_cosine")}</td>'
            f'<td>{_fmt(r, "median_cosine")}</td><td>{r["min_cosine"]:.4f}</td>'
            f'<td>{int(r["most_divergent_layer"])}</td>'
            f'<td>{r["mean_MAE"]:.4f}</td><td>{_fmt(r, "std_MAE")}</td>'
            f'<td>{_fmt(r, "median_MAE")}</td><td>{int(r["max_MAE_layer"])}</td>'
            f'<td>{_fmt(r, "mean_overlap")}</td><td>{_fmt(r, "std_overlap")}</td>'
            f'<td>{_fmt(r, "median_overlap")}</td>'
            f'<td>{int(r["min_overlap_layer"]) if "min_overlap_layer" in r and r["min_overlap_layer"] == r["min_overlap_layer"] else "—"}</td>'
            f'</tr>'
            for _, r in pr_s.iterrows()
        )

    _SUMMARY_THEAD = """
      <thead>
        <tr>
          <th rowspan="2">Source</th>
          <th colspan="4">Cosine similarity</th>
          <th colspan="4">MAE</th>
          <th colspan="4">Top-1% overlap (Jaccard)</th>
        </tr>
        <tr>
          <th>Mean</th><th>Std</th><th>Median</th><th>Min (layer)</th>
          <th>Mean</th><th>Std</th><th>Median</th><th>Max (layer)</th>
          <th>Mean</th><th>Std</th><th>Median</th><th>Min (layer)</th>
        </tr>
      </thead>"""

    def _chart_block(pid: str, layers: list, pc_s: pd.DataFrame) -> tuple[str, str]:
        cos_json = json.dumps({"labels": layers, "datasets": _chart_datasets(pc_s, "cosine")})
        mae_json = json.dumps({"labels": layers, "datasets": _chart_datasets(pc_s, "MAE")})
        ovl_json = json.dumps({"labels": layers, "datasets": _chart_datasets(pc_s, "overlap")})
        html = f"""
            <div class="chart-wrap"><h4>Cosine similarity per layer</h4><canvas id="cos_{pid}"></canvas></div>
            <div class="chart-wrap"><h4>MAE per layer</h4><canvas id="mae_{pid}"></canvas></div>
            <div class="chart-wrap"><h4>Top-1% activation overlap (Jaccard) per layer</h4><canvas id="ovl_{pid}"></canvas></div>"""
        js = f"""
          new Chart(document.getElementById('cos_{pid}'), {{
            type: 'line', data: {cos_json},
            options: {{ plugins: {{ legend: {{ position: 'top' }} }},
              scales: {{ x: {{ title: {{ display: true, text: 'Layer' }} }},
                         y: {{ title: {{ display: true, text: 'Cosine' }}, min: 0, max: 1 }} }} }}
          }});
          new Chart(document.getElementById('mae_{pid}'), {{
            type: 'line', data: {mae_json},
            options: {{ plugins: {{ legend: {{ position: 'top' }} }},
              scales: {{ x: {{ title: {{ display: true, text: 'Layer' }} }},
                         y: {{ title: {{ display: true, text: 'MAE' }}, min: 0 }} }} }}
          }});
          new Chart(document.getElementById('ovl_{pid}'), {{
            type: 'line', data: {ovl_json},
            options: {{ plugins: {{ legend: {{ position: 'top' }} }},
              scales: {{ x: {{ title: {{ display: true, text: 'Layer' }} }},
                         y: {{ title: {{ display: true, text: 'Jaccard overlap' }}, min: 0, max: 1 }} }} }}
          }});"""
        return html, js

    # ── inter-run pair cards ──────────────────────────────────────────────────
    pair_sections_html = ""
    for idx, pair in enumerate(pairs):
        pr_all = report_df[report_df["pair"] == pair]
        pc_all = cosines_df[cosines_df["pair"] == pair]

        run_a    = pr_all["run_a"].iloc[0]
        run_b    = pr_all["run_b"].iloc[0]
        gen_a    = pr_all["generated_a"].iloc[0]
        gen_b    = pr_all["generated_b"].iloc[0]
        prompt_a = pr_all["prompt_a"].iloc[0] if "prompt_a" in pr_all.columns else ""
        prompt_b = pr_all["prompt_b"].iloc[0] if "prompt_b" in pr_all.columns else ""

        state_blocks_html = ""
        chart_scripts = ""

        for state_label in ["initial", "final"]:
            pr_s = pr_all[pr_all["state"] == state_label]
            pc_s = pc_all[pc_all["state"] == state_label]
            if pr_s.empty or pc_s.empty:
                continue

            layers = sorted(pc_s["layer"].unique().tolist())
            pid    = f"p{idx}_{state_label[0]}"
            canvases, js = _chart_block(pid, layers, pc_s)
            chart_scripts += js

            state_blocks_html += f"""
          <div class="state-block">
            <h3 class="state-heading">{STATE_LABELS[state_label]}</h3>
            <div class="charts-row">{canvases}</div>
            <table class="summary-table">{_SUMMARY_THEAD}<tbody>{_summary_table_rows(pr_s)}</tbody></table>
          </div>"""

        pair_sections_html += f"""
        <section class="pair-card">
          <h2 class="pair-title">{pair}</h2>
          <dl class="gen-labels">
            <dt>A</dt><dd><code>{run_a}</code> &mdash; <em>{gen_a}</em></dd>
            <dt>B</dt><dd><code>{run_b}</code> &mdash; <em>{gen_b}</em></dd>
          </dl>
          <div class="prompts-row">
            <details class="prompt-block"><summary>Prompt A</summary><pre>{html_module.escape(prompt_a)}</pre></details>
            <details class="prompt-block"><summary>Prompt B</summary><pre>{html_module.escape(prompt_b)}</pre></details>
          </div>
          {state_blocks_html}
        </section>
        <script>{chart_scripts}</script>"""

    # ── intra-run cards ───────────────────────────────────────────────────────
    intra_sections_html = ""
    for idx, run_name in enumerate(intra_runs):
        pr_s = intra_report_df[intra_report_df["run"] == run_name]
        pc_s = intra_cosines_df[intra_cosines_df["run"] == run_name]
        if pr_s.empty or pc_s.empty:
            continue

        generated = str(pr_s["generated"].iloc[0]) if "generated" in pr_s.columns else ""
        prompt    = str(pr_s["prompt"].iloc[0]) if "prompt" in pr_s.columns else ""
        layers    = sorted(pc_s["layer"].unique().tolist())
        pid       = f"ir{idx}"
        canvases, js = _chart_block(pid, layers, pc_s)

        # ── top-k explorer ────────────────────────────────────────────────────
        topk_html = ""
        topk_js   = ""
        topk_data = run_topk.get(run_name)
        if topk_data:
            max_k    = max(len(s["top"]) for s in topk_data)
            topk_json = json.dumps(topk_data)
            eid = f"topk_{idx}"
            topk_html = f"""
          <div class="topk-explorer">
            <div class="topk-controls">
              <span class="topk-label">Top-k</span>
              <input  type="range" id="{eid}_range" min="1" max="{max_k}" value="5">
              <b id="{eid}_val">5</b>
              <span class="topk-avail">/ {max_k} captured per step</span>
            </div>
            <div class="topk-wrap">
              <table class="topk-table" id="{eid}_tbl">
                <thead></thead>
                <tbody></tbody>
              </table>
            </div>
          </div>"""
            topk_js = f"""
          (function() {{
            const data = {topk_json};
            const range = document.getElementById('{eid}_range');
            const valEl = document.getElementById('{eid}_val');
            const thead = document.querySelector('#{eid}_tbl thead');
            const tbody = document.querySelector('#{eid}_tbl tbody');
            function probBg(p) {{
              const t = Math.min(1.0, p * 5);
              const r = Math.round(220 * (1 - t));
              const g = Math.round(50 + 150 * t);
              return 'rgba(' + r + ',' + g + ',50,0.55)';
            }}
            function esc(s) {{
              return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            }}
            // Header: one column per step, fixed regardless of k
            thead.innerHTML = '<tr>'
              + '<th class="rank-col">Rank</th>'
              + data.map(step => '<th class="step-col-hdr"><div>'
                  + '<span class="step-num-hdr">Step ' + step.step + '</span>'
                  + '<span class="step-kind-hdr ' + step.kind + '">' + step.kind + '</span>'
                  + '<code class="step-gen-hdr">' + (step.generated_token ? esc(step.generated_token) : '—') + '</code>'
                  + '</th>').join('')
              + '</div></tr>';
            // Body: one row per rank, one cell per step
            function render(k) {{
              const rows = [];
              for (let rank = 0; rank < k; rank++) {{
                let cells = '<th class="rank-num">' + (rank + 1) + '</th>';
                data.forEach(step => {{
                  const t = step.top[rank];
                  if (!t) {{ cells += '<td class="topk-na">—</td>'; return; }}
                  const pct = (t.prob * 100).toFixed(1);
                  cells += '<td class="topk-cell" style="background:' + probBg(t.prob) + '">'
                    + '<span class="tok-text">' + esc(t.token) + '</span>'
                    + '<span class="tok-prob">' + pct + '%</span>'
                    + '</td>';
                }});
                rows.push('<tr>' + cells + '</tr>');
              }}
              tbody.innerHTML = rows.join('');
            }}
            range.addEventListener('input', () => {{ valEl.textContent = range.value; render(+range.value); }});
            render(5);
          }})();"""
        else:
            topk_html = '<p class="topk-missing">Top-k predictions not captured for this run. Set <code>capture.top_k_probs &gt; 0</code> and re-run.</p>'

        intra_sections_html += f"""
        <section class="pair-card">
          <h2 class="pair-title">{run_name}</h2>
          <dl class="gen-labels">
            <dt>output</dt><dd><em>{html_module.escape(generated[:80])}</em></dd>
          </dl>
          <div class="prompts-row">
            <details class="prompt-block"><summary>Prompt</summary><pre>{html_module.escape(prompt)}</pre></details>
          </div>
          <div class="state-block">
            <h3 class="state-heading">Initial → Final (same run)</h3>
            <div class="charts-row">{canvases}</div>
            <table class="summary-table">{_SUMMARY_THEAD}<tbody>{_summary_table_rows(pr_s)}</tbody></table>
          </div>
          <div class="state-block">
            <h3 class="state-heading">Generation step top-k explorer</h3>
            {topk_html}
          </div>
        </section>
        <script>{js}{topk_js}</script>"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Activation Comparison Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem 2rem; background: #f8fafc; color: #1e293b; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
    .meta {{ color: #64748b; font-size: 0.85rem; margin-bottom: 2rem; }}
    h2.section-heading {{ font-size: 1.2rem; margin: 2rem 0 0.75rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.4rem; }}
    h3.state-subheading {{ font-size: 1rem; margin: 1.5rem 0 0.5rem; color: #475569; border-left: 3px solid #cbd5e1; padding-left: 0.5rem; }}
    .overview-wrap {{ overflow-x: auto; margin-bottom: 2rem; }}
    section.pair-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
      padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }}
    h2.pair-title {{ font-size: 0.95rem; font-family: monospace; margin: 0 0 0.75rem; word-break: break-all; color: #334155; }}
    dl.gen-labels {{ display: grid; grid-template-columns: 3.5rem 1fr; gap: 0.2rem 0.5rem; font-size: 0.82rem; margin: 0 0 1rem; }}
    dl dt {{ font-weight: 700; color: #64748b; }}
    dl dd {{ margin: 0; }}
    .prompts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem; }}
    .prompt-block {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.78rem; }}
    .prompt-block summary {{ cursor: pointer; font-weight: 600; color: #475569; user-select: none; }}
    .prompt-block pre {{ margin: 0.5rem 0 0; white-space: pre-wrap; word-break: break-word; color: #334155; font-size: 0.75rem; max-height: 180px; overflow-y: auto; }}
    .state-block {{ margin-top: 1.5rem; padding-top: 1rem; border-top: 1px dashed #e2e8f0; }}
    .state-block:first-child {{ margin-top: 0; padding-top: 0; border-top: none; }}
    h3.state-heading {{
      font-size: 0.85rem; font-weight: 700; color: #fff;
      background: #475569; border-radius: 4px; padding: 0.25rem 0.65rem;
      margin: 0 0 1rem; display: inline-block;
    }}
    .state-block:last-child h3.state-heading {{ background: #0f766e; }}
    .charts-row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 1.25rem; }}
    .chart-wrap h4 {{ font-size: 0.82rem; font-weight: 600; margin: 0 0 0.4rem; color: #475569; text-align: center; }}
    canvas {{ max-height: 240px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 0.35rem 0.6rem; border: 1px solid #e2e8f0; text-align: right; }}
    th {{ background: #f1f5f9; text-align: center; font-weight: 600; }}
    td:first-child {{ text-align: left; }}
    .pair-cell {{ font-family: monospace; font-size: 0.72rem; word-break: break-all; }}
    .stat-cell {{ background: #f1f5f9; font-style: italic; }}
    @media (max-width: 700px) {{ .charts-row, .prompts-row {{ grid-template-columns: 1fr; }} }}
    h3.src-heading {{ font-size: 0.9rem; font-weight: 700; color: #475569; margin: 1rem 0 0.4rem;
                      padding: 0.2rem 0.5rem; background: #f1f5f9; border-radius: 4px; display: inline-block; }}
    .hm-grid {{ display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 0.5rem; }}
    .hm-grid-2 {{ flex-direction: column; }}
    .hm-item {{ flex: 1 1 auto; min-width: 0; }}
    .hm-full {{ width: 100%; }}
    .hm-title {{ font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 0.3rem; }}
    .hm-wrap {{ overflow-x: auto; }}
    table.hm {{ width: auto; border-collapse: collapse; font-size: 0.7rem; }}
    table.hm th, table.hm td {{ padding: 0.2rem 0.35rem; border: 1px solid #e2e8f0; }}
    table.hm th {{ background: #f1f5f9; font-weight: 600; white-space: nowrap; }}
    .hm-col-hdr {{ text-align: center; min-width: 2.2rem; }}
    .hm-row-hdr {{ text-align: left; white-space: nowrap; font-weight: 600; color: #334155; background: #f8fafc; }}
    .hm-cell {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .hm-na {{ background: #f1f5f9; }}
    .run-legend {{ border-collapse: collapse; font-size: 0.78rem; margin-bottom: 1rem; }}
    .run-legend td {{ padding: 0.2rem 0.6rem; border: 1px solid #e2e8f0; }}
    .run-legend td:first-child {{ font-weight: 700; text-align: center; background: #f1f5f9; }}
    .run-lnk {{ cursor: pointer; color: #2563eb; text-decoration: underline dotted; font-weight: 700; }}
    .run-lnk:hover {{ color: #1d4ed8; }}
    #run-panel {{
      position: fixed; top: 0; right: 0; width: 420px; height: 100vh;
      background: #fff; border-left: 2px solid #e2e8f0;
      box-shadow: -4px 0 20px rgba(0,0,0,.12);
      display: flex; flex-direction: column;
      transform: translateX(100%); transition: transform 0.25s ease;
      z-index: 1000; overflow: hidden;
    }}
    #run-panel.open {{ transform: translateX(0); }}
    #run-panel-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 0.75rem 1rem; background: #f1f5f9; border-bottom: 1px solid #e2e8f0; flex-shrink: 0;
    }}
    #run-panel-title {{ font-weight: 700; font-size: 1rem; color: #334155; }}
    #run-panel-close {{
      cursor: pointer; background: none; border: none; font-size: 1.3rem;
      color: #64748b; padding: 0.1rem 0.45rem; border-radius: 4px; line-height: 1;
    }}
    #run-panel-close:hover {{ background: #e2e8f0; color: #1e293b; }}
    #run-panel-body {{ flex: 1; overflow-y: auto; padding: 1rem; }}
    .panel-label {{ font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin: 1rem 0 0.3rem; }}
    .panel-label:first-child {{ margin-top: 0; }}
    #panel-run-name {{ font-family: monospace; font-size: 0.82rem; color: #334155; margin: 0 0 0.25rem; }}
    .panel-pre {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.75rem; font-size: 0.75rem; font-family: monospace; white-space: pre-wrap; word-break: break-word; color: #334155; max-height: 42vh; overflow-y: auto; margin: 0; }}
    body.panel-open {{ margin-right: 422px; transition: margin-right 0.25s ease; }}
    /* ── top-k explorer ─────────────────────────────────────────────────── */
    .topk-explorer {{ margin-top: 0.5rem; }}
    .topk-controls {{
      display: flex; align-items: center; gap: 0.75rem;
      margin-bottom: 0.75rem; font-size: 0.83rem; color: #334155;
    }}
    .topk-label {{ font-weight: 700; }}
    .topk-controls input[type=range] {{ width: 160px; accent-color: #0f766e; cursor: pointer; }}
    .topk-controls b {{ min-width: 1.5rem; text-align: right; font-size: 1rem; color: #0f766e; }}
    .topk-avail {{ color: #94a3b8; font-size: 0.75rem; }}
    .topk-wrap {{ overflow-x: auto; }}
    table.topk-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    table.topk-table {{ width: auto; }}
    table.topk-table th, table.topk-table td {{
      padding: 0.25rem 0.45rem; border: 1px solid #e2e8f0; white-space: nowrap;
    }}
    table.topk-table th {{ background: #f1f5f9; font-weight: 600; text-align: center; }}
    .rank-col {{ min-width: 2.8rem; }}
    .rank-num {{ text-align: center; font-variant-numeric: tabular-nums; color: #64748b; font-size: 0.75rem; background: #f8fafc; }}
    .step-col-hdr div {{
      min-width: 5rem; padding: 0.3rem 0.5rem !important;
      display: flex; flex-direction: column; align-items: center; gap: 2px;
      border: 1px solid #e2e8f0; background: #f1f5f9;
    }}
    .step-num-hdr {{ font-size: 0.7rem; color: #64748b; font-variant-numeric: tabular-nums; }}
    .step-kind-hdr {{ font-size: 0.68rem; font-weight: 700; }}
    .step-kind-hdr.prefill {{ color: #7c3aed; }}
    .step-kind-hdr.decode  {{ color: #0369a1; }}
    .step-gen-hdr {{ font-size: 0.72rem; font-family: monospace; color: #1e293b; white-space: pre; }}
    .topk-cell {{
      text-align: center; vertical-align: middle;
      padding: 0.2rem 0.4rem !important;
    }}
    .topk-cell .tok-text {{ display: block; font-family: monospace; font-size: 0.75rem; white-space: pre; color: #1e293b; }}
    .topk-cell .tok-prob {{ display: block; font-size: 0.65rem; color: #475569; font-style: italic; }}
    .topk-na {{ text-align: center; color: #cbd5e1; font-size: 0.75rem; }}
    .topk-missing {{ font-size: 0.82rem; color: #94a3b8; font-style: italic; margin: 0.5rem 0; }}
    .topk-missing code {{ background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 3px; font-style: normal; }}
  </style>
</head>
<body>
  <div id="run-panel">
    <div id="run-panel-header">
      <span id="run-panel-title">Run details</span>
      <button id="run-panel-close" title="Close">&times;</button>
    </div>
    <div id="run-panel-body">
      <div class="panel-label">Run</div><p id="panel-run-name"></p>
      <div class="panel-label">Prompt</div><pre id="panel-prompt" class="panel-pre"></pre>
      <div class="panel-label">Generated output</div><pre id="panel-generated" class="panel-pre"></pre>
    </div>
  </div>
  <h1>Activation Comparison Report</h1>
  <p class="meta">Generated {now} &middot; {len(pairs)} inter-run pair(s) &middot; {len(intra_runs)} run(s) with intra-run comparison</p>

  <!-- ═══════════════════════ INTRA-RUN SECTION ═══════════════════════════ -->
  <h2 class="section-heading">Initial → Final state within each run</h2>
  <p class="meta">How much does each run's residual stream change from the last prompt token to the last generated token?</p>
  <h3 class="state-subheading">Overview &mdash; mean cosine per run &times; source</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Run</th>
          <th colspan="4">Mean cosine per source</th>
          <th colspan="3">Across sources</th>
        </tr>
        <tr>{overview_header}</tr>
      </thead>
      <tbody>{overview_rows_intra}</tbody>
    </table>
  </div>
  <h3 class="state-subheading">Aggregate statistics</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Source</th>
          <th colspan="3">Cosine similarity</th>
          <th colspan="3">MAE</th>
          <th colspan="3">Top-1% overlap</th>
        </tr>
        <tr>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
        </tr>
      </thead>
      <tbody>{agg_rows_intra}</tbody>
    </table>
  </div>
  <h3 class="state-subheading">Layer heatmaps &mdash; runs &times; layers</h3>
  <p class="meta">Rows = runs &middot; Columns = transformer layers &middot; Color: green = similar, red = divergent</p>
  {layer_sections_intra}
  <h3 class="state-subheading">Per-run details &amp; generation explorer</h3>
  {intra_sections_html}

  <!-- ═══════════════════════ INTER-RUN SECTION ═══════════════════════════ -->
  <h2 class="section-heading">Inter-run comparison &mdash; overview (mean cosine per pair &times; source)</h2>
  <h3 class="state-subheading">Initial state &mdash; prefill, last prompt token</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Pair</th>
          <th colspan="4">Mean cosine per source</th>
          <th colspan="3">Across sources</th>
        </tr>
        <tr>{overview_header}</tr>
      </thead>
      <tbody>{overview_rows_initial}</tbody>
    </table>
  </div>
  <h3 class="state-subheading">Final state &mdash; last generation step, last token</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Pair</th>
          <th colspan="4">Mean cosine per source</th>
          <th colspan="3">Across sources</th>
        </tr>
        <tr>{overview_header}</tr>
      </thead>
      <tbody>{overview_rows_final}</tbody>
    </table>
  </div>

  <h2 class="section-heading">Inter-run aggregate statistics across all pairs</h2>
  <p class="meta">Mean &pm; std across pairs for each source.</p>
  <h3 class="state-subheading">Initial state</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Source</th>
          <th colspan="3">Cosine similarity (across layers)</th>
          <th colspan="3">MAE (across layers)</th>
          <th colspan="3">Top-1% overlap (across layers)</th>
        </tr>
        <tr>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
        </tr>
      </thead>
      <tbody>{agg_rows_initial}</tbody>
    </table>
  </div>
  <h3 class="state-subheading">Final state</h3>
  <div class="overview-wrap">
    <table>
      <thead>
        <tr>
          <th rowspan="2">Source</th>
          <th colspan="3">Cosine similarity (across layers)</th>
          <th colspan="3">MAE (across layers)</th>
          <th colspan="3">Top-1% overlap (across layers)</th>
        </tr>
        <tr>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
          <th>Mean&nbsp;(mean&pm;std)</th><th>Std&nbsp;(mean&pm;std)</th><th>Median&nbsp;(mean&pm;std)</th>
        </tr>
      </thead>
      <tbody>{agg_rows_final}</tbody>
    </table>
  </div>

  <h2 class="section-heading">Inter-run run &times; run similarity matrices</h2>
  <p class="meta">Each cell shows the metric for that run pair. Diagonal is empty. Column labels refer to the run legend below.</p>
  <table class="run-legend"><tbody>{run_legend_html}</tbody></table>
  <h3 class="state-subheading">Initial state</h3>
  {rr_sections_initial}
  <h3 class="state-subheading">Final state</h3>
  {rr_sections_final}

  <h2 class="section-heading">Inter-run layer heatmaps &mdash; pairs &times; layers</h2>
  <p class="meta">Rows = run pairs &middot; Columns = transformer layers &middot; Color: green = similar, red = divergent</p>
  <h3 class="state-subheading">Initial state</h3>
  {layer_sections_initial}
  <h3 class="state-subheading">Final state</h3>
  {layer_sections_final}

  <h2 class="section-heading">Inter-run per-pair details</h2>
  {pair_sections_html}

  <script>
    const RUN_DATA = {runs_js_data_json};
    const panel      = document.getElementById('run-panel');
    const panelTitle = document.getElementById('run-panel-title');
    const panelName  = document.getElementById('panel-run-name');
    const panelPrompt    = document.getElementById('panel-prompt');
    const panelGenerated = document.getElementById('panel-generated');

    function openPanel(runLabel) {{
      const d = RUN_DATA[runLabel];
      if (!d) return;
      panelTitle.textContent     = runLabel;
      panelName.textContent      = d.name;
      panelPrompt.textContent    = d.prompt    || '(none)';
      panelGenerated.textContent = d.generated || '(none)';
      panel.classList.add('open');
      document.body.classList.add('panel-open');
    }}
    document.getElementById('run-panel-close').addEventListener('click', () => {{
      panel.classList.remove('open');
      document.body.classList.remove('panel-open');
    }});
    document.addEventListener('click', (e) => {{
      const el = e.target.closest('.run-lnk');
      if (!el) return;
      const runLabel = el.dataset.run;
      if (runLabel) openPanel(runLabel);
    }});
  </script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def main(runs_dir: Path) -> None:
    runs = discover_runs(runs_dir)

    if len(runs) < 2:
        print(f"Need at least 2 valid runs in {runs_dir}, found {len(runs)}.")
        return

    pairs = list(combinations(runs, 2))
    print(f"Found {len(runs)} runs → {len(pairs)} pair(s)\n")
    print("=" * 80)

    all_summaries: list[dict] = []
    all_cosines: list[dict] = []

    for run_a, run_b in pairs:
        pair_key = f"{run_a.path.name}  vs  {run_b.path.name}"
        gen_a    = run_label(run_a, short=True)
        gen_b    = run_label(run_b, short=True)
        prompt_a = (run_a.summary().get("prompt", "") or "").strip()
        prompt_b = (run_b.summary().get("prompt", "") or "").strip()

        print(f"\n{'─' * 80}")
        print(f"  A: {run_label(run_a)}")
        print(f"  B: {run_label(run_b)}")
        print(f"{'─' * 80}")

        for state_label, state_fn in [("initial", initial_state), ("final", final_state)]:
            summary_df, cosines_df = compare_pair(run_a, run_b, state_fn)
            print(f"\n  [{state_label} state]")
            print(summary_df.to_string())

            for src, row in summary_df.iterrows():
                all_summaries.append({
                    "state": state_label,
                    "pair": pair_key,
                    "run_a": run_a.path.name,
                    "run_b": run_b.path.name,
                    "generated_a": gen_a,
                    "generated_b": gen_b,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "source": src,
                    **row.to_dict(),
                })

            for _, row in cosines_df.iterrows():
                all_cosines.append({
                    "state": state_label,
                    "pair": pair_key,
                    "run_a": run_a.path.name,
                    "run_b": run_b.path.name,
                    "generated_a": gen_a,
                    "generated_b": gen_b,
                    **row.to_dict(),
                })

    # ── intra-run: initial vs final + top-k explorer ──────────────────────────
    print(f"\n{'=' * 80}")
    print("  Intra-run: initial state vs final state")
    print(f"{'=' * 80}")

    intra_summaries: list[dict] = []
    intra_cosines: list[dict] = []
    run_topk: dict[str, list[dict] | None] = {}

    for run in runs:
        run_name  = run.path.name
        generated = (run.summary().get("generated", "") or "").replace("\n", " ").strip()
        prompt    = (run.summary().get("prompt", "") or "").strip()

        print(f"\n  {run_label(run)}")
        summary_df, cosines_df = compare_run_states(run)
        print(summary_df.to_string())

        topk = get_step_topk(run)
        run_topk[run_name] = topk
        if topk:
            print(f"    top-k: {len(topk)} steps × {len(topk[0]['top'])} tokens")
        else:
            print("    top-k: not captured (capture.top_k_probs=0)")

        for src, row in summary_df.iterrows():
            intra_summaries.append({
                "run": run_name,
                "generated": generated,
                "prompt": prompt,
                "source": src,
                **row.to_dict(),
            })

        for _, row in cosines_df.iterrows():
            intra_cosines.append({
                "run": run_name,
                **row.to_dict(),
            })

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = runs_dir.parent
    summary_path       = out_dir / "comparison_report.csv"
    cosines_path       = out_dir / "comparison_cosines.csv"
    intra_report_path  = out_dir / "intra_run_report.csv"
    intra_cosines_path = out_dir / "intra_run_cosines.csv"
    html_path          = out_dir / f"comparison_report_{ts}.html"

    report_df        = pd.DataFrame(all_summaries)
    cosines_df       = pd.DataFrame(all_cosines)
    intra_report_df  = pd.DataFrame(intra_summaries)
    intra_cosines_df = pd.DataFrame(intra_cosines)

    report_df.to_csv(summary_path, index=False)
    cosines_df.to_csv(cosines_path, index=False)
    intra_report_df.to_csv(intra_report_path, index=False)
    intra_cosines_df.to_csv(intra_cosines_path, index=False)

    generate_html_report(report_df, cosines_df, intra_report_df, intra_cosines_df, run_topk, html_path)

    print(f"\nInter-run summary → {summary_path}")
    print(f"Inter-run layers  → {cosines_path}")
    print(f"Intra-run summary → {intra_report_path}")
    print(f"Intra-run layers  → {intra_cosines_path}")
    print(f"HTML report       → {html_path}")


if __name__ == "__main__":
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _project_root / "runs"
    main(runs_dir)
