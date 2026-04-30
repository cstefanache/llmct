"""Self-contained HTML report generator for Activation Lab."""
from __future__ import annotations

import base64
import html as html_mod
import json
from typing import Any

import numpy as np

from . import compute
from .loader import RunRegistry, load_npz, resolve_npz
from .render import matrix_to_png

# ── constants ──────────────────────────────────────────────────────────────────

CONV_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"]

SOURCE_COLORS: dict[str, str] = {
    "hidden_in": "#2563eb",
    "hidden_out": "#16a34a",
    "attn_out": "#dc2626",
    "mlp_down_out": "#a855f7",
    "qkv_last": "#f97316",
}

PAIR_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316",
    "#0891b2", "#ca8a04", "#ec4899", "#475569", "#84cc16",
]

HEATMAP_VARIANTS = ["a", "b", "abs_diff", "sq_err", "hadamard", "ratio"]
VARIANT_LABELS: dict[str, str] = {
    "a": "A (raw)",
    "b": "B (raw)",
    "abs_diff": "Absolute Difference",
    "sq_err": "Squared Error",
    "hadamard": "Hadamard Product",
    "ratio": "Normalized Ratio (a / |b|)",
}

# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       font-size: 14px; color: #111827; margin: 0; background: #f9fafb; }
.page { max-width: 1100px; margin: 0 auto; padding: 24px 20px 60px; }
h1 { font-size: 22px; margin-bottom: 4px; }
.subtitle { color: #6b7280; font-size: 12px; margin-bottom: 28px; }
h2 { font-size: 17px; margin: 32px 0 10px; padding-bottom: 6px;
     border-bottom: 2px solid #e5e7eb; }
h3 { font-size: 14px; margin: 20px 0 8px; color: #374151; }
h4 { font-size: 13px; margin: 14px 0 6px; color: #4b5563; font-weight: 600; }
.section { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 16px 20px; margin-bottom: 16px; }
.section + .section { margin-top: 0; }
.subsection { border-left: 3px solid #e5e7eb; padding-left: 14px;
              margin: 16px 0; }
.muted { color: #6b7280; font-size: 12px; }
.kv { display: grid; grid-template-columns: max-content 1fr;
      gap: 4px 16px; margin-bottom: 8px; }
.kv dt { font-weight: 600; color: #374151; }
.kv dd { margin: 0; color: #111827; word-break: break-all; }
table.metrics { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
table.metrics th, table.metrics td {
  border: 1px solid #e5e7eb; padding: 4px 10px; text-align: right; }
table.metrics th { background: #f3f4f6; text-align: left; }
table.metrics td:first-child { text-align: left; font-weight: 600; }
.img-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.img-card { text-align: center; }
.img-card .label { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
.img-card img { max-width: 300px; display: block; border: 1px solid #e5e7eb;
                border-radius: 4px; }
.chart-wrap { position: relative; margin: 6px 0; background: #fff;
              border: 1px solid #f3f4f6; border-radius: 4px; }
pre { background: #f3f4f6; border-radius: 4px; padding: 10px 12px;
      font-size: 11px; overflow: auto; max-height: 300px; }
.depth-badge { display: inline-block; background: #dcfce7; color: #166534;
               border-radius: 4px; padding: 2px 8px; font-size: 12px;
               font-weight: 600; margin: 2px 4px; }
.depth-badge.none { background: #fee2e2; color: #991b1b; }
.toc { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
       padding: 14px 18px; margin-bottom: 24px; }
.toc h2 { margin-top: 0; }
.toc ul { margin: 0; padding-left: 20px; }
.toc li { margin: 3px 0; }
.toc a { color: #2563eb; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.source-tag { display: inline-block; padding: 1px 7px; border-radius: 10px;
              font-size: 11px; font-weight: 600; color: #fff; margin-right: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; }
.legend-swatch { width: 22px; height: 3px; border-radius: 2px; }
"""

# ── Chart.js initializer (runs after Chart.js CDN script is loaded) ────────────

_CHART_JS = r"""
(function () {
  /* ── Line chart ──────────────────────────────────────────────────────────────
   * canvas[data-chart="VAR_NAME"]
   * Variable: { xs: number[], series: [{ label, data, stroke }] }
   */
  function initLine(canvas) {
    var d = window[canvas.getAttribute("data-chart")];
    if (!d || !d.xs || d.xs.length === 0) return;
    new Chart(canvas, {
      type: "line",
      data: {
        labels: d.xs,
        datasets: d.series.map(function (s) {
          return {
            label: s.label || "",
            data: s.data,
            borderColor: s.stroke || "#2563eb",
            backgroundColor: (s.stroke || "#2563eb") + "18",
            borderWidth: 1.5,
            pointRadius: 2,
            pointHoverRadius: 5,
            fill: false,
            spanGaps: true,
            tension: 0.15,
          };
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { size: 11 }, boxWidth: 18, padding: 12 },
          },
          tooltip: { bodyFont: { size: 11 }, titleFont: { size: 11 } },
        },
        scales: {
          x: {
            ticks: { font: { size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 16 },
            grid: { color: "#f0f0f0" },
          },
          y: {
            ticks: { font: { size: 10 } },
            grid: { color: "#f0f0f0" },
          },
        },
      },
    });
  }

  /* ── PCA scatter ─────────────────────────────────────────────────────────────
   * canvas[data-pca="VAR_NAME"]
   * Variable: { coords, ref_indices, layer_indices, layers,
   *             explained_variance, ref_labels, colors }
   * Each dataset = one ref; points sorted by layer, line drawn through them.
   * Point opacity encodes layer depth (lighter = early, darker = deep).
   */
  function initPca(canvas) {
    var d = window[canvas.getAttribute("data-pca")];
    if (!d || !d.coords || d.coords.length === 0) return;

    var colors = d.colors || ["#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316"];
    var maxLi = Math.max.apply(null, d.layer_indices.concat([1]));

    // Group points by ref, sorted by layer index
    var byRef = (d.ref_labels || []).map(function () { return []; });
    d.coords.forEach(function (c, idx) {
      var ri = d.ref_indices[idx];
      if (byRef[ri]) byRef[ri].push({ x: c[0], y: c[1], li: d.layer_indices[idx] });
    });
    byRef.forEach(function (pts) { pts.sort(function (a, b) { return a.li - b.li; }); });

    var ev = d.explained_variance || [0, 0];
    var datasets = byRef.map(function (pts, ri) {
      var color = colors[ri % colors.length];
      return {
        label: (d.ref_labels || [])[ri] || ("ref " + ri),
        data: pts.map(function (p) {
          return { x: p.x, y: p.y, layer: d.layers ? d.layers[p.li] : p.li };
        }),
        // Per-point background opacity based on layer depth
        backgroundColor: pts.map(function (p) {
          var alpha = Math.round((0.25 + 0.75 * (p.li / maxLi)) * 255)
            .toString(16).padStart(2, "0");
          return color + alpha;
        }),
        borderColor: color,
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: true,
        borderWidth: 1,
        tension: 0,
        fill: false,
      };
    });

    new Chart(canvas, {
      type: "scatter",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { size: 11 }, boxWidth: 18, padding: 12 },
          },
          tooltip: {
            bodyFont: { size: 11 },
            callbacks: {
              label: function (ctx) {
                var pt = ctx.raw;
                return ctx.dataset.label + "  layer " + (pt.layer != null ? pt.layer : ctx.dataIndex);
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "PC1 (" + (ev[0] * 100).toFixed(1) + "%)",
              font: { size: 11 },
            },
            ticks: { font: { size: 10 } },
            grid: { color: "#f0f0f0" },
          },
          y: {
            title: {
              display: true,
              text: "PC2 (" + (ev[1] * 100).toFixed(1) + "%)",
              font: { size: 11 },
            },
            ticks: { font: { size: 10 } },
            grid: { color: "#f0f0f0" },
          },
        },
      },
    });
  }

  document.querySelectorAll("canvas[data-chart]").forEach(initLine);
  document.querySelectorAll("canvas[data-pca]").forEach(initPca);
}());
"""

# ── low-level helpers ──────────────────────────────────────────────────────────

def _b64png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _safe(s: Any) -> str:
    return html_mod.escape(str(s))


def _clean_for_json(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if not np.isfinite(f) else f
    if isinstance(v, dict):
        return {k: _clean_for_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_for_json(i) for i in v]
    return v


def _jv(v: Any) -> str:
    """Serialize value to compact JSON string safe for JS embedding."""
    return json.dumps(_clean_for_json(v))


def _js_var(name: str, value: Any, comment: str) -> str:
    """Emit a commented JS const declaration."""
    lines = ["", "/*"]
    for line in comment.strip().split("\n"):
        lines.append(f" * {line}")
    lines.append(" */")
    lines.append(f"var {name} = {_jv(value)};")
    return "\n".join(lines)


def _canvas_chart(var_name: str, height: int = 220) -> str:
    return (
        f'<div class="chart-wrap" style="height:{height}px;max-width:700px">'
        f'<canvas data-chart="{var_name}"></canvas>'
        f'</div>'
    )


def _canvas_pca(var_name: str) -> str:
    return (
        f'<div class="chart-wrap" style="height:380px;max-width:540px">'
        f'<canvas data-pca="{var_name}"></canvas>'
        f'</div>'
    )


def _html_matrix_heatmap(
    matrix: list,
    labels: list[str],
    scheme: str = "diverging",
    domain: tuple[float, float] | None = None,
) -> str:
    """Render an N×N matrix as a coloured HTML table (no JS required)."""
    n = len(labels)
    flat = [v for row in matrix for v in row if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    if not flat:
        return '<p class="muted">no data</p>'
    lo = domain[0] if domain else min(flat)
    hi = domain[1] if domain else max(flat)
    rng = (hi - lo) or 1.0

    def _bg(v: float | None) -> str:
        if v is None:
            return "#f5f5f5"
        t = max(0.0, min(1.0, (v - lo) / rng))
        if scheme == "sequential":
            return f"rgb({round(255-195*t)},{round(255-125*t)},{round(255-35*t)})"
        if t <= 0.5:
            u = t * 2
            return f"rgb({round(70+185*u)},{round(130+125*u)},{round(220-20*u)})"
        u = (t - 0.5) * 2
        return f"rgb(255,{round(255-185*u)},{round(200-130*u)})"

    def _fg(v: float | None) -> str:
        if v is None:
            return "#aaa"
        t = max(0.0, min(1.0, (v - lo) / rng))
        if scheme == "sequential":
            return "#fff" if t > 0.55 else "#333"
        return "#fff" if abs(t - 0.5) > 0.3 else "#333"

    rows_html = ""
    for i, row in enumerate(matrix):
        cells = ""
        for v in row[:n]:
            val_str = f"{v:.3f}" if v is not None else "—"
            cells += (
                f'<td style="background:{_bg(v)};color:{_fg(v)};'
                f'text-align:center;min-width:52px;font-size:11px">{val_str}</td>'
            )
        rows_html += f'<tr><td style="font-size:10px;white-space:nowrap">{_safe(labels[i][:24])}</td>{cells}</tr>'

    header_cells = "".join(
        f'<th style="font-size:10px;max-width:80px;word-break:break-all">{_safe(l[:16])}</th>'
        for l in labels
    )
    return (
        f'<div style="overflow-x:auto">'
        f'<table class="metrics">'
        f'<thead><tr><th></th>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


def _img_embed(data_uri: str, label: str = "", width: int = 320) -> str:
    return (
        f'<div class="img-card">'
        f'<div class="label">{_safe(label)}</div>'
        f'<img src="{data_uri}" alt="{_safe(label)}" style="max-width:{width}px">'
        f"</div>"
    )


def _source_tag(src: str) -> str:
    color = SOURCE_COLORS.get(src, "#555")
    return f'<span class="source-tag" style="background:{color}">{_safe(src)}</span>'


def _section(title: str, sec_id: str, content: str, accent: str = "") -> str:
    style = f' style="border-left:3px solid {accent}"' if accent else ""
    return (
        f'<div class="section" id="{_safe(sec_id)}"{style}>'
        f"<h2>{_safe(title)}</h2>"
        f"{content}"
        f"</div>"
    )


def _subsection(title: str, sub_id: str, content: str) -> str:
    return (
        f'<div class="subsection" id="{_safe(sub_id)}">'
        f"<h3>{_safe(title)}</h3>"
        f"{content}"
        f"</div>"
    )


def _metrics_table(rows: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """rows: [(row_label, [(col_label, value_str), ...]), ...]"""
    if not rows:
        return ""
    cols = [c for c, _ in rows[0][1]]
    thead = "<tr><th>source</th>" + "".join(f"<th>{_safe(c)}</th>" for c in cols) + "</tr>"
    tbody = ""
    for row_label, cells in rows:
        tbody += "<tr><td>" + _safe(row_label) + "</td>"
        for _, val in cells:
            tbody += f"<td>{_safe(val)}</td>"
        tbody += "</tr>"
    return f'<table class="metrics">\n<thead>{thead}</thead>\n<tbody>{tbody}</tbody>\n</table>'


# ── image helpers ──────────────────────────────────────────────────────────────

def _stacked_png(tensors: dict, extract_fn, cmap: str, signed: bool) -> bytes | None:
    layers = compute.layers_available(tensors)
    if not layers:
        return None
    mats = [extract_fn(tensors, li) for li in layers]
    return matrix_to_png(np.vstack(mats), cmap=cmap, signed=signed)


def _pair_heatmap_png(
    tensors_a: dict, tensors_b: dict,
    source: str, variant: str,
) -> bytes | None:
    """Produce a pair heatmap PNG (all-layers stacked)."""
    if source == "attention":
        if not (compute.has_attention(tensors_a) and compute.has_attention(tensors_b)):
            return None
        def _extract(t):
            layers = compute.layers_available(t)
            return np.vstack([compute.attention_matrix(t, li) for li in layers])
        cmap, signed = "viridis", False
    elif source == "qk":
        if not (compute.has_qk(tensors_a) and compute.has_qk(tensors_b)):
            return None
        def _extract(t):
            layers = compute.layers_available(t)
            return np.vstack([compute.qk_scores(t, li) for li in layers])
        cmap, signed = "RdBu_r", True
    else:  # qkv
        if not (compute.has_qkv_last(tensors_a) and compute.has_qkv_last(tensors_b)):
            return None
        def _extract(t):
            return compute.qkv_last_matrix(t)
        cmap, signed = "RdBu_r", True

    if variant == "a":
        try:
            return matrix_to_png(_extract(tensors_a), cmap=cmap, signed=signed)
        except Exception:
            return None
    if variant == "b":
        try:
            return matrix_to_png(_extract(tensors_b), cmap=cmap, signed=signed)
        except Exception:
            return None

    try:
        ma = _extract(tensors_a)
        mb = _extract(tensors_b)
        if ma.shape != mb.shape:
            return None
        diffs = compute.diff_matrices(ma, mb)
        mat = diffs[variant]
        if variant in ("abs_diff", "sq_err"):
            signed = False
            cmap = "magma"
        return matrix_to_png(mat, cmap=cmap, signed=signed)
    except Exception:
        return None


# ── chart data builders ────────────────────────────────────────────────────────

def _chart_data_entropy(entropy_resp: dict) -> dict:
    return {
        "xs": entropy_resp["layers"],
        "series": [{"label": "entropy (nats)", "data": entropy_resp["entropy"], "stroke": "#2563eb"}],
    }


def _chart_data_convergence(conv: dict, source: str) -> dict:
    return {
        "xs": conv["adj_layers"],
        "series": [
            {"label": "adj cosine — last token", "data": conv["adj_cosine_last"], "stroke": "#2563eb"},
            {"label": "adj cosine — mean seq",   "data": conv["adj_cosine_mean"], "stroke": "#93c5fd"},
            {"label": "delta norm — last token", "data": conv["delta_norm_last"], "stroke": "#dc2626"},
            {"label": "delta norm — mean seq",   "data": conv["delta_norm_mean"], "stroke": "#fca5a5"},
        ],
    }


def _chart_data_logit_stats(ls: dict, metric: str) -> dict:
    label_map = {
        "entropy": "entropy (nats)",
        "effective_vocab": "effective vocab size",
        "top1_prob": "top-1 probability",
    }
    color_map = {"entropy": "#2563eb", "effective_vocab": "#a855f7", "top1_prob": "#16a34a"}
    return {
        "xs": ls["steps"],
        "series": [{"label": label_map[metric], "data": ls[metric], "stroke": color_map[metric]}],
    }


# ── section builders ──────────────────────────────────────────────────────────

def _build_entropy_block(tensors: dict) -> tuple[str, list[str]]:
    """Returns (html, js_vars)."""
    if not compute.has_attention(tensors):
        return '<p class="muted">no attention weights captured in this NPZ</p>', []
    vals = compute.attention_entropy(tensors)
    layers = compute.layers_available(tensors)
    chart_var = "DATA_entropy"
    js = [_js_var(
        chart_var,
        {"xs": layers, "series": [{"label": "entropy (nats)", "data": vals, "stroke": "#2563eb"}]},
        "Attention entropy per layer\n"
        "xs     – layer indices where attention weights were captured\n"
        "series[0].data – per-layer mean attention entropy in nats\n"
        "  (Shannon entropy of the head-averaged softmax attention distribution;\n"
        "   higher = more diffuse; lower = sharper / more focused)",
    )]
    html = (
        '<h4>Attention entropy per layer</h4>'
        + _canvas_chart(chart_var)
    )
    return html, js


def _build_convergence_block(
    tensors: dict,
    conv_sources: list[str],
    prefix: str = "",
) -> tuple[str, list[str]]:
    """Generate convergence subsections for each source in conv_sources."""
    all_html = ""
    all_js: list[str] = []
    for src in conv_sources:
        try:
            conv = compute.residual_convergence(tensors, source=src)
        except Exception:
            continue
        if not conv["adj_layers"]:
            continue
        var_name = f"DATA_convergence_{prefix}{src}"
        all_js.append(_js_var(
            var_name,
            {"xs": conv["adj_layers"], "series": _chart_data_convergence(conv, src)["series"]},
            f"Residual stream convergence — source: {src}\n"
            "xs – layer transition indices (n → n+1, x-axis of the chart below)\n"
            "series[0] adj cosine (last token) – cosine(h[n+1], h[n]) at the final token;\n"
            "  1.0 = the layer did not change the residual stream at all\n"
            "series[1] adj cosine (mean seq)   – same metric averaged over all token positions\n"
            "series[2] delta norm (last token) – ||h[n+1]−h[n]|| / ||h[n]|| at last token;\n"
            "  0.0 = no change; higher = larger relative update\n"
            "series[3] delta norm (mean seq)   – same metric averaged over the full sequence\n"
            f"effective_depth_last = {conv['effective_depth_last']!r}\n"
            f"  first layer index where adj_cosine_last stays ≥ 0.99 permanently\n"
            f"effective_depth_mean = {conv['effective_depth_mean']!r}",
        ))
        edl = conv["effective_depth_last"]
        edm = conv["effective_depth_mean"]
        badge = lambda v: (
            f'<span class="depth-badge">{_safe(v)}</span>'
            if v is not None else
            '<span class="depth-badge none">n/a</span>'
        )
        stats = (
            f"<p>Effective depth (last token): {badge(edl)}"
            f"&nbsp;&nbsp;Effective depth (mean seq): {badge(edm)}</p>"
        )
        html_block = (
            stats
            + "<h4>Adjacent-layer cosine similarity &amp; relative update norm</h4>"
            + _canvas_chart(var_name)
        )
        all_html += _subsection(
            f"Convergence — source: {src}",
            f"conv-{prefix}{src}",
            html_block,
        )
    return all_html, all_js


def _build_metrics_block(
    tensors_a: dict,
    tensors_b: dict,
    sources: list[str],
) -> tuple[str, list[str]]:
    """Per-source comparison metrics (pair)."""
    if not sources:
        return '<p class="muted">no sources selected</p>', []
    metrics = compute.pair_metrics(tensors_a, tensors_b, sources)
    all_html = ""
    all_js: list[str] = []

    # Summary table
    def _avg(arr: list) -> float:
        v = [x for x in arr if x is not None and np.isfinite(x)]
        return sum(v) / len(v) if v else float("nan")

    table_rows = []
    for src in sources:
        m = metrics.get(src, {})
        table_rows.append((src, [
            ("mean cos (avg)", f"{_avg(m.get('mean_cos', [])):.4f}"),
            ("min cos (min)",  f"{min((x for x in m.get('min_cos', []) if x is not None and np.isfinite(x)), default=float('nan')):.4f}"),
            ("mean MAE (avg)", f"{_avg(m.get('mean_mae', [])):.3e}"),
            ("overlap (avg)",  f"{_avg(m.get('mean_overlap', [])):.4f}"),
        ]))
    all_html += "<h4>Summary — one row per source</h4>" + _metrics_table(table_rows)

    # Per-source chart data variables (one per source with all metrics)
    all_series_cos = []
    all_series_mae = []
    all_series_std = []
    all_series_jac = []
    xs = []

    for src in sources:
        m = metrics.get(src, {})
        if not m or not m.get("layers"):
            continue
        xs = m["layers"]
        color = SOURCE_COLORS.get(src, "#555")
        all_series_cos.append({"label": src, "data": m.get("mean_cos", []), "stroke": color})
        all_series_mae.append({"label": src, "data": m.get("mean_mae", []), "stroke": color})
        all_series_std.append({"label": src, "data": m.get("std_diff", []), "stroke": color})
        all_series_jac.append({"label": src, "data": m.get("mean_overlap", []), "stroke": color})

    if xs:
        charts = [
            ("DATA_metrics_cos", {"xs": xs, "series": all_series_cos},
             "Cosine similarity per layer — all active sources overlaid\n"
             "xs – layer indices\n"
             "series[i].data – mean cosine similarity between A and B activations\n"
             "  at each layer for the named source; 1.0 = identical direction",
             "Cosine similarity per layer"),
            ("DATA_metrics_mae", {"xs": xs, "series": all_series_mae},
             "Mean absolute error per layer — all active sources\n"
             "xs – layer indices\n"
             "series[i].data – mean |A[t,d] − B[t,d]| over all token positions t\n"
             "  and hidden dimensions d for the named source",
             "Mean absolute error per layer"),
            ("DATA_metrics_std", {"xs": xs, "series": all_series_std},
             "Standard deviation of (A − B) per layer\n"
             "xs – layer indices\n"
             "series[i].data – std dev of the element-wise difference A − B\n"
             "  at each layer; higher = more spread in the disagreement",
             "STD of difference per layer"),
            ("DATA_metrics_jaccard", {"xs": xs, "series": all_series_jac},
             "Top-1% activation channel Jaccard overlap per layer\n"
             "xs – layer indices\n"
             "series[i].data – average Jaccard index between the sets of the top-1%\n"
             "  channels by |activation| in A and B; 1.0 = identical active channels",
             "Top-1% Jaccard per layer"),
        ]
        for var_name, data, comment, title in charts:
            all_js.append(_js_var(var_name, data, comment))
            all_html += f"<h4>{_safe(title)}</h4>" + _canvas_chart(var_name)

    return all_html, all_js


def _build_pca_block(
    adv: dict,
    sources: list[str],
    ref_labels: list[str],
    prefix: str = "",
) -> tuple[str, list[str]]:
    all_html = ""
    all_js: list[str] = []
    pca = adv.get("pca", {})
    for src in sources:
        if src not in pca:
            continue
        pdata = pca[src]
        var_name = f"DATA_pca_{prefix}{src}"
        all_js.append(_js_var(
            var_name,
            {**pdata, "ref_labels": ref_labels, "colors": PAIR_COLORS},
            f"PCA layer trajectories — source: {src}\n"
            "coords         – [pc1, pc2] for each (ref, layer) point; mean activation\n"
            "                 vector per ref per layer projected onto the top-2 PCs\n"
            "                 computed jointly across ALL refs\n"
            "ref_indices    – index into ref_labels for each coords entry\n"
            "layer_indices  – index into layers[] for each coords entry\n"
            "layers         – list of actual layer numbers\n"
            "explained_variance – [pc1_fraction, pc2_fraction] of total variance\n"
            "ref_labels     – display names for each ref (same order as refs)\n"
            "Trajectories that overlay each other = similar representations;\n"
            "paths that fan out = snapshots agree early but diverge at deeper layers",
        ))
        ev = pdata.get("explained_variance", [0, 0])
        all_html += _subsection(
            f"PCA Layer Trajectories — {_source_tag(src)} "
            f"(PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%)",
            f"pca-{prefix}{src}",
            _canvas_pca(var_name),
        )
    return all_html, all_js


def _build_spectral_block(
    adv: dict,
    sources: list[str],
    ref_labels: list[str],
    prefix: str = "",
) -> tuple[str, list[str]]:
    all_html = ""
    all_js: list[str] = []
    svd = adv.get("svd", {})
    for src in sources:
        if src not in svd:
            continue
        sdata = svd[src]
        layers = sdata["layers"]
        refs_data = sdata["refs"]
        charts: list[tuple[str, str, str, str]] = [
            ("spectral_norm",  "DATA_svd_spectral_"  + prefix + src,
             f"Spectral norm (‖h‖₂) per layer — source: {src}\n"
             "xs – layer indices\n"
             "series[i].data – spectral norm (largest singular value) of the\n"
             "  activation matrix at that layer for the named ref; for T=1 captures\n"
             "  this equals the L2 norm of the activation vector",
             "Spectral norm per layer"),
            ("nuclear_norm",   "DATA_svd_nuclear_"   + prefix + src,
             f"Nuclear norm (Σσᵢ) per layer — source: {src}\n"
             "xs – layer indices\n"
             "series[i].data – sum of all singular values of the activation matrix;\n"
             "  reflects total energy spread across all feature directions",
             "Nuclear norm per layer"),
            ("effective_rank", "DATA_svd_effrank_"   + prefix + src,
             f"Effective rank per layer — source: {src}\n"
             "xs – layer indices\n"
             "series[i].data – exp(entropy of normalized singular values);\n"
             "  a continuous count of how many independent directions are used;\n"
             "  null where T=1 (rank-1 capture trivially gives effective_rank=1)",
             "Effective rank per layer"),
        ]
        sub_html = ""
        for field, var_name, comment, title in charts:
            series = [
                {"label": ref_labels[ri], "data": [v for v in rd[field]], "stroke": PAIR_COLORS[ri % len(PAIR_COLORS)]}
                for ri, rd in enumerate(refs_data)
            ]
            all_js.append(_js_var(var_name, {"xs": layers, "series": series}, comment))
            sub_html += f"<h4>{_safe(title)}</h4>" + _canvas_chart(var_name)
        all_html += _subsection(
            f"Spectral Analysis — {_source_tag(src)}",
            f"spectral-{prefix}{src}",
            sub_html,
        )
    return all_html, all_js


def _build_group_block(
    grp: dict,
    sources: list[str],
    ref_labels: list[str],
    prefix: str = "",
) -> tuple[str, list[str]]:
    all_html = ""
    all_js: list[str] = []
    n = len(ref_labels)
    for src in sources:
        if src not in grp:
            continue
        g = grp[src]
        layers = g["layers"]
        sub_html = ""

        # Cosine to centroid
        centroid_series = [
            {
                "label": ref_labels[i],
                "data": [row[i] for row in g["cos_to_centroid"]],
                "stroke": PAIR_COLORS[i % len(PAIR_COLORS)],
            }
            for i in range(n)
        ]
        var_centroid = f"DATA_group_centroid_{prefix}{src}"
        all_js.append(_js_var(
            var_centroid,
            {"xs": layers, "series": centroid_series},
            f"Cosine to group centroid per layer — source: {src}\n"
            "xs – layer indices\n"
            "series[i].data – cosine similarity of ref i's mean activation to the\n"
            "  group centroid (mean over all refs) at each layer;\n"
            "  1.0 = snapshot agrees with the group; lower = outlier at that depth",
        ))
        sub_html += "<h4>Cosine to group centroid</h4>" + _canvas_chart(var_centroid)

        # Layer divergence
        var_div = f"DATA_group_divergence_{prefix}{src}"
        all_js.append(_js_var(
            var_div,
            {"xs": layers, "series": [{"label": "1 − mean pairwise cos", "data": g["divergence"], "stroke": "#dc2626"}]},
            f"Layer divergence — source: {src}\n"
            "xs – layer indices\n"
            "series[0].data – 1 − (mean pairwise cosine across all N(N-1)/2 ref pairs)\n"
            "  at each layer; 0 = all refs agree perfectly; spikes mark where they diverge",
        ))
        sub_html += "<h4>Layer divergence (1 − mean pairwise cosine)</h4>" + _canvas_chart(var_div)

        # Pairwise cosine matrix
        var_cos_mat = f"DATA_group_pairwise_cos_{prefix}{src}"
        all_js.append(_js_var(
            var_cos_mat,
            {"matrix": g["pairwise_cos"], "labels": ref_labels, "scheme": "diverging", "domain": [-1, 1]},
            f"All-pairs cosine similarity matrix — source: {src}\n"
            "matrix – N×N layer-averaged pairwise cosine between ref mean activations;\n"
            "  diagonal = 1.0 (self); off-diagonal = similarity between each pair of refs",
        ))
        sub_html += "<h4>Pairwise cosine similarity (layer-averaged)</h4>"
        sub_html += _html_matrix_heatmap(g["pairwise_cos"], ref_labels, "diverging", (-1.0, 1.0))

        # Top-k Jaccard matrix
        var_jac_mat = f"DATA_group_jaccard_{prefix}{src}"
        all_js.append(_js_var(
            var_jac_mat,
            {"matrix": g["topk_jaccard"], "labels": ref_labels, "scheme": "sequential", "domain": [0, 1]},
            f"Top-1% channel Jaccard matrix — source: {src}\n"
            "matrix – N×N layer-averaged Jaccard index between top-1% active channel\n"
            "  sets; 1.0 = identical active channels; high Jaccard + low cosine =\n"
            "  same channels, opposite magnitudes/signs",
        ))
        sub_html += "<h4>Top-1% channel Jaccard (layer-averaged)</h4>"
        sub_html += _html_matrix_heatmap(g["topk_jaccard"], ref_labels, "sequential", (0.0, 1.0))

        all_html += _subsection(
            f"Group Analysis — {_source_tag(src)}",
            f"group-{prefix}{src}",
            sub_html,
        )
    return all_html, all_js


# ── page assembler ─────────────────────────────────────────────────────────────

def _full_page(title: str, subtitle: str, toc_entries: list[tuple[str, str]], sections: str, js_vars: list[str]) -> str:
    toc_items = "".join(f'<li><a href="#{_safe(sid)}">{_safe(lbl)}</a></li>' for lbl, sid in toc_entries)
    js_block = "\n".join(js_vars)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_safe(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">
<h1>{_safe(title)}</h1>
<div class="subtitle">{_safe(subtitle)}</div>
<div class="toc"><h2>Contents</h2><ul>{toc_items}</ul></div>
{sections}
</div>
<script>
{js_block}
</script>
<script>
{_CHART_JS}
</script>
</body>
</html>"""


# ── public report generators ───────────────────────────────────────────────────

def generate_run_report(registry: RunRegistry, run_id: str) -> str:
    run = registry.run_json(run_id)
    steps = registry.steps_index(run_id)

    js_vars: list[str] = []
    sections = ""
    toc: list[tuple[str, str]] = []

    # ── overview ──────────────────────────────────────────────────────────────
    sc = run.get("scenario", {})
    model = run.get("model", {})
    env = run.get("env", {})
    prompt = sc.get("prompt", {})
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    messages = prompt.get("messages", [])
    dl = "<dl class='kv'>"
    for k, v in [("name", sc.get("name")), ("model", model.get("id")), ("device", env.get("device")), ("created", run.get("created_at"))]:
        if v:
            dl += f"<dt>{_safe(k)}</dt><dd>{_safe(v)}</dd>"
    dl += "</dl>"
    msgs_html = "<ol>"
    for m in messages:
        msgs_html += f"<li><b>{_safe(m.get('role','?'))}:</b> {_safe(str(m.get('content',''))[:400])}</li>"
    msgs_html += "</ol>"
    toc.append(("Run Overview", "overview"))
    sections += _section("Run Overview", "overview", dl + "<h3>Prompt messages</h3>" + msgs_html)

    # ── logit stats ───────────────────────────────────────────────────────────
    logit_steps, logit_entropy, logit_ev, logit_top1 = [], [], [], []
    for s in (steps or []):
        ls = s.get("logit_stats")
        if ls:
            logit_steps.append(s["step"])
            logit_entropy.append(ls.get("entropy"))
            logit_ev.append(ls.get("effective_vocab"))
            logit_top1.append(ls.get("top1_prob"))

    if logit_steps:
        ls_dict = {"steps": logit_steps, "entropy": logit_entropy, "effective_vocab": logit_ev, "top1_prob": logit_top1}
        charts_html = ""
        for metric, title, comment in [
            ("entropy",        "Entropy (nats)",
             "Next-token logit distribution entropy per generation step\n"
             "steps – generation step index\n"
             "series[0].data – Shannon entropy of the full-vocabulary softmax in nats;\n"
             "  low = model is confident (peaked distribution);\n"
             "  high = model is uncertain (flat distribution)"),
            ("effective_vocab", "Effective vocabulary size — exp(entropy)",
             "Effective vocabulary size per generation step\n"
             "steps – generation step index\n"
             "series[0].data – exp(entropy); number of tokens the model effectively\n"
             "  spreads probability mass across; uniform over V tokens → value = V"),
            ("top1_prob",      "Top-1 token probability",
             "Top-1 probability per generation step\n"
             "steps – generation step index\n"
             "series[0].data – probability mass on the single most likely next token"),
        ]:
            colors = {"entropy": "#2563eb", "effective_vocab": "#a855f7", "top1_prob": "#16a34a"}
            labels = {"entropy": "entropy (nats)", "effective_vocab": "effective vocab", "top1_prob": "top-1 prob"}
            var_name = f"DATA_logit_{metric}"
            js_vars.append(_js_var(
                var_name,
                {"xs": logit_steps, "series": [{"label": labels[metric], "data": ls_dict[metric], "stroke": colors[metric]}]},
                comment,
            ))
            charts_html += f"<h4>{_safe(title)}</h4>" + _canvas_chart(var_name, height=160)

        toc.append(("Next-token distribution peakedness", "logit-stats"))
        sections += _section("Next-token distribution peakedness", "logit-stats", charts_html, accent="#a855f7")

    # ── full run.json ─────────────────────────────────────────────────────────
    toc.append(("Full run.json", "run-json"))
    js_vars.append(_js_var(
        "DATA_run_json",
        run,
        "Full run.json — scenario configuration, model info, environment details,\n"
        "and per-step metadata written by the capture harness",
    ))
    sections += _section(
        "Full run.json", "run-json",
        f"<pre>{_safe(json.dumps(_clean_for_json(run), indent=2))}</pre>",
    )

    subtitle = f"Run: {run_id} · {run.get('created_at', '')}"
    return _full_page(f"Activation Lab — Run report: {sc.get('name', run_id)}", subtitle, toc, sections, js_vars)


def generate_npz_report(registry: RunRegistry, ref: dict, sources: list[str]) -> str:
    """ref has keys: run_id, kind, name"""
    run_id, kind, name = ref["run_id"], ref["kind"], ref["name"]
    from .loader import resolve_npz
    path = resolve_npz(registry, run_id, kind, name)
    tensors = load_npz(path)
    meta_inv = {"keys": list(tensors.keys()), "layers": compute.layers_available(tensors)}
    has_attn = compute.has_attention(tensors)
    has_qk_ = compute.has_qk(tensors)
    has_qkv = compute.has_qkv_last(tensors)

    js_vars: list[str] = []
    sections = ""
    toc: list[tuple[str, str]] = []

    # ── meta ──────────────────────────────────────────────────────────────────
    js_vars.append(_js_var(
        "DATA_meta",
        {"run_id": run_id, "kind": kind, "name": name,
         "layers": meta_inv["layers"], "has_attention": has_attn,
         "has_qk": has_qk_, "has_qkv_last": has_qkv,
         "keys": meta_inv["keys"]},
        "NPZ metadata\n"
        "run_id  – identifier of the parent run\n"
        "kind    – 'snapshot', 'reference', or 'step'\n"
        "name    – NPZ filename stem\n"
        "layers  – list of captured layer indices\n"
        "keys    – all tensor keys present in the NPZ file",
    ))
    dl = "<dl class='kv'>"
    for k, v in [("run_id", run_id), ("kind", kind), ("name", name), ("layers", str(meta_inv["layers"])), ("has_attention", has_attn), ("has_qk", has_qk_), ("has_qkv_last", has_qkv)]:
        dl += f"<dt>{_safe(k)}</dt><dd>{_safe(v)}</dd>"
    dl += "</dl>"
    toc.append(("NPZ overview", "overview"))
    sections += _section("NPZ overview", "overview", dl)

    # ── images — all layers stacked ───────────────────────────────────────────
    imgs_html = ""
    if has_attn:
        png = _stacked_png(tensors, compute.attention_matrix, "viridis", False)
        if png:
            imgs_html += "<h4>Attention softmax — all layers stacked (rows = layers)</h4>"
            imgs_html += _img_embed(_b64png(png), "attention softmax", width=500)
    if has_qk_:
        png = _stacked_png(tensors, compute.qk_scores, "RdBu_r", True)
        if png:
            imgs_html += "<h4>Raw q·k / √d_k scores — all layers stacked</h4>"
            imgs_html += _img_embed(_b64png(png), "q·k scores", width=500)
    if has_qkv:
        try:
            mat = compute.qkv_last_matrix(tensors)
            png = matrix_to_png(mat, cmap="RdBu_r", signed=True)
            imgs_html += "<h4>Stacked ⟨q,k⟩·v at last token — rows = layers</h4>"
            imgs_html += _img_embed(_b64png(png), "qkv_last", width=500)
        except Exception:
            pass

    # Per-layer images
    layers = meta_inv["layers"]
    if has_attn and len(layers) > 1:
        imgs_html += "<h4>Attention softmax — per-layer</h4><div class='img-grid'>"
        for li in layers:
            try:
                mat = compute.attention_matrix(tensors, li)
                png = matrix_to_png(mat, cmap="viridis", signed=False)
                imgs_html += _img_embed(_b64png(png), f"layer {li}", width=200)
            except Exception:
                pass
        imgs_html += "</div>"
    if has_qk_ and len(layers) > 1:
        imgs_html += "<h4>Raw q·k — per-layer</h4><div class='img-grid'>"
        for li in layers:
            try:
                mat = compute.qk_scores(tensors, li)
                png = matrix_to_png(mat, cmap="RdBu_r", signed=True)
                imgs_html += _img_embed(_b64png(png), f"layer {li}", width=200)
            except Exception:
                pass
        imgs_html += "</div>"

    if imgs_html:
        toc.append(("Attention heatmaps", "heatmaps"))
        sections += _section("Attention heatmaps", "heatmaps", imgs_html)

    # ── entropy ───────────────────────────────────────────────────────────────
    ent_html, ent_js = _build_entropy_block(tensors)
    js_vars.extend(ent_js)
    toc.append(("Attention entropy", "entropy"))
    sections += _section("Attention entropy", "entropy", ent_html)

    # ── convergence — one subsection per source ───────────────────────────────
    conv_html, conv_js = _build_convergence_block(tensors, CONV_SOURCES)
    js_vars.extend(conv_js)
    toc.append(("Residual stream convergence", "convergence"))
    sections += _section(
        "Residual stream convergence",
        "convergence",
        "<p class='muted'>Adjacent-layer cosine similarity near 1.0 means the stream has settled. "
        "Effective depth = first layer where cosine ≥ 0.99 permanently.</p>"
        + conv_html,
        accent="#16a34a",
    )

    subtitle = f"{kind}:{name}  ·  run {run_id}"
    return _full_page(f"Activation Lab — NPZ report: {name}", subtitle, toc, sections, js_vars)


def generate_pair_report(registry: RunRegistry, a: dict, b: dict, sources: list[str]) -> str:
    from .loader import resolve_npz
    ta = load_npz(resolve_npz(registry, a["run_id"], a["kind"], a["name"]))
    tb = load_npz(resolve_npz(registry, b["run_id"], b["kind"], b["name"]))

    a_label = f"{a['run_id'][:15]} / {a['kind']} / {a['name']}"
    b_label = f"{b['run_id'][:15]} / {b['kind']} / {b['name']}"

    js_vars: list[str] = []
    sections = ""
    toc: list[tuple[str, str]] = []

    # ── overview ──────────────────────────────────────────────────────────────
    dl = (
        "<dl class='kv'>"
        f"<dt>A</dt><dd>{_safe(a_label)}</dd>"
        f"<dt>B</dt><dd>{_safe(b_label)}</dd>"
        f"<dt>sources</dt><dd>{_safe(', '.join(sources) or '(none)')}</dd>"
        "</dl>"
    )
    toc.append(("Pair overview", "overview"))
    sections += _section("Pair overview", "overview", dl)

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics_html, metrics_js = _build_metrics_block(ta, tb, sources)
    js_vars.extend(metrics_js)
    toc.append(("Per-source comparison metrics", "metrics"))
    sections += _section("Per-source comparison metrics", "metrics", metrics_html)

    # ── convergence — per source ──────────────────────────────────────────────
    conv_a_html, conv_a_js = _build_convergence_block(ta, CONV_SOURCES, prefix="A_")
    conv_b_html, conv_b_js = _build_convergence_block(tb, CONV_SOURCES, prefix="B_")
    js_vars.extend(conv_a_js)
    js_vars.extend(conv_b_js)
    toc.append(("Residual stream convergence", "convergence"))
    sections += _section(
        "Residual stream convergence",
        "convergence",
        f"<h3>A: {_safe(a_label)}</h3>" + conv_a_html
        + f"<h3>B: {_safe(b_label)}</h3>" + conv_b_html,
        accent="#16a34a",
    )

    # ── advanced (PCA) — per source ───────────────────────────────────────────
    if sources:
        try:
            adv = {
                "svd": compute.svd_analysis([ta, tb], sources),
                "pca": compute.pca_analysis([ta, tb], sources),
            }
            pca_html, pca_js = _build_pca_block(adv, sources, [a_label, b_label])
            js_vars.extend(pca_js)
            toc.append(("PCA layer trajectories", "pca"))
            sections += _section("PCA layer trajectories", "pca", pca_html, accent="#2563eb")

            spec_html, spec_js = _build_spectral_block(adv, sources, [a_label, b_label])
            js_vars.extend(spec_js)
            toc.append(("Spectral analysis", "spectral"))
            sections += _section("Spectral analysis", "spectral", spec_html)
        except Exception:
            pass

    # ── heatmaps — all variants ───────────────────────────────────────────────
    for hm_source, hm_label in [("attention", "Attention softmax"), ("qk", "Raw q·k scores"), ("qkv", "Stacked qkv_last")]:
        grid_html = "<div class='img-grid'>"
        any_img = False
        for variant in HEATMAP_VARIANTS:
            png = _pair_heatmap_png(ta, tb, hm_source, variant)
            if png:
                grid_html += _img_embed(_b64png(png), VARIANT_LABELS[variant], width=250)
                any_img = True
        grid_html += "</div>"
        if any_img:
            toc.append((f"Heatmaps — {hm_label}", f"heatmap-{hm_source}"))
            sections += _section(
                f"Heatmaps — {hm_label} (all layers stacked, all variants)",
                f"heatmap-{hm_source}",
                grid_html,
            )

    subtitle = f"A: {a_label}  ↔  B: {b_label}"
    return _full_page("Activation Lab — Pair comparison report", subtitle, toc, sections, js_vars)


def generate_multi_report(registry: RunRegistry, refs: list[dict], sources: list[str]) -> str:
    from .loader import resolve_npz
    loaded = [load_npz(resolve_npz(registry, r["run_id"], r["kind"], r["name"])) for r in refs]
    ref_labels = [f"{r['run_id'][:15]} / {r['kind']} / {r['name']}" for r in refs]

    js_vars: list[str] = []
    sections = ""
    toc: list[tuple[str, str]] = []

    # ── overview ──────────────────────────────────────────────────────────────
    ol = "<ol>" + "".join(f"<li>{_safe(lbl)}</li>" for lbl in ref_labels) + "</ol>"
    toc.append(("Multi-compare overview", "overview"))
    sections += _section(
        "Multi-compare overview", "overview",
        ol + f"<p class='muted'>Sources: {_safe(', '.join(sources) or '(none)')}</p>",
    )

    # ── attention entropy — all refs ──────────────────────────────────────────
    ent_series = []
    ent_xs = []
    for i, t in enumerate(loaded):
        if not compute.has_attention(t):
            continue
        vals = compute.attention_entropy(t)
        layers = compute.layers_available(t)
        if not ent_xs:
            ent_xs = layers
        ent_series.append({"label": ref_labels[i], "data": vals, "stroke": PAIR_COLORS[i % len(PAIR_COLORS)]})
    if ent_series:
        js_vars.append(_js_var(
            "DATA_entropy_all",
            {"xs": ent_xs, "series": ent_series},
            "Attention entropy per layer — all refs overlaid\n"
            "xs – layer indices\n"
            "series[i].data – per-layer mean attention entropy in nats for ref i;\n"
            "  higher = more diffuse attention; dips that align across refs indicate\n"
            "  layers where the entire group sharpens attention together",
        ))
        toc.append(("Attention entropy", "entropy"))
        sections += _section(
            "Attention entropy per layer",
            "entropy",
            _canvas_chart("DATA_entropy_all"),
        )

    # ── convergence — per convergence source ─────────────────────────────────
    conv_html = ""
    for src in CONV_SOURCES:
        src_html = ""
        for i, t in enumerate(loaded):
            try:
                conv = compute.residual_convergence(t, source=src)
            except Exception:
                continue
            if not conv["adj_layers"]:
                continue
            var_name = f"DATA_conv_multi_{src}_{i}"
            js_vars.append(_js_var(
                var_name,
                {"xs": conv["adj_layers"], "series": _chart_data_convergence(conv, src)["series"]},
                f"Residual stream convergence — ref {i} ({ref_labels[i]}) — source: {src}\n"
                "series[0/1] adj cosine last/mean – adjacent-layer cosine similarity at last token / mean\n"
                "series[2/3] delta norm last/mean – relative update norm at last token / mean",
            ))
            edl = conv["effective_depth_last"]
            edm = conv["effective_depth_mean"]
            badge = lambda v: (f'<span class="depth-badge">{v}</span>' if v is not None else '<span class="depth-badge none">n/a</span>')
            ref_html = (
                f"<b>{_safe(ref_labels[i])}</b> — "
                f"eff.depth last: {badge(edl)}, mean: {badge(edm)}"
                + _canvas_chart(var_name, height=180)
            )
            src_html += f"<div style='margin-bottom:16px'>{ref_html}</div>"
        if src_html:
            conv_html += _subsection(f"Source: {src}", f"conv-multi-{src}", src_html)
    if conv_html:
        toc.append(("Residual stream convergence", "convergence"))
        sections += _section("Residual stream convergence", "convergence", conv_html, accent="#16a34a")

    # ── group analysis — per source ───────────────────────────────────────────
    if sources and len(loaded) >= 2:
        try:
            grp = compute.group_analysis(loaded, sources)
            grp_html, grp_js = _build_group_block(grp, sources, ref_labels, prefix="multi_")
            js_vars.extend(grp_js)
            toc.append(("Group analysis", "group"))
            sections += _section("Group analysis", "group", grp_html, accent="#a855f7")
        except Exception:
            pass

    # ── PCA + spectral — per source ───────────────────────────────────────────
    if sources and len(loaded) >= 2:
        try:
            adv = {
                "svd": compute.svd_analysis(loaded, sources),
                "pca": compute.pca_analysis(loaded, sources),
            }
            pca_html, pca_js = _build_pca_block(adv, sources, ref_labels, prefix="multi_")
            js_vars.extend(pca_js)
            toc.append(("PCA layer trajectories", "pca"))
            sections += _section("PCA layer trajectories", "pca", pca_html, accent="#2563eb")

            spec_html, spec_js = _build_spectral_block(adv, sources, ref_labels, prefix="multi_")
            js_vars.extend(spec_js)
            toc.append(("Spectral analysis", "spectral"))
            sections += _section("Spectral analysis", "spectral", spec_html)
        except Exception:
            pass

    subtitle = f"{len(refs)} refs · sources: {', '.join(sources) or 'none'}"
    return _full_page("Activation Lab — Multi-compare report", subtitle, toc, sections, js_vars)
