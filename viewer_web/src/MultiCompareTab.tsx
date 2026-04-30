import { useEffect, useMemo, useState } from "react";
import {
  NpzRef,
  AdvancedMetricsResponse,
  GroupAnalysisResponse,
  PcaSourceResult,
  SvdSourceResult,
  EntropyResponse,
  ConvergenceResponse,
  compareAdvanced,
  compareGroup,
  getEntropy,
  getNpzConvergence,
  openReport,
} from "./api";
import { LineChart, LineSeries } from "./LineChart";
import { makeRefLabels } from "./refLabels";

const PAIR_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316",
  "#0891b2", "#ca8a04", "#ec4899", "#475569", "#84cc16",
];

const ALL_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out", "qkv_last"] as const;
const CONVERGENCE_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"] as const;

// ─── shared explanation block ────────────────────────────────────────────────

function Explain({ what, how }: { what: string; how: string }) {
  return (
    <div className="muted" style={{ fontSize: 11, marginBottom: 8, lineHeight: 1.5 }}>
      <div><b>What:</b> {what}</div>
      <div><b>How to read:</b> {how}</div>
    </div>
  );
}

// ─── PCA scatter (SVG) ────────────────────────────────────────────────────────

export function PcaScatter({
  data, refLabels,
}: {
  data: PcaSourceResult;
  refLabels: string[];
}) {
  const W = 500, H = 380, PAD = 44;
  const { coords, ref_indices, layer_indices } = data;

  if (coords.length === 0) return <div className="muted">no PCA data</div>;

  const xs = coords.map((c) => c[0]);
  const ys = coords.map((c) => c[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xr = xmax - xmin || 1, yr = ymax - ymin || 1;
  const tx = (x: number) => PAD + ((x - xmin) / xr) * (W - 2 * PAD);
  const ty = (y: number) => H - PAD - ((y - ymin) / yr) * (H - 2 * PAD);

  const byRef: { x: number; y: number; li: number }[][] = refLabels.map(() => []);
  coords.forEach(([x, y], idx) => {
    byRef[ref_indices[idx]].push({ x, y, li: layer_indices[idx] });
  });
  byRef.forEach((pts) => pts.sort((a, b) => a.li - b.li));

  const maxLi = Math.max(...layer_indices, 1);

  return (
    <div>
      <svg
        width={W} height={H}
        style={{ display: "block", maxWidth: "100%", fontSize: 10 }}
        viewBox={`0 0 ${W} ${H}`}
      >
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#ddd" strokeWidth={1} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#ddd" strokeWidth={1} />
        <text x={W / 2} y={H - 6} textAnchor="middle" fill="#999" fontSize={10}>
          PC1 ({(data.explained_variance[0] * 100).toFixed(1)}%)
        </text>
        <text
          x={10} y={H / 2} textAnchor="middle" fill="#999" fontSize={10}
          transform={`rotate(-90, 10, ${H / 2})`}
        >
          PC2 ({(data.explained_variance[1] * 100).toFixed(1)}%)
        </text>

        {byRef.map((pts, ri) => {
          const color = PAIR_COLORS[ri % PAIR_COLORS.length];
          return (
            <g key={ri}>
              {pts.map((p, i) =>
                i > 0 ? (
                  <line key={i}
                    x1={tx(pts[i - 1].x)} y1={ty(pts[i - 1].y)}
                    x2={tx(p.x)} y2={ty(p.y)}
                    stroke={color} strokeWidth={1} strokeOpacity={0.3}
                  />
                ) : null,
              )}
              {pts.map((p, i) => (
                <circle key={i}
                  cx={tx(p.x)} cy={ty(p.y)} r={3.5}
                  fill={color}
                  opacity={0.25 + 0.75 * (p.li / maxLi)}
                >
                  <title>layer {data.layers[p.li]}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px", marginTop: 8, paddingLeft: PAD }}>
        {refLabels.map((label, ri) => {
          const color = PAIR_COLORS[ri % PAIR_COLORS.length];
          return (
            <div key={ri} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11 }}>
              <svg width={24} height={12} style={{ flexShrink: 0 }}>
                <line x1={0} y1={6} x2={16} y2={6} stroke={color} strokeWidth={1.5} strokeOpacity={0.5} />
                <circle cx={16} cy={6} r={3.5} fill={color} />
              </svg>
              <span style={{ color, fontWeight: 600 }}>{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── reusable NxN heatmap ────────────────────────────────────────────────────

function MatrixHeatmap({
  matrix, labels, scheme = "diverging", domain,
}: {
  matrix: (number | null)[][];
  labels: string[];
  scheme?: "diverging" | "sequential";
  domain?: [number, number];
}) {
  const n = labels.length;
  const flat = matrix.flat().filter((v): v is number => v !== null && isFinite(v));
  if (flat.length === 0) {
    return <div className="muted">no data</div>;
  }
  const lo = domain?.[0] ?? Math.min(...flat);
  const hi = domain?.[1] ?? Math.max(...flat);
  const range = hi - lo || 1;

  const cellBg = (v: number | null): string => {
    if (v === null || !isFinite(v)) return "#f5f5f5";
    const t = Math.max(0, Math.min(1, (v - lo) / range));
    if (scheme === "sequential") {
      // white → blue
      return `rgb(${Math.round(255 - 195 * t)},${Math.round(255 - 125 * t)},${Math.round(255 - 35 * t)})`;
    }
    // diverging: blue (lo) → white (mid) → red (hi)
    if (t <= 0.5) {
      const u = t * 2;
      return `rgb(${Math.round(70 + 185 * u)},${Math.round(130 + 125 * u)},${Math.round(220 - 20 * u)})`;
    }
    const u = (t - 0.5) * 2;
    return `rgb(255,${Math.round(255 - 185 * u)},${Math.round(200 - 130 * u)})`;
  };
  const txt = (v: number | null) => {
    if (v === null || !isFinite(v)) return "#aaa";
    const t = Math.max(0, Math.min(1, (v - lo) / range));
    return scheme === "sequential"
      ? (t > 0.55 ? "#fff" : "#333")
      : (Math.abs(t - 0.5) > 0.3 ? "#fff" : "#333");
  };

  return (
    <table className="metrics-table" style={{ width: "auto" }}>
      <thead>
        <tr>
          <th></th>
          {labels.map((l, j) => (
            <th key={j} style={{ fontSize: 10, maxWidth: 90, wordBreak: "break-all" }}>{l}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <td style={{ fontSize: 10 }}>{labels[i]}</td>
            {row.slice(0, n).map((v, j) => (
              <td
                key={j}
                style={{
                  background: cellBg(v),
                  color: txt(v),
                  textAlign: "center",
                  fontSize: 11,
                  minWidth: 56,
                }}
              >
                {v !== null && isFinite(v) ? v.toFixed(3) : "—"}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─── dendrogram (SVG) ───────────────────────────────────────────────────────

function Dendrogram({
  linkage, labels,
}: {
  linkage: [number, number, number, number][];
  labels: string[];
}) {
  const n = labels.length;
  if (linkage.length < n - 1 || n < 2) {
    return <div className="muted">not enough data to cluster</div>;
  }

  // Members of every cluster (leaves + internals).
  const members: number[][] = [];
  for (let i = 0; i < n; i++) members.push([i]);
  for (const [a, b] of linkage) {
    const ai = Math.round(a), bi = Math.round(b);
    members.push([...members[ai], ...members[bi]]);
  }

  // Leaf order: take the deepest member list and use it as ordering.
  const leafOrder = members[members.length - 1];

  // Cluster heights (0 for leaves, dist for internals).
  const heights = new Array<number>(n + linkage.length).fill(0);
  for (let k = 0; k < linkage.length; k++) heights[n + k] = linkage[k][2];

  // x position for each cluster id = mean leaf position over its members.
  const xPosOfLeaf = new Map<number, number>();
  leafOrder.forEach((leaf, idx) => xPosOfLeaf.set(leaf, idx));
  const xPos = (cid: number): number =>
    members[cid].reduce((s, m) => s + (xPosOfLeaf.get(m) ?? 0), 0) / members[cid].length;

  const W = Math.max(360, n * 84);
  const H = 220;
  const PAD_L = 40;
  const PAD_R = 12;
  const PAD_T = 12;
  const PAD_B = 90;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const maxH = Math.max(...heights, 1e-9);
  const tx = (i: number) => PAD_L + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const ty = (h: number) => PAD_T + innerH - (h / maxH) * innerH;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block", maxWidth: "100%" }}>
      {/* y-axis */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + innerH} stroke="#ccc" strokeWidth={1} />
      <text x={4} y={PAD_T + 4} fontSize={10} fill="#999">{maxH.toFixed(2)}</text>
      <text x={4} y={PAD_T + innerH} fontSize={10} fill="#999">0</text>
      <text
        x={10} y={PAD_T + innerH / 2} fontSize={10} fill="#999" textAnchor="middle"
        transform={`rotate(-90, 10, ${PAD_T + innerH / 2})`}
      >
        distance (1 − cos)
      </text>

      {/* merges */}
      {linkage.map(([a, b, dist], k) => {
        const ai = Math.round(a), bi = Math.round(b);
        const xa = tx(xPos(ai));
        const xb = tx(xPos(bi));
        const ya = ty(heights[ai]);
        const yb = ty(heights[bi]);
        const yh = ty(dist);
        return (
          <g key={k}>
            <line x1={xa} y1={ya} x2={xa} y2={yh} stroke="#374151" strokeWidth={1.4} />
            <line x1={xb} y1={yb} x2={xb} y2={yh} stroke="#374151" strokeWidth={1.4} />
            <line x1={xa} y1={yh} x2={xb} y2={yh} stroke="#374151" strokeWidth={1.4} />
          </g>
        );
      })}

      {/* leaf labels */}
      {leafOrder.map((leaf, i) => {
        const x = tx(i);
        const y = PAD_T + innerH;
        return (
          <g key={leaf}>
            <circle cx={x} cy={y} r={3.5} fill={PAIR_COLORS[leaf % PAIR_COLORS.length]} />
            <text
              x={x} y={y + 8} fontSize={10}
              fill={PAIR_COLORS[leaf % PAIR_COLORS.length]}
              textAnchor="end"
              transform={`rotate(-45, ${x}, ${y + 8})`}
            >
              {labels[leaf]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── PCA distance-to-centroid (computed from existing PCA coords) ─────────

function pcaDistanceToCentroidSeries(
  pca: PcaSourceResult, refLabels: string[],
): { xs: number[]; series: LineSeries[] } {
  const L = pca.layers.length;
  const N = refLabels.length;
  // collect coords by [layer][snapshot]
  const grid: ([number, number] | null)[][] =
    Array.from({ length: L }, () => Array(N).fill(null));
  pca.coords.forEach((c, idx) => {
    grid[pca.layer_indices[idx]][pca.ref_indices[idx]] = c;
  });
  const data: number[][] = Array.from({ length: N }, () => []);
  for (let li = 0; li < L; li++) {
    const present = grid[li].filter((c): c is [number, number] => c !== null);
    if (present.length === 0) {
      for (let n = 0; n < N; n++) data[n].push(NaN);
      continue;
    }
    const cx = present.reduce((s, c) => s + c[0], 0) / present.length;
    const cy = present.reduce((s, c) => s + c[1], 0) / present.length;
    for (let n = 0; n < N; n++) {
      const c = grid[li][n];
      if (c === null) data[n].push(NaN);
      else data[n].push(Math.hypot(c[0] - cx, c[1] - cy));
    }
  }
  return {
    xs: pca.layers,
    series: data.map((d, i) => ({
      label: refLabels[i],
      data: d,
      stroke: PAIR_COLORS[i % PAIR_COLORS.length],
    })),
  };
}

// ─── spectral charts ──────────────────────────────────────────────────────────

function SpectralCharts({
  svd, refLabels,
}: {
  svd: SvdSourceResult;
  refLabels: string[];
}) {
  const xs = svd.layers;
  const hasEffRank = svd.refs.some((r) => r.effective_rank.some((v) => v !== null));

  const mkSeries = (
    field: "spectral_norm" | "nuclear_norm" | "effective_rank",
  ): LineSeries[] =>
    svd.refs.map((r, ri) => ({
      label: refLabels[ri],
      data: r[field].map((v) => v ?? NaN) as number[],
      stroke: PAIR_COLORS[ri % PAIR_COLORS.length],
    }));

  return (
    <>
      <div className="section">
        <h4>Spectral norm (‖h‖₂) per layer</h4>
        <Explain
          what="Largest singular value of each snapshot's per-layer activation matrix — i.e. the magnitude of its dominant direction. For T=1 captures it reduces to the L2 norm."
          how="Higher = stronger overall activation at that layer. Snapshots whose lines track each other have similar magnitude profiles; sudden gaps mark layers where one snapshot 'fires harder' than another."
        />
        <LineChart xs={xs} series={mkSeries("spectral_norm")} />
      </div>
      <div className="section">
        <h4>Nuclear norm (Σσᵢ) per layer</h4>
        <Explain
          what="Sum of singular values — total energy spread across all directions of the activation matrix."
          how="High nuclear with low spectral norm = energy spread across many features (rich representation). High spectral with low nuclear = concentrated on a single mode."
        />
        <LineChart xs={xs} series={mkSeries("nuclear_norm")} />
      </div>
      {hasEffRank && (
        <div className="section">
          <h4>Effective rank per layer</h4>
          <Explain
            what="exp(entropy of normalized singular values). A continuous count of 'how many directions are actually used' at that layer."
            how="Rising curves = layer engages more independent features; flat low values = activations collapse onto a few directions. Compare snapshots to spot where some collapse and others don't."
          />
          <LineChart xs={xs} series={mkSeries("effective_rank")} />
        </div>
      )}
    </>
  );
}

// ─── main tab ────────────────────────────────────────────────────────────────

export function MultiCompareTab({
  refs, sources,
}: {
  refs: NpzRef[];
  sources: string[];
}) {
  const [adv, setAdv] = useState<AdvancedMetricsResponse | null>(null);
  const [advError, setAdvError] = useState<string | null>(null);
  const [group, setGroup] = useState<GroupAnalysisResponse | null>(null);
  const [groupError, setGroupError] = useState<string | null>(null);
  const [source, setSource] = useState<string>(sources[0] ?? "hidden_out");
  const [entropyData, setEntropyData] = useState<(EntropyResponse | null)[]>([]);
  const [convData, setConvData] = useState<(ConvergenceResponse | null)[]>([]);
  const [convSource, setConvSource] = useState<string>("hidden_out");
  const handleDownload = () => openReport({ kind: "multi", refs, sources });

  const refsKey = useMemo(
    () => refs.map((r) => `${r.run_id}|${r.kind}|${r.name}`).join("::"),
    [refs],
  );
  const sourcesKey = sources.join(",");

  useEffect(() => {
    setAdvError(null); setAdv(null);
    setGroupError(null); setGroup(null);
    if (sources.length === 0 || refs.length < 2) return;
    compareAdvanced(refs, sources).then(setAdv).catch((e) => setAdvError(String(e)));
    compareGroup(refs, sources).then(setGroup).catch((e) => setGroupError(String(e)));
  }, [refsKey, sourcesKey]);

  useEffect(() => {
    if (sources.length === 0) return;
    if (!sources.includes(source)) setSource(sources[0]);
  }, [sourcesKey]);

  useEffect(() => {
    setEntropyData(refs.map(() => null));
    refs.forEach((ref, i) => {
      getEntropy(ref)
        .then((e) =>
          setEntropyData((prev) => {
            const next = [...prev]; next[i] = e; return next;
          }),
        )
        .catch(() => { /* npz lacks attention */ });
    });
  }, [refsKey]);

  useEffect(() => {
    setConvData(refs.map(() => null));
    refs.forEach((ref, i) => {
      getNpzConvergence(ref, convSource)
        .then((c) =>
          setConvData((prev) => {
            const next = [...prev]; next[i] = c; return next;
          }),
        )
        .catch(console.error);
    });
  }, [refsKey, convSource]);

  if (refs.length < 2) return <div className="empty">select at least 2 npz from the sidebar</div>;
  if (sources.length === 0) return <div className="empty">select at least one source in the header</div>;

  const refLabels = makeRefLabels(refs);

  // ── entropy series ──
  const entropySeries: LineSeries[] = entropyData
    .map((e, i) =>
      e ? { label: refLabels[i], data: e.entropy, stroke: PAIR_COLORS[i % PAIR_COLORS.length] } : null,
    )
    .filter((s): s is LineSeries => s !== null);
  const entropyXs = entropyData.find(Boolean)?.layers ?? [];
  const entropyMissing = entropyData.length - entropySeries.length;

  // ── convergence series ──
  const convXs = convData.find(Boolean)?.adj_layers ?? [];
  const mkConvSeries = (
    field: "adj_cosine_last" | "adj_cosine_mean" | "delta_norm_last" | "delta_norm_mean",
  ): LineSeries[] =>
    convData
      .map((c, i) =>
        c ? { label: refLabels[i], data: c[field], stroke: PAIR_COLORS[i % PAIR_COLORS.length] } : null,
      )
      .filter((s): s is LineSeries => s !== null);
  const loadedConv = convData.filter(Boolean).length;

  // ── group-analysis derived series for selected source ──
  const grp = group?.[source];
  const cosCentroidSeries: LineSeries[] = grp
    ? refLabels.map((lbl, i) => ({
        label: lbl,
        data: grp.cos_to_centroid.map((row) => (row[i] === null || row[i] === undefined ? NaN : row[i] as number)),
        stroke: PAIR_COLORS[i % PAIR_COLORS.length],
      }))
    : [];
  const divergenceSeries: LineSeries[] = grp
    ? [{
        label: "1 − mean pairwise cos",
        data: grp.divergence.map((v) => (v === null ? NaN : v)),
        stroke: "#dc2626",
      }]
    : [];

  const pcaDistView = adv?.pca[source]
    ? pcaDistanceToCentroidSeries(adv.pca[source], refLabels)
    : null;

  return (
    <div>
      <div className="controls">
        <button className="btn-toggle" onClick={handleDownload}>↓ open report
        </button>
        <span className="muted"><b>Refs ({refs.length}):</b> {refLabels.join(", ")}</span>
      </div>

      <div className="controls">
        <label>
          Source (group / PCA / spectral):
          <select value={source} onChange={(e) => setSource(e.target.value)} style={{ marginLeft: 4 }}>
            {ALL_SOURCES.filter((s) => sources.includes(s)).map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>

      {/* ── attention entropy ─────────────────────────────── */}
      <div className="section">
        <h4>Attention entropy per layer</h4>
        <Explain
          what="Mean entropy (in nats) of the softmax attention distribution at each layer, one curve per snapshot."
          how="Higher = attention is spread broadly across tokens; lower = focused on a few. Curves that dip together at the same layer indicate snapshots that all sharpen attention at that depth."
        />
        {entropySeries.length > 0
          ? <LineChart xs={entropyXs} series={entropySeries} />
          : <div className="muted">no snapshot has attention weights captured</div>}
        {entropyMissing > 0 && entropySeries.length > 0 && (
          <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
            {entropyMissing} of {refs.length} snapshots have no attention weights and are omitted.
          </div>
        )}
      </div>

      {/* ── residual stream convergence ───────────────────── */}
      <div className="section" style={{ borderTop: "2px solid #16a34a", marginTop: 24 }}>
        <h4 style={{ color: "#16a34a" }}>Residual Stream Convergence — per snapshot</h4>
        <Explain
          what="How much the residual stream changes between adjacent layers, computed independently per snapshot for the selected source."
          how="Cosine close to 1 and ‖Δh‖/‖h‖ close to 0 mean the layer barely modifies the stream. Effective depth is the first layer at which cosine stays ≥ 0.99 forever — earlier = the network 'finishes thinking' sooner."
        />
        <div className="controls">
          <label>
            Source:
            <select value={convSource} onChange={(e) => setConvSource(e.target.value)} style={{ marginLeft: 4 }}>
              {CONVERGENCE_SOURCES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          {loadedConv > 0 && loadedConv < refs.length && (
            <span className="muted">{loadedConv}/{refs.length} loaded…</span>
          )}
        </div>
        {loadedConv > 0 && (
          <div className="section">
            <h5 style={{ marginTop: 8, marginBottom: 4 }}>Effective depth per snapshot</h5>
            <table className="metrics-table" style={{ width: "auto" }}>
              <thead>
                <tr>
                  <th>snapshot</th>
                  <th>eff. depth (last)</th>
                  <th>eff. depth (mean)</th>
                </tr>
              </thead>
              <tbody>
                {convData.map((c, i) => (
                  <tr key={i}>
                    <td style={{ color: PAIR_COLORS[i % PAIR_COLORS.length], fontWeight: 600 }}>
                      {refLabels[i]}
                    </td>
                    <td>{c?.effective_depth_last ?? "—"}</td>
                    <td>{c?.effective_depth_mean ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {loadedConv === 0 && <div className="muted">computing…</div>}
        {loadedConv > 0 && (
          <>
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Adjacent-layer cosine — last token</h5>
            <LineChart xs={convXs} series={mkConvSeries("adj_cosine_last")} height={220} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Adjacent-layer cosine — mean over sequence</h5>
            <LineChart xs={convXs} series={mkConvSeries("adj_cosine_mean")} height={220} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ — last token</h5>
            <LineChart xs={convXs} series={mkConvSeries("delta_norm_last")} height={220} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ — mean over sequence</h5>
            <LineChart xs={convXs} series={mkConvSeries("delta_norm_mean")} height={220} />
          </>
        )}
      </div>

      {/* ── group analysis ─────────────────────────────────── */}
      <div className="section" style={{ borderTop: "2px solid #a855f7", marginTop: 24 }}>
        <h4 style={{ color: "#a855f7" }}>Group analysis — matching patterns across snapshots (source: {source})</h4>
        {groupError && <div className="err">{groupError}</div>}
        {!group && !groupError && <div className="muted">computing group metrics…</div>}

        {grp && (
          <>
            {/* cosine to centroid */}
            <div className="section">
              <h5>Cosine to group centroid per layer</h5>
              <Explain
                what="At each layer we compute the mean activation across all snapshots (the 'group centroid') and plot every snapshot's cosine similarity to that centroid."
                how="Curves near 1.0 = snapshot agrees with the group at that layer. A line that dips at a specific layer flags where that snapshot diverges from the consensus. Similar curves = matching pattern; outlier line = the odd one out."
              />
              <LineChart xs={grp.layers} series={cosCentroidSeries} height={260} />
            </div>

            {/* layer divergence */}
            <div className="section">
              <h5>Layer divergence: 1 − mean pairwise cosine</h5>
              <Explain
                what="At each layer, average pairwise cosine across all N(N−1)/2 snapshot pairs, then plot 1 minus that. A scalar 'how much the group disagrees' per depth."
                how="Near 0 = all snapshots agree at this layer. Spikes mark depths where snapshots diverge most — a useful pointer to where in the network the differences live."
              />
              <LineChart xs={grp.layers} series={divergenceSeries} height={220} />
            </div>

            {/* all-pairs cosine matrix */}
            <div className="section">
              <h5>All-pairs cosine similarity matrix (layer-averaged)</h5>
              <Explain
                what="N×N heatmap of pairwise cosine between snapshots' per-layer mean activations, averaged over layers."
                how="Bright-red blocks along the diagonal = clusters of snapshots that match each other. Cool-blue cells = pairs whose representations disagree on average. Compare blocks to spot groups."
              />
              <MatrixHeatmap matrix={grp.pairwise_cos} labels={refLabels} domain={[-1, 1]} />
            </div>

            {/* dendrogram */}
            <div className="section">
              <h5>Hierarchical clustering (average-link, distance = 1 − cos)</h5>
              <Explain
                what="Average-linkage agglomerative clustering on the all-pairs cosine distance matrix above."
                how="Read bottom-up: snapshots that join early (low height) are most similar; later joins indicate weaker grouping. Y-axis = distance at which the merge happens."
              />
              <Dendrogram linkage={grp.linkage} labels={refLabels} />
            </div>

            {/* top-k Jaccard matrix */}
            <div className="section">
              <h5>Top-1% channel Jaccard matrix (layer-averaged)</h5>
              <Explain
                what="For each layer, take each snapshot's top 1% channels by |activation|, compute pairwise Jaccard of those index sets, then average across layers."
                how="Complementary to cosine: catches snapshots that are 'driven by the same neurons' even when their magnitudes differ. High values where cosine is also high = robust matching pattern. High Jaccard but low cosine = same active channels with opposite signs/magnitudes."
              />
              <MatrixHeatmap matrix={grp.topk_jaccard} labels={refLabels} scheme="sequential" domain={[0, 1]} />
            </div>
          </>
        )}
      </div>

      {/* ── PCA trajectories ──────────────────────────────────── */}
      <div className="section" style={{ borderTop: "2px solid #2563eb", marginTop: 24 }}>
        <h4 style={{ color: "#2563eb" }}>PCA Layer Trajectories (source: {source})</h4>
        <Explain
          what="2D projection of every (snapshot, layer) mean activation onto the top-2 principal components computed jointly across all snapshots."
          how="Each colored path is one snapshot moving through layers; opacity goes from light (early layer) to dark (late). Trajectories that overlay each other = snapshots evolve through similar representations; paths that fan out late = snapshots agree early but diverge in deeper layers."
        />
        {advError && <div className="err">{advError}</div>}
        {!adv && !advError && <div className="muted">computing PCA…</div>}
        {adv && (
          adv.pca[source]
            ? <PcaScatter data={adv.pca[source]} refLabels={refLabels} />
            : <div className="muted">no PCA data for {source}</div>
        )}
      </div>

      {/* ── PCA distance-to-centroid ──────────────────────────── */}
      {pcaDistView && (
        <div className="section">
          <h5>Distance to PCA centroid per layer</h5>
          <Explain
            what="Per layer, distance in PC1–PC2 space from each snapshot's mean activation to the layer's centroid (mean of all snapshots)."
            how="Quantifies what the scatter shows visually. A line near 0 = snapshot sits on top of the consensus at that layer; a spike marks the depth at which it strays. Easier to compare than reading distances off the 2D plot."
          />
          <LineChart xs={pcaDistView.xs} series={pcaDistView.series} height={220} />
        </div>
      )}

      {/* ── spectral charts ───────────────────────────────────── */}
      {adv && adv.svd[source] && (
        <SpectralCharts svd={adv.svd[source]} refLabels={refLabels} />
      )}
    </div>
  );
}
