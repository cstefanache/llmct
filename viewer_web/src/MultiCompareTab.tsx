import { useEffect, useMemo, useState } from "react";
import {
  NpzRef,
  MetricsResponse,
  AdvancedMetricsResponse,
  PcaSourceResult,
  CkaSourceResult,
  SvdSourceResult,
  compareMetrics,
  compareAdvanced,
} from "./api";
import { LineChart, LineSeries } from "./LineChart";
import { makeRefLabels } from "./refLabels";

const PAIR_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316",
  "#0891b2", "#ca8a04", "#ec4899", "#475569", "#84cc16",
];

const ALL_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out", "qkv_last"] as const;

// ─── small helpers ────────────────────────────────────────────────────────────

const avg = (a: (number | null)[]) => {
  const v = a.filter((x): x is number => x !== null);
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : 0;
};
const mn = (a: number[]) => (a.length ? Math.min(...a) : 0);

// ─── PCA scatter (SVG) ────────────────────────────────────────────────────────

function PcaScatter({
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

  // group by ref, sorted by layer order
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
        {/* axes */}
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
              {pts.length > 0 && (
                <text
                  x={tx(pts[pts.length - 1].x) - 150}
                  y={ty(pts[pts.length - 1].y) - 10}
                  fill={color} fontSize={10} fontWeight={600}
                >
                  {refLabels[ri]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="muted" style={{ textAlign: "center", fontSize: 11, marginTop: 2 }}>
        Each point = one layer's mean activation. Opacity: light = early layer → dark = late.
      </div>
    </div>
  );
}

// ─── CKA matrix (colored table) ──────────────────────────────────────────────

function CkaMatrix({
  cka, refLabels,
}: {
  cka: CkaSourceResult;
  refLabels: string[];
}) {
  const n = refLabels.length;

  // Layer-average the N×N matrices
  const sumMat = Array.from({ length: n }, () => new Array(n).fill(0));
  const cntMat = Array.from({ length: n }, () => new Array(n).fill(0));
  cka.matrices.forEach((mat) => {
    mat.forEach((row, i) => {
      row.forEach((v, j) => {
        if (v !== null && isFinite(v)) {
          sumMat[i][j] += v;
          cntMat[i][j]++;
        }
      });
    });
  });

  const allNull = cntMat.every((row) => row.every((c) => c === 0));
  if (allNull) {
    return (
      <div className="muted" style={{ fontSize: 12 }}>
        CKA requires T &gt; 1 per layer — single-token reference captures are not supported.
      </div>
    );
  }

  const mat = sumMat.map((row, i) =>
    row.map((s, j) => (cntMat[i][j] > 0 ? s / cntMat[i][j] : null)),
  );

  // blue(0) → white(0.5) → red(1)
  const cellBg = (v: number | null): string => {
    if (v === null) return "#f5f5f5";
    const t = Math.max(0, Math.min(1, v));
    if (t <= 0.5) {
      const u = t * 2;
      return `rgb(${Math.round(70 + 185 * u)},${Math.round(130 + 125 * u)},${Math.round(220 - 20 * u)})`;
    }
    const u = (t - 0.5) * 2;
    return `rgb(255,${Math.round(255 - 185 * u)},${Math.round(200 - 130 * u)})`;
  };
  const textColor = (v: number | null) =>
    v !== null && Math.abs(v - 0.5) > 0.3 ? "#fff" : "#333";

  return (
    <table className="metrics-table" style={{ width: "auto" }}>
      <thead>
        <tr>
          <th></th>
          {refLabels.map((l, j) => (
            <th key={j} style={{ fontSize: 10, maxWidth: 90, wordBreak: "break-all" }}>{l}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {mat.map((row, i) => (
          <tr key={i}>
            <td style={{ fontSize: 10 }}>{refLabels[i]}</td>
            {row.map((v, j) => (
              <td
                key={j}
                style={{
                  background: cellBg(v),
                  color: textColor(v),
                  textAlign: "center",
                  fontSize: 11,
                  minWidth: 56,
                }}
              >
                {v !== null ? v.toFixed(3) : "—"}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
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
        <h4>Spectral norm (‖h‖₂) per layer — representation magnitude</h4>
        <LineChart xs={xs} series={mkSeries("spectral_norm")} />
      </div>
      <div className="section">
        <h4>Nuclear norm (Σσᵢ) per layer</h4>
        <LineChart xs={xs} series={mkSeries("nuclear_norm")} />
      </div>
      {hasEffRank && (
        <div className="section">
          <h4>Effective rank per layer</h4>
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
  const [resp, setResp] = useState<MetricsResponse | null>(null);
  const [adv, setAdv] = useState<AdvancedMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [advError, setAdvError] = useState<string | null>(null);
  const [source, setSource] = useState<string>(sources[0] ?? "hidden_out");

  const refsKey = useMemo(
    () => refs.map((r) => `${r.run_id}|${r.kind}|${r.name}`).join("::"),
    [refs],
  );
  const sourcesKey = sources.join(",");

  useEffect(() => {
    setError(null);
    setResp(null);
    if (sources.length === 0 || refs.length < 2) return;
    compareMetrics(refs, sources).then(setResp).catch((e) => setError(String(e)));
  }, [refsKey, sourcesKey]);

  useEffect(() => {
    setAdvError(null);
    setAdv(null);
    if (sources.length === 0 || refs.length < 2) return;
    compareAdvanced(refs, sources).then(setAdv).catch((e) => setAdvError(String(e)));
  }, [refsKey, sourcesKey]);

  useEffect(() => {
    if (sources.length === 0) return;
    if (!sources.includes(source)) setSource(sources[0]);
  }, [sourcesKey]);

  if (refs.length < 2) return <div className="empty">select at least 2 npz from the sidebar</div>;
  if (sources.length === 0) return <div className="empty">select at least one source in the header</div>;

  const refLabels = makeRefLabels(refs);
  const pairLabel = (i: number, j: number) => `${refLabels[i]} ↔ ${refLabels[j]}`;

  return (
    <div>
      <div className="controls">
        <span className="muted"><b>Refs ({refs.length}):</b> {refLabels.join(", ")}</span>
      </div>

      <div className="controls">
        <label>
          Source:
          <select value={source} onChange={(e) => setSource(e.target.value)} style={{ marginLeft: 4 }}>
            {ALL_SOURCES.filter((s) => sources.includes(s)).map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>

      {/* ── pair comparison ─────────────────────────────────── */}
      {error && <div className="err">{error}</div>}
      {!resp && !error && <div className="empty">loading pair metrics…</div>}
      {resp && (() => {
        const xs = (resp.pairs[0]?.metrics[source]?.layers ?? []) as number[];
        const mkSeries = (
          metric: "mean_cos" | "mean_mae" | "std_diff" | "mean_overlap",
        ): LineSeries[] =>
          resp.pairs
            .map((p, idx) => {
              const m = p.metrics[source];
              if (!m) return null;
              return {
                label: pairLabel(p.i, p.j),
                data: m[metric] as number[],
                stroke: PAIR_COLORS[idx % PAIR_COLORS.length],
              };
            })
            .filter((x): x is LineSeries => !!x);

        return (
          <>
            <div className="section">
              <h4>Per-pair summary — averaged over layers (source: {source})</h4>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>pair</th>
                    <th>mean cos</th>
                    <th>min cos</th>
                    <th>mean MAE</th>
                    <th>std diff</th>
                    <th>overlap</th>
                  </tr>
                </thead>
                <tbody>
                  {resp.pairs.map((p) => {
                    const m = p.metrics[source];
                    if (!m) return (
                      <tr key={`${p.i}-${p.j}`}>
                        <td>{pairLabel(p.i, p.j)}</td>
                        <td colSpan={5} className="muted">no data</td>
                      </tr>
                    );
                    return (
                      <tr key={`${p.i}-${p.j}`}>
                        <td>{pairLabel(p.i, p.j)}</td>
                        <td>{avg(m.mean_cos).toFixed(4)}</td>
                        <td>{mn(m.min_cos).toFixed(4)}</td>
                        <td>{avg(m.mean_mae).toExponential(3)}</td>
                        <td>{avg(m.std_diff).toExponential(3)}</td>
                        <td>{avg(m.mean_overlap).toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="section"><h4>Cosine similarity per layer</h4>
              <LineChart xs={xs} series={mkSeries("mean_cos")} />
            </div>
            <div className="section"><h4>MAE per layer</h4>
              <LineChart xs={xs} series={mkSeries("mean_mae")} />
            </div>
            <div className="section"><h4>STD of difference per layer</h4>
              <LineChart xs={xs} series={mkSeries("std_diff")} />
            </div>
            <div className="section"><h4>Top-1% activation overlap (Jaccard) per layer</h4>
              <LineChart xs={xs} series={mkSeries("mean_overlap")} />
            </div>
          </>
        );
      })()}

      {/* ── advanced analysis ───────────────────────────────── */}
      <div className="section" style={{ borderTop: "2px solid #2563eb", marginTop: 24 }}>
        <h4 style={{ color: "#2563eb" }}>Advanced Analysis (source: {source})</h4>

        {advError && <div className="err">{advError}</div>}
        {!adv && !advError && <div className="muted">computing advanced metrics…</div>}

        {adv && (
          <>
            <h4 style={{ marginTop: 16 }}>PCA Layer Trajectories</h4>
            {adv.pca[source]
              ? <PcaScatter data={adv.pca[source]} refLabels={refLabels} />
              : <div className="muted">no PCA data for {source}</div>}

            <h4 style={{ marginTop: 20 }}>Linear CKA Similarity Matrix (layer-averaged)</h4>
            <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>
              Linear CKA measures representational similarity invariant to orthogonal transforms
              and isotropic scaling. 1 = identical geometry, 0 = orthogonal.
            </div>
            {adv.cka[source]
              ? <CkaMatrix cka={adv.cka[source]} refLabels={refLabels} />
              : <div className="muted">no CKA data for {source}</div>}
          </>
        )}
      </div>

      {/* spectral section is separate so it can show even if PCA/CKA are loading */}
      {adv && adv.svd[source] && (
        <SpectralCharts svd={adv.svd[source]} refLabels={refLabels} />
      )}
    </div>
  );
}
