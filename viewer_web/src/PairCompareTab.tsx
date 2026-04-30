import { useEffect, useState } from "react";
import {
  NpzRef, NpzMeta, PairMetrics, AdvancedMetricsResponse, ConvergenceResponse,
  compareMetrics, comparePairHeatmap, compareAdvanced, getNpzMeta, getNpzConvergence,
  openReport,
} from "./api";
import { LineChart, LineSeries } from "./LineChart";
import { PcaScatter } from "./MultiCompareTab";

const CONVERGENCE_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"];

const SOURCE_COLORS: Record<string, string> = {
  hidden_in: "#2563eb",
  hidden_out: "#16a34a",
  attn_out: "#dc2626",
  mlp_down_out: "#a855f7",
  qkv_last: "#f97316",
};

const VARIANTS: { key: "a" | "b" | "abs_diff" | "sq_err" | "hadamard" | "ratio"; label: string }[] = [
  { key: "a", label: "A" },
  { key: "b", label: "B" },
  { key: "abs_diff", label: "Absolute Difference" },
  { key: "sq_err", label: "Squared Error" },
  { key: "hadamard", label: "Hadamard Product" },
  { key: "ratio", label: "Normalized Ratio" },
];

export function PairCompareTab({
  a, b, sources,
}: {
  a: NpzRef;
  b: NpzRef;
  sources: string[];
}) {
  const [metrics, setMetrics] = useState<PairMetrics | null>(null);
  const [metaA, setMetaA] = useState<NpzMeta | null>(null);
  const [metaB, setMetaB] = useState<NpzMeta | null>(null);
  const [layer, setLayer] = useState<string>("all");
  const [error, setError] = useState<string | null>(null);
  const handleDownload = () => openReport({ kind: "pair", a, b, sources });
  const [attnUrls, setAttnUrls] = useState<Record<string, string>>({});
  const [qkUrls, setQkUrls] = useState<Record<string, string>>({});
  const [qkvUrls, setQkvUrls] = useState<Record<string, string>>({});
  const [adv, setAdv] = useState<AdvancedMetricsResponse | null>(null);
  const [advError, setAdvError] = useState<string | null>(null);
  const [pcaSource, setPcaSource] = useState<string>(sources[0] ?? "hidden_in");
  const [convergenceA, setConvergenceA] = useState<ConvergenceResponse | null>(null);
  const [convergenceB, setConvergenceB] = useState<ConvergenceResponse | null>(null);
  const [convSource, setConvSource] = useState<string>("hidden_out");

  useEffect(() => {
    setError(null);
    setMetrics(null);
    if (sources.length === 0) return;
    compareMetrics([a, b], sources)
      .then((r) => setMetrics(r.pairs[0]?.metrics ?? null))
      .catch((e) => setError(String(e)));
  }, [a.run_id, a.kind, a.name, b.run_id, b.kind, b.name, sources.join(",")]);

  useEffect(() => {
    getNpzMeta(a).then(setMetaA).catch(console.error);
    getNpzMeta(b).then(setMetaB).catch(console.error);
  }, [a.run_id, a.kind, a.name, b.run_id, b.kind, b.name]);

  useEffect(() => {
    let cancelled = false;
    const urls: Record<string, string> = {};
    (async () => {
      if (metaA?.has_attention && metaB?.has_attention) {
        for (const v of VARIANTS) {
          try {
            urls[v.key] = await comparePairHeatmap(a, b, "attention", layer, v.key);
          } catch (e) { /* noop */ }
          if (cancelled) return;
        }
        setAttnUrls({ ...urls });
      } else {
        setAttnUrls({});
      }
    })();
    return () => { cancelled = true; };
  }, [metaA, metaB, layer, a.run_id, a.kind, a.name, b.run_id, b.kind, b.name]);

  useEffect(() => {
    let cancelled = false;
    const urls: Record<string, string> = {};
    (async () => {
      if (metaA?.has_qk && metaB?.has_qk) {
        for (const v of VARIANTS) {
          try {
            urls[v.key] = await comparePairHeatmap(a, b, "qk", layer, v.key);
          } catch (e) { /* noop */ }
          if (cancelled) return;
        }
        setQkUrls({ ...urls });
      } else {
        setQkUrls({});
      }
    })();
    return () => { cancelled = true; };
  }, [metaA, metaB, layer, a.run_id, a.kind, a.name, b.run_id, b.kind, b.name]);

  useEffect(() => {
    let cancelled = false;
    const urls: Record<string, string> = {};
    (async () => {
      if (metaA?.has_qkv_last && metaB?.has_qkv_last) {
        for (const v of VARIANTS) {
          try {
            // qkv is a full (L, D) matrix; layer selector is ignored server-side.
            urls[v.key] = await comparePairHeatmap(a, b, "qkv", "all", v.key);
          } catch (e) { /* noop */ }
          if (cancelled) return;
        }
        setQkvUrls({ ...urls });
      } else {
        setQkvUrls({});
      }
    })();
    return () => { cancelled = true; };
  }, [metaA, metaB, a.run_id, a.kind, a.name, b.run_id, b.kind, b.name]);

  useEffect(() => {
    setAdv(null);
    setAdvError(null);
    if (sources.length === 0) return;
    compareAdvanced([a, b], sources)
      .then(setAdv)
      .catch((e) => setAdvError(String(e)));
  }, [a.run_id, a.kind, a.name, b.run_id, b.kind, b.name, sources.join(",")]);

  useEffect(() => {
    if (!sources.includes(pcaSource)) setPcaSource(sources[0] ?? "hidden_in");
  }, [sources.join(",")]);

  useEffect(() => {
    setConvergenceA(null);
    setConvergenceB(null);
    getNpzConvergence(a, convSource).then(setConvergenceA).catch(console.error);
    getNpzConvergence(b, convSource).then(setConvergenceB).catch(console.error);
  }, [a.run_id, a.kind, a.name, b.run_id, b.kind, b.name, convSource]);

  if (error) return <div className="err">{error}</div>;
  if (sources.length === 0) return <div className="empty">select at least one source (hidden_in, hidden_out, attn_out, mlp_down_out) in the header</div>;
  if (!metrics) return <div className="empty">computing metrics…</div>;

  const commonLayers = metaA?.layers ?? [];

  const mkSeries = (metric: keyof PairMetrics[string]): LineSeries[] =>
    sources
      .map((src) => {
        const m = metrics[src];
        if (!m) return null;
        const data = m[metric] as number[];
        return { label: src, data, stroke: SOURCE_COLORS[src] ?? "#555" };
      })
      .filter((x): x is LineSeries => !!x);

  const aName = `${a.run_id} ${a.kind}:${a.name}`;
  const bName = `${b.run_id} ${b.kind}:${b.name}`;

  const xsForMetrics = (metrics[sources[0]]?.layers ?? []) as number[];

  return (
    <div>
      <div className="controls">
        <button className="btn-toggle" onClick={handleDownload}>↓ open report</button>
        <span className="muted"><b>A</b>: {aName}</span>
        <span className="muted"><b>B</b>: {bName}</span>
      </div>

      <div className="section">
        <h4>Per-layer metrics (one row per source)</h4>
        <table className="metrics-table">
          <thead>
            <tr>
              <th>source</th>
              <th>mean cos (avg)</th>
              <th>min cos (min)</th>
              <th>mean MAE (avg)</th>
              <th>mean overlap (avg)</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((src) => {
              const m = metrics[src];
              if (!m) return null;
              const avg = (a: number[]) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
              const mn = (a: number[]) => a.length ? Math.min(...a) : 0;
              return (
                <tr key={src}>
                  <td>{src}</td>
                  <td>{avg(m.mean_cos).toFixed(4)}</td>
                  <td>{mn(m.min_cos).toFixed(4)}</td>
                  <td>{avg(m.mean_mae).toExponential(3)}</td>
                  <td>{avg(m.mean_overlap).toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="section"><h4>Cosine similarity per layer</h4>
        <LineChart xs={xsForMetrics} series={mkSeries("mean_cos")} />
      </div>
      <div className="section"><h4>MAE per layer</h4>
        <LineChart xs={xsForMetrics} series={mkSeries("mean_mae")} />
      </div>
      <div className="section"><h4>STD of difference per layer</h4>
        <LineChart xs={xsForMetrics} series={mkSeries("std_diff")} />
      </div>
      <div className="section"><h4>Top-1% activation overlap (Jaccard) per layer</h4>
        <LineChart xs={xsForMetrics} series={mkSeries("mean_overlap")} />
      </div>

      <div className="section" style={{ borderTop: "2px solid #16a34a", marginTop: 24 }}>
        <h4 style={{ color: "#16a34a" }}>Residual Stream Convergence</h4>
        <div className="muted" style={{ marginBottom: 8 }}>
          Does the residual stream stop changing early?
          Adjacent-layer cosine similarity near 1.0 means the stream has settled.
          Effective depth = first layer where cosine ≥ 0.99 and never drops below again.
        </div>
        <div className="controls">
          <label>
            Source:
            <select value={convSource} onChange={(e) => setConvSource(e.target.value)} style={{ marginLeft: 4 }}>
              {CONVERGENCE_SOURCES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          {convergenceA && (
            <span className="muted">
              A eff. depth: last=<b>{convergenceA.effective_depth_last ?? "—"}</b>{" "}
              mean=<b>{convergenceA.effective_depth_mean ?? "—"}</b>
            </span>
          )}
          {convergenceB && (
            <span className="muted">
              B eff. depth: last=<b>{convergenceB.effective_depth_last ?? "—"}</b>{" "}
              mean=<b>{convergenceB.effective_depth_mean ?? "—"}</b>
            </span>
          )}
        </div>
        {(!convergenceA || !convergenceB) && <div className="muted">computing…</div>}
        {convergenceA && convergenceB && (() => {
          const nA = convergenceA.adj_layers.length;
          const nB = convergenceB.adj_layers.length;
          const n = Math.min(nA, nB);
          const xs = convergenceA.adj_layers.slice(0, n);
          return (
            <>
              <h5 style={{ marginTop: 8, marginBottom: 4 }}>Adjacent-layer cosine similarity (1.0 = no change)</h5>
              <LineChart
                xs={xs}
                series={[
                  { label: "A last token", data: convergenceA.adj_cosine_last.slice(0, n), stroke: "#2563eb" },
                  { label: "A mean", data: convergenceA.adj_cosine_mean.slice(0, n), stroke: "#93c5fd" },
                  { label: "B last token", data: convergenceB.adj_cosine_last.slice(0, n), stroke: "#dc2626" },
                  { label: "B mean", data: convergenceB.adj_cosine_mean.slice(0, n), stroke: "#fca5a5" },
                ]}
                height={220}
              />
              <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ (0 = no change)</h5>
              <LineChart
                xs={xs}
                series={[
                  { label: "A last token", data: convergenceA.delta_norm_last.slice(0, n), stroke: "#2563eb" },
                  { label: "A mean", data: convergenceA.delta_norm_mean.slice(0, n), stroke: "#93c5fd" },
                  { label: "B last token", data: convergenceB.delta_norm_last.slice(0, n), stroke: "#dc2626" },
                  { label: "B mean", data: convergenceB.delta_norm_mean.slice(0, n), stroke: "#fca5a5" },
                ]}
                height={220}
              />
            </>
          );
        })()}
      </div>

      <div className="section" style={{ borderTop: "2px solid #2563eb", marginTop: 24 }}>
        <h4 style={{ color: "#2563eb" }}>PCA Layer Trajectories</h4>
        <div className="controls">
          <label>
            Source:
            <select value={pcaSource} onChange={(e) => setPcaSource(e.target.value)} style={{ marginLeft: 4 }}>
              {sources.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
        </div>
        {advError && <div className="err">{advError}</div>}
        {!adv && !advError && <div className="muted">computing PCA…</div>}
        {adv && (
          adv.pca[pcaSource]
            ? <PcaScatter data={adv.pca[pcaSource]} refLabels={[aName, bName]} />
            : <div className="muted">no PCA data for {pcaSource}</div>
        )}
      </div>

      <div className="section">
        <h4>Attention &amp; q·k heatmaps</h4>
        <div className="controls">
          <label>
            Layer:
            <select value={layer} onChange={(e) => setLayer(e.target.value)} style={{ marginLeft: 4 }}>
              <option value="all">All layers stacked</option>
              {commonLayers.map((li) => <option key={li} value={String(li)}>layer {li}</option>)}
            </select>
          </label>
        </div>

        <h4 style={{ marginTop: 12 }}>Attention weights — softmax</h4>
        {Object.keys(attnUrls).length ? (
          <div className="grid5">
            {VARIANTS.map((v) => (
              <div key={v.key}>
                <div className="muted">{v.label}</div>
                {attnUrls[v.key]
                  ? <img className="heatmap-img" src={attnUrls[v.key]} alt={v.label} />
                  : <div className="muted">—</div>}
              </div>
            ))}
          </div>
        ) : <div className="muted">one of the npz lacks attention data</div>}

        <h4 style={{ marginTop: 12 }}>Raw q·k / √d_k scores</h4>
        {Object.keys(qkUrls).length ? (
          <div className="grid5">
            {VARIANTS.map((v) => (
              <div key={v.key}>
                <div className="muted">{v.label}</div>
                {qkUrls[v.key]
                  ? <img className="heatmap-img" src={qkUrls[v.key]} alt={v.label} />
                  : <div className="muted">—</div>}
              </div>
            ))}
          </div>
        ) : <div className="muted">one of the npz lacks q/k data</div>}

        <h4 style={{ marginTop: 12 }}>Stacked ⟨q,k⟩·v at last token (rows = layers)</h4>
        {Object.keys(qkvUrls).length ? (
          <div className="grid5">
            {VARIANTS.map((v) => (
              <div key={v.key}>
                <div className="muted">{v.label}</div>
                {qkvUrls[v.key]
                  ? <img className="heatmap-img" src={qkvUrls[v.key]} alt={v.label} />
                  : <div className="muted">—</div>}
              </div>
            ))}
          </div>
        ) : <div className="muted">one of the npz lacks qkv_last data</div>}
      </div>
    </div>
  );
}
