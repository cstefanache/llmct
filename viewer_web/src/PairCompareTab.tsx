import { useEffect, useState } from "react";
import {
  NpzRef, NpzMeta, PairMetrics, compareMetrics, comparePairHeatmap, getNpzMeta,
} from "./api";
import { LineChart, LineSeries } from "./LineChart";

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
  const [attnUrls, setAttnUrls] = useState<Record<string, string>>({});
  const [qkUrls, setQkUrls] = useState<Record<string, string>>({});
  const [qkvUrls, setQkvUrls] = useState<Record<string, string>>({});

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

  const aName = `${a.kind}:${a.name}`;
  const bName = `${b.kind}:${b.name}`;

  const xsForMetrics = (metrics[sources[0]]?.layers ?? []) as number[];

  return (
    <div>
      <div className="controls">
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
