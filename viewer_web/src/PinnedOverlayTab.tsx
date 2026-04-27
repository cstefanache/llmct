import { useEffect, useMemo, useState } from "react";
import {
  NpzRef, EntropyResponse, ConvergenceResponse,
  getEntropy, getNpzConvergence,
} from "./api";
import { LineChart, LineSeries } from "./LineChart";
import { makeRefLabels } from "./refLabels";

const PAIR_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316",
  "#0891b2", "#ca8a04", "#ec4899", "#475569", "#84cc16",
];

const CONVERGENCE_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"];

export function PinnedOverlayTab({ refs }: { refs: NpzRef[] }) {
  const [convSource, setConvSource] = useState("hidden_out");
  const [entropyData, setEntropyData] = useState<(EntropyResponse | null)[]>([]);
  const [convData, setConvData] = useState<(ConvergenceResponse | null)[]>([]);

  const refsKey = useMemo(
    () => refs.map((r) => `${r.run_id}|${r.kind}|${r.name}`).join("::"),
    [refs],
  );

  useEffect(() => {
    setEntropyData(refs.map(() => null));
    refs.forEach((ref, i) => {
      getEntropy(ref)
        .then((e) =>
          setEntropyData((prev) => {
            const next = [...prev];
            next[i] = e;
            return next;
          }),
        )
        .catch(console.error);
    });
  }, [refsKey]);

  useEffect(() => {
    setConvData(refs.map(() => null));
    refs.forEach((ref, i) => {
      getNpzConvergence(ref, convSource)
        .then((c) =>
          setConvData((prev) => {
            const next = [...prev];
            next[i] = c;
            return next;
          }),
        )
        .catch(console.error);
    });
  }, [refsKey, convSource]);

  if (refs.length < 2) return <div className="empty">pin 2+ npz tabs to see overlay</div>;

  const refLabels = makeRefLabels(refs);

  const entropySeries: LineSeries[] = entropyData
    .map((e, i) =>
      e ? { label: refLabels[i], data: e.entropy, stroke: PAIR_COLORS[i % PAIR_COLORS.length] } : null,
    )
    .filter((s): s is LineSeries => s !== null);
  const entropyXs = entropyData.find(Boolean)?.layers ?? [];

  const convXs = convData.find(Boolean)?.adj_layers ?? [];

  const mkSeries = (
    field: "adj_cosine_last" | "adj_cosine_mean" | "delta_norm_last" | "delta_norm_mean",
  ): LineSeries[] =>
    convData
      .map((c, i) =>
        c ? { label: refLabels[i], data: c[field], stroke: PAIR_COLORS[i % PAIR_COLORS.length] } : null,
      )
      .filter((s): s is LineSeries => s !== null);

  const loadedConv = convData.filter(Boolean).length;

  return (
    <div>
      <div className="controls">
        <span className="muted">
          <b>Overlay ({refs.length} refs):</b> {refLabels.join(", ")}
        </span>
      </div>

      <div className="section">
        <h4>Attention entropy per layer</h4>
        {entropySeries.length > 0
          ? <LineChart xs={entropyXs} series={entropySeries} />
          : <div className="muted">loading entropy…</div>}
      </div>

      <div className="section" style={{ borderTop: "2px solid #16a34a", marginTop: 24 }}>
        <h4 style={{ color: "#16a34a" }}>Residual Stream Convergence — overlay</h4>
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
        {loadedConv === 0 && <div className="muted">computing…</div>}
        {loadedConv > 0 && (
          <>
            <h5 style={{ marginTop: 8, marginBottom: 4 }}>Adjacent-layer cosine similarity — last token</h5>
            <LineChart xs={convXs} series={mkSeries("adj_cosine_last")} height={200} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Adjacent-layer cosine similarity — mean over sequence</h5>
            <LineChart xs={convXs} series={mkSeries("adj_cosine_mean")} height={200} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ — last token</h5>
            <LineChart xs={convXs} series={mkSeries("delta_norm_last")} height={200} />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ — mean over sequence</h5>
            <LineChart xs={convXs} series={mkSeries("delta_norm_mean")} height={200} />
          </>
        )}
      </div>
    </div>
  );
}
