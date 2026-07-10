import { useEffect, useMemo, useRef, useState } from "react";
import { NpzRef, LensResponse, LensLayer, LensToken, getLogitLens } from "./api";
import { makeRefLabels } from "./refLabels";
import { useLabelOverrides } from "./labelStore";
import { RefLabelEditor } from "./RefLabelEditor";
import { LayerChips, LayerRangeSlider } from "./LayerFilter";

const PAIR_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#a855f7", "#f97316",
  "#0891b2", "#ca8a04", "#ec4899", "#475569", "#84cc16",
];

// ─── display helpers ──────────────────────────────────────────────────────────

/** Make token whitespace visible in a nowrap table cell. */
export function showToken(tok: string | null, id: number): string {
  if (tok === null) return `#${id}`;
  if (tok === "") return "∅";
  return tok.replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/ /g, "·");
}

export const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

/** Horizontal probability bar-fill background: fills `prob` fraction of the cell. */
export function barBg(prob: number, color: string): string {
  const x = Math.max(0, Math.min(100, prob * 100));
  return `linear-gradient(90deg, ${color} ${x}%, transparent ${x}%)`;
}

function Explain({ what, how }: { what: string; how: string }) {
  return (
    <div className="muted" style={{ fontSize: 11, marginBottom: 8, lineHeight: 1.5 }}>
      <div><b>What:</b> {what}</div>
      <div><b>How to read:</b> {how}</div>
    </div>
  );
}

// ─── single-snapshot table (spec §2) ──────────────────────────────────────────

/** Aggregate top-k tokens across the selected layers by their max probability. */
function aggregateTop(byLayer: Map<number, LensLayer>, layers: number[], topK: number) {
  const best = new Map<number, { token: string | null; prob: number }>();
  for (const l of layers) {
    const frame = byLayer.get(l);
    if (!frame) continue;
    for (const t of frame.predicted) {
      const cur = best.get(t.id);
      if (!cur || t.prob > cur.prob) best.set(t.id, { token: t.token, prob: t.prob });
    }
  }
  return [...best.entries()]
    .map(([id, v]) => ({ id, ...v }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, topK);
}

function LensCell({ tok, color }: { tok: LensToken | { id: number; token: string | null; prob: number }; color: string }) {
  return (
    <td
      className="lens-cell"
      style={{ background: barBg(tok.prob, color) }}
      title={`${tok.token === null ? "" : JSON.stringify(tok.token) + " "}id=${tok.id} · ${pct(tok.prob)}`}
    >
      <span className="lens-tok">{showToken(tok.token, tok.id)}</span>
      <span className="lens-prob">{pct(tok.prob)}</span>
    </td>
  );
}

function LensTable({
  data, layers, topK, color,
}: {
  data: LensResponse;
  layers: number[];       // selected + sorted layer numbers
  topK: number;
  color: string;
}) {
  const byLayer = useMemo(() => {
    const m = new Map<number, LensLayer>();
    data.layers.forEach((l) => m.set(l.layer, l));
    return m;
  }, [data]);
  const present = layers.filter((l) => byLayer.has(l));
  const agg = aggregateTop(byLayer, present, topK);
  const rows = Array.from({ length: topK }, (_, r) => r);

  return (
    <div className="lens-scroll">
      <table className="lens-table">
        <thead>
          <tr>
            <th className="lens-rank-h">#</th>
            {present.map((l) => (
              <th key={l}>L{l}</th>
            ))}
            <th className="lens-agg-h" title="Top tokens across the selected layers, ranked by their peak probability">Σ top</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <td className="lens-rank">{r + 1}</td>
              {present.map((l) => {
                const frame = byLayer.get(l)!;
                const tok = frame.predicted[r];
                return tok
                  ? <LensCell key={l} tok={tok} color={color} />
                  : <td key={l} className="lens-cell lens-empty">—</td>;
              })}
              {agg[r]
                ? <LensCell tok={agg[r]} color="rgba(100,116,139,0.30)" />
                : <td className="lens-cell lens-empty">—</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── global divergence summary (spec §3) ──────────────────────────────────────

function GlobalSummary({
  datas, labels, layers,
}: {
  datas: (LensResponse | null)[];
  labels: string[];
  layers: number[];
}) {
  // argmax id + token/prob per (snapshot, layer)
  const byLayerPerRef = datas.map((d) => {
    const m = new Map<number, LensLayer>();
    d?.layers.forEach((l) => m.set(l.layer, l));
    return m;
  });
  // A layer diverges if the snapshots that have data don't all share one argmax id.
  const diverges = (l: number): boolean => {
    const ids = byLayerPerRef.map((m) => m.get(l)?.argmax_id).filter((v) => v !== undefined);
    return new Set(ids).size > 1;
  };
  const baseId = (l: number) => byLayerPerRef.find((m) => m.get(l))?.get(l)?.argmax_id;

  return (
    <div className="lens-scroll">
      <table className="lens-table lens-summary">
        <thead>
          <tr>
            <th className="lens-rank-h">snapshot</th>
            {layers.map((l) => (
              <th key={l} className={diverges(l) ? "lens-diverge-h" : ""}>
                L{l}{diverges(l) ? " ⚠" : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {datas.map((d, i) => (
            <tr key={i}>
              <td className="lens-rank" style={{ borderLeft: `3px solid ${PAIR_COLORS[i % PAIR_COLORS.length]}` }}>
                {labels[i]}
              </td>
              {layers.map((l) => {
                const frame = byLayerPerRef[i].get(l);
                if (!frame) return <td key={l} className="lens-cell lens-empty">—</td>;
                const differs = diverges(l) && frame.argmax_id !== baseId(l);
                return (
                  <td
                    key={l}
                    className={`lens-cell${differs ? " lens-diverge" : ""}`}
                    title={`layer ${l} · id=${frame.argmax_id} · ${pct(frame.argmax_prob)}`}
                  >
                    <span className="lens-tok">{showToken(frame.argmax_token, frame.argmax_id)}</span>
                    <span className="lens-prob">{pct(frame.argmax_prob)}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── main tab ─────────────────────────────────────────────────────────────────

export function LensCompareTab({ refs }: { refs: NpzRef[] }) {
  const [topK, setTopK] = useState(10);
  const [position, setPosition] = useState(-1);
  const [mode, setMode] = useState<"single" | "comparison">(refs.length >= 2 ? "comparison" : "single");
  const [singleIdx, setSingleIdx] = useState(0);
  const [lensData, setLensData] = useState<(LensResponse | null)[]>([]);
  const [errors, setErrors] = useState<(string | null)[]>([]);
  const [loading, setLoading] = useState(false);
  const [hiddenLayers, setHiddenLayers] = useState<Set<number>>(new Set());
  const lastClicked = useRef<number | null>(null);
  useLabelOverrides();

  const refsKey = useMemo(
    () => refs.map((r) => `${r.run_id}|${r.kind}|${r.name}`).join("::"),
    [refs],
  );

  useEffect(() => {
    if (refs.length === 0) return;
    setLensData(refs.map(() => null));
    setErrors(refs.map(() => null));
    setLoading(true);
    let cancelled = false;
    Promise.all(
      refs.map((ref, i) =>
        getLogitLens(ref, position, topK)
          .then((d) => { if (!cancelled) setLensData((prev) => { const n = [...prev]; n[i] = d; return n; }); })
          .catch((e) => { if (!cancelled) setErrors((prev) => { const n = [...prev]; n[i] = String(e); return n; }); }),
      ),
    ).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refsKey, topK, position]);

  useEffect(() => {
    setMode(refs.length >= 2 ? "comparison" : "single");
    if (singleIdx >= refs.length) setSingleIdx(0);
  }, [refsKey]);

  if (refs.length === 0)
    return <div className="empty">select one or more snapshots from the sidebar, then open Token Lens</div>;

  const refLabels = makeRefLabels(refs);

  // union of layers across loaded snapshots
  const layerSet = new Set<number>();
  lensData.forEach((d) => d?.layers.forEach((l) => layerSet.add(l.layer)));
  const allLayers = [...layerSet].sort((a, b) => a - b);

  const toggleLayer = (layer: number, shift: boolean) => {
    setHiddenLayers((prev) => {
      const next = new Set(prev);
      const willHide = !next.has(layer);
      const from = shift && lastClicked.current !== null ? lastClicked.current : layer;
      const lo = Math.min(from, layer), hi = Math.max(from, layer);
      for (const l of allLayers) if (l >= lo && l <= hi) { if (willHide) next.add(l); else next.delete(l); }
      return next;
    });
    lastClicked.current = layer;
  };
  const showAllLayers = () => setHiddenLayers(new Set());
  const hideAllLayers = () => setHiddenLayers(new Set(allLayers));
  const shownLayers = allLayers.filter((l) => !hiddenLayers.has(l));
  const rangeFrom = shownLayers.length ? shownLayers[0] : (allLayers[0] ?? 0);
  const rangeTo = shownLayers.length ? shownLayers[shownLayers.length - 1] : (allLayers[allLayers.length - 1] ?? 0);
  const setRange = (from: number, to: number) =>
    setHiddenLayers(new Set(allLayers.filter((l) => l < from || l > to)));

  const modelId = lensData.find(Boolean)?.model_id;
  const anyLoaded = lensData.some(Boolean);

  return (
    <div>
      <div className="controls">
        <label>
          Token depth (x){" "}
          <input
            type="number" min={1} max={30} value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(30, +e.target.value || 1)))}
            style={{ width: 52 }}
          />
        </label>
        <label>
          Position{" "}
          <input
            type="number" value={position}
            onChange={(e) => setPosition(Number.isFinite(+e.target.value) ? +e.target.value : -1)}
            style={{ width: 52 }}
            title="Sequence position to read the next-token prediction at (-1 = last token)"
          />
        </label>
        {refs.length >= 2 && (
          <span className="lens-mode">
            <button className={`btn-toggle ${mode === "single" ? "lens-mode-on" : ""}`} onClick={() => setMode("single")}>single</button>
            <button className={`btn-toggle ${mode === "comparison" ? "lens-mode-on" : ""}`} onClick={() => setMode("comparison")}>comparison</button>
          </span>
        )}
        {mode === "single" && refs.length > 1 && (
          <label>
            snapshot{" "}
            <select value={singleIdx} onChange={(e) => setSingleIdx(+e.target.value)}>
              {refs.map((_, i) => <option key={i} value={i}>{refLabels[i]}</option>)}
            </select>
          </label>
        )}
        {loading && <span className="muted">computing… (first call loads {modelId ?? "the model"}, may take a while)</span>}
        {modelId && <span className="muted">model: {modelId}</span>}
      </div>

      <Explain
        what="Logit lens — each intermediate layer's residual stream projected through the model's final norm + unembedding to next-token probabilities."
        how="Columns are layers, rows the top-x predicted tokens (bar = probability). Watch the prediction sharpen with depth; the far-right Σ column is the top tokens aggregated across shown layers."
      />

      <RefLabelEditor refs={refs} colors={PAIR_COLORS} />

      {allLayers.length > 0 && (
        <>
          <LayerChips
            layers={allLayers} hidden={hiddenLayers}
            onToggle={toggleLayer} onShowAll={showAllLayers} onHideAll={hideAllLayers}
          />
          <LayerRangeSlider min={allLayers[0]} max={allLayers[allLayers.length - 1]} from={rangeFrom} to={rangeTo} onChange={setRange} />
        </>
      )}

      {errors.map((e, i) => e && <div key={i} className="err">{refLabels[i]}: {e}</div>)}

      {!anyLoaded && !loading && errors.every((e) => !e) && (
        <div className="empty">no per-layer data</div>
      )}

      {/* comparison: global divergence summary first, then stacked per-snapshot tables */}
      {mode === "comparison" && anyLoaded && (
        <div className="section">
          <h4>Global summary — top token per layer (⚠ = snapshots diverge)</h4>
          <GlobalSummary datas={lensData} labels={refLabels} layers={shownLayers} />
        </div>
      )}

      {mode === "comparison"
        ? lensData.map((d, i) =>
            d ? (
              <div className="section" key={i}>
                <h4><span className="lens-swatch" style={{ background: PAIR_COLORS[i % PAIR_COLORS.length] }} /> {refLabels[i]}</h4>
                <LensTable data={d} layers={shownLayers} topK={topK} color="rgba(37,99,235,0.22)" />
              </div>
            ) : null,
          )
        : (() => {
            const d = lensData[singleIdx];
            if (!d) return null;
            return (
              <div className="section">
                <h4>{refLabels[singleIdx]}</h4>
                <LensTable data={d} layers={shownLayers} topK={topK} color="rgba(37,99,235,0.22)" />
              </div>
            );
          })()}
    </div>
  );
}
