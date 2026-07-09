import { LineChart, LineSeries } from "./LineChart";
import { PcaSourceResult } from "./api";

// ─── filtering helpers ────────────────────────────────────────────────────────

/** Drop hidden layers from a line chart's xs + every series (kept points only). */
export function applyLayerFilter(
  xs: number[], series: LineSeries[], hidden: Set<number>,
): { xs: number[]; series: LineSeries[] } {
  if (hidden.size === 0) return { xs, series };
  const keep: number[] = [];
  xs.forEach((x, i) => { if (!hidden.has(x)) keep.push(i); });
  return {
    xs: keep.map((i) => xs[i]),
    series: series.map((s) => ({ ...s, data: keep.map((i) => s.data[i]) })),
  };
}

/** Drop coords whose layer is hidden from a PCA result. */
export function filterPcaLayers(data: PcaSourceResult, hidden: Set<number>): PcaSourceResult {
  if (hidden.size === 0) return data;
  const keep = data.layer_indices
    .map((li, i) => (hidden.has(data.layers[li]) ? -1 : i))
    .filter((i) => i >= 0);
  return {
    ...data,
    coords: keep.map((i) => data.coords[i]),
    ref_indices: keep.map((i) => data.ref_indices[i]),
    layer_indices: keep.map((i) => data.layer_indices[i]),
  };
}

// ─── line chart that respects the global layer filter ─────────────────────────

export function FilteredLineChart({
  xs, series, hidden, title, height, xLabel,
}: {
  xs: number[];
  series: LineSeries[];
  hidden: Set<number>;
  title?: string;
  height?: number;
  xLabel?: string;
}) {
  const f = applyLayerFilter(xs, series, hidden);
  return <LineChart xs={f.xs} series={f.series} title={title} height={height} xLabel={xLabel} />;
}

// ─── the chip row ─────────────────────────────────────────────────────────────

/** Dual-thumb slider that picks a contiguous [from, to] window of layers. */
export function LayerRangeSlider({
  min, max, from, to, onChange,
}: {
  min: number;
  max: number;
  from: number;
  to: number;
  onChange: (from: number, to: number) => void;
}) {
  if (max <= min) return null;
  const pct = (v: number) => ((v - min) / (max - min)) * 100;
  return (
    <div className="layer-range">
      <span className="layer-range-val">{from}</span>
      <div className="layer-range-track">
        <div
          className="layer-range-fill"
          style={{ left: `${pct(from)}%`, right: `${100 - pct(to)}%` }}
        />
        <input
          type="range" min={min} max={max} value={from}
          onChange={(e) => onChange(Math.min(+e.target.value, to), to)}
        />
        <input
          type="range" min={min} max={max} value={to}
          onChange={(e) => onChange(from, Math.max(+e.target.value, from))}
        />
      </div>
      <span className="layer-range-val">{to}</span>
    </div>
  );
}

export function LayerChips({
  layers, hidden, onToggle, onShowAll, onHideAll,
}: {
  layers: number[];
  hidden: Set<number>;
  /** shift = true → toggle the whole band from the last click to this layer. */
  onToggle: (layer: number, shift: boolean) => void;
  onShowAll: () => void;
  onHideAll: () => void;
}) {
  if (layers.length === 0) return null;
  const shown = layers.length - layers.filter((l) => hidden.has(l)).length;
  return (
    <div className="layer-filter">
      <div className="layer-filter-head">
        <span className="layer-filter-label">
          Layers <span className="muted">({shown}/{layers.length} shown)</span>
        </span>
        <button className="btn-toggle" onClick={onShowAll} disabled={hidden.size === 0}>show all</button>
        <button className="btn-toggle" onClick={onHideAll} disabled={shown === 0}>hide all</button>
      </div>
      <div className="layer-chip-row">
        {layers.map((l) => (
          <button
            key={l}
            className={`layer-chip ${hidden.has(l) ? "off" : "on"}`}
            title={hidden.has(l) ? `layer ${l} — hidden (click to show)` : `layer ${l} — shown (click to hide, shift-click for range)`}
            onClick={(e) => onToggle(l, e.shiftKey)}
          >
            {l}
          </button>
        ))}
      </div>
    </div>
  );
}
