import { useEffect, useState } from "react";
import {
  NpzRef, NpzMeta, EntropyResponse, ConvergenceResponse,
  attentionPngUrl, qkPngUrl, qkvPngUrl, getEntropy, getNpzMeta, getNpzConvergence,
} from "./api";
import { LineChart } from "./LineChart";

const CONVERGENCE_SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out"];

export function NpzAttentionTab({ npz }: { npz: NpzRef }) {
  const [meta, setMeta] = useState<NpzMeta | null>(null);
  const [layer, setLayer] = useState<string>("all");
  const [entropy, setEntropy] = useState<EntropyResponse | null>(null);
  const [convergence, setConvergence] = useState<ConvergenceResponse | null>(null);
  const [convSource, setConvSource] = useState<string>("hidden_out");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    setMeta(null);
    setEntropy(null);
    setConvergence(null);
    getNpzMeta(npz).then(setMeta).catch((e) => setError(String(e)));
  }, [npz.run_id, npz.kind, npz.name]);

  useEffect(() => {
    if (!meta?.has_attention) return;
    getEntropy(npz).then(setEntropy).catch((e) => setError(String(e)));
  }, [meta, npz.run_id, npz.kind, npz.name]);

  useEffect(() => {
    if (!meta || meta.layers.length < 2) return;
    setConvergence(null);
    getNpzConvergence(npz, convSource).then(setConvergence).catch(console.error);
  }, [meta, npz.run_id, npz.kind, npz.name, convSource]);

  if (error) return <div className="err">{error}</div>;
  if (!meta) return <div className="empty">loading…</div>;

  const attnPng = meta.has_attention ? attentionPngUrl(npz, layer) : null;
  const qkPng = meta.has_qk ? qkPngUrl(npz, layer) : null;
  const qkvPng = meta.has_qkv_last ? qkvPngUrl(npz) : null;

  return (
    <div>
      <div className="controls">
        <label>
          Layer:
          <select value={layer} onChange={(e) => setLayer(e.target.value)} style={{ marginLeft: 4 }}>
            <option value="all">All layers stacked</option>
            {meta.layers.map((li) => (
              <option key={li} value={String(li)}>layer {li}</option>
            ))}
          </select>
        </label>
        <span className="muted">{npz.kind} — {npz.name}</span>
      </div>

      <div className="section">
        <h4>Attention weights — softmax</h4>
        {attnPng ? <img className="heatmap-img" src={attnPng} alt="attention" /> : <div className="muted">no attention captured in this npz</div>}
      </div>

      <div className="section">
        <h4>Raw q·k / √d_k scores</h4>
        {qkPng ? <img className="heatmap-img" src={qkPng} alt="qk" /> : <div className="muted">no q/k tensors in this npz</div>}
      </div>

      <div className="section">
        <h4>Stacked ⟨q,k⟩·v at last token — rows = layers</h4>
        {qkvPng
          ? <img className="heatmap-img" src={qkvPng} alt="qkv_last" />
          : <div className="muted">no qkv_last tensor in this npz</div>}
      </div>

      <div className="section">
        <h4>Attention entropy per layer</h4>
        {entropy
          ? <LineChart xs={entropy.layers} series={[{ label: "entropy (nats)", data: entropy.entropy, stroke: "#2563eb" }]} />
          : <div className="muted">no attention entropy available</div>}
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
          {convergence && (
            <>
              <span className="muted">
                eff. depth (last token): <b>{convergence.effective_depth_last ?? "—"}</b>
              </span>
              <span className="muted">
                eff. depth (mean): <b>{convergence.effective_depth_mean ?? "—"}</b>
              </span>
            </>
          )}
        </div>
        {meta.layers.length < 2 && <div className="muted">need ≥2 captured layers</div>}
        {meta.layers.length >= 2 && !convergence && <div className="muted">computing…</div>}
        {convergence && convergence.adj_layers.length > 0 && (
          <>
            <h5 style={{ marginTop: 8, marginBottom: 4 }}>Adjacent-layer cosine similarity (1.0 = no change)</h5>
            <LineChart
              xs={convergence.adj_layers}
              series={[
                { label: "last token", data: convergence.adj_cosine_last, stroke: "#2563eb" },
                { label: "mean over sequence", data: convergence.adj_cosine_mean, stroke: "#93c5fd" },
              ]}
              height={200}
            />
            <h5 style={{ marginTop: 16, marginBottom: 4 }}>Relative update norm ‖Δh‖/‖h‖ (0 = no change)</h5>
            <LineChart
              xs={convergence.adj_layers}
              series={[
                { label: "last token", data: convergence.delta_norm_last, stroke: "#dc2626" },
                { label: "mean over sequence", data: convergence.delta_norm_mean, stroke: "#fca5a5" },
              ]}
              height={200}
            />
          </>
        )}
      </div>
    </div>
  );
}
