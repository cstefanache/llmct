import { useEffect, useState } from "react";
import { NpzRef, NpzMeta, EntropyResponse, attentionPngUrl, qkPngUrl, qkvPngUrl, getEntropy, getNpzMeta } from "./api";
import { LineChart } from "./LineChart";

export function NpzAttentionTab({ npz }: { npz: NpzRef }) {
  const [meta, setMeta] = useState<NpzMeta | null>(null);
  const [layer, setLayer] = useState<string>("all");
  const [entropy, setEntropy] = useState<EntropyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    setMeta(null);
    setEntropy(null);
    getNpzMeta(npz).then(setMeta).catch((e) => setError(String(e)));
  }, [npz.run_id, npz.kind, npz.name]);

  useEffect(() => {
    if (!meta?.has_attention) return;
    getEntropy(npz).then(setEntropy).catch((e) => setError(String(e)));
  }, [meta, npz.run_id, npz.kind, npz.name]);

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
    </div>
  );
}
