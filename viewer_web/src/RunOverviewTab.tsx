import { useEffect, useState } from "react";
import { getRun, getRunLogitStats, LogitStatsResponse } from "./api";
import { LineChart } from "./LineChart";

export function RunOverviewTab({ runId }: { runId: string }) {
  const [run, setRun] = useState<any>(null);
  const [logitStats, setLogitStats] = useState<LogitStatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRun(null);
    setLogitStats(null);
    setError(null);
    getRun(runId).then(setRun).catch((e) => setError(String(e)));
    getRunLogitStats(runId).then(setLogitStats).catch(() => {/* older runs lack logit_stats */});
  }, [runId]);

  if (error) return <div className="err">{error}</div>;
  if (!run) return <div className="empty">loading…</div>;

  return (
    <div>
      <div className="section">
        <h4>Scenario</h4>
        <div><b>name:</b> {run.scenario?.name}</div>
        <div><b>model:</b> {run.model?.id}</div>
        <div><b>device:</b> {run.env?.device}</div>
        <div><b>created:</b> {run.created_at}</div>
      </div>
      <div className="section">
        <h4>Prompt messages</h4>
        <ol>
          {(run.scenario?.prompt?.messages ?? []).map((m: any, i: number) => (
            <li key={i}>
              <b>{m.role}:</b> {String(m.content).slice(0, 200)}
            </li>
          ))}
        </ol>
      </div>

      {logitStats && logitStats.steps.length > 0 && (
        <div className="section" style={{ borderTop: "2px solid #a855f7", marginTop: 24 }}>
          <h4 style={{ color: "#a855f7" }}>Next-token distribution peakedness</h4>
          <div className="muted" style={{ marginBottom: 8 }}>
            Computed on the full vocabulary softmax at each generation step.
            Low entropy = confident (peaked); high entropy = uncertain (flat).
          </div>

          <h5 style={{ marginTop: 8, marginBottom: 4 }}>Entropy (nats)</h5>
          <LineChart
            xs={logitStats.steps}
            series={[{ label: "entropy (nats)", data: logitStats.entropy, stroke: "#2563eb" }]}
            height={180}
            xLabel="step"
          />

          <h5 style={{ marginTop: 16, marginBottom: 4 }}>Effective vocabulary size — exp(entropy)</h5>
          <div className="muted" style={{ marginBottom: 4 }}>
            Number of tokens the model is effectively spreading weight across. Uniform over V tokens → exp(entropy) = V.
          </div>
          <LineChart
            xs={logitStats.steps}
            series={[{ label: "effective vocab", data: logitStats.effective_vocab, stroke: "#a855f7" }]}
            height={180}
            xLabel="step"
          />

          <h5 style={{ marginTop: 16, marginBottom: 4 }}>Top-1 probability</h5>
          <div className="muted" style={{ marginBottom: 4 }}>
            Probability mass on the single most likely next token.
          </div>
          <LineChart
            xs={logitStats.steps}
            series={[{ label: "top-1 prob", data: logitStats.top1_prob, stroke: "#16a34a" }]}
            height={180}
            xLabel="step"
          />
        </div>
      )}

      <div className="section"><h4>Full run.json</h4>
        <pre style={{ fontSize: 11, background: "#fafafa", padding: 8, overflow: "auto", maxHeight: 400 }}>
          {JSON.stringify(run, null, 2)}
        </pre>
      </div>
    </div>
  );
}
