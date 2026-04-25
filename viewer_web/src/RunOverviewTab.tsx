import { useEffect, useState } from "react";
import { getRun } from "./api";

export function RunOverviewTab({ runId }: { runId: string }) {
  const [run, setRun] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRun(runId).then(setRun).catch((e) => setError(String(e)));
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
      <div className="section"><h4>Full run.json</h4>
        <pre style={{ fontSize: 11, background: "#fafafa", padding: 8, overflow: "auto", maxHeight: 400 }}>
          {JSON.stringify(run, null, 2)}
        </pre>
      </div>
    </div>
  );
}
