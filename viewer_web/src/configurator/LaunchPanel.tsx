import { useEffect, useRef, useState } from "react";
import { getJob, launchScenario, saveScenario, validateScenario } from "../api";
import { ScenarioModel } from "./scenarioModel";

interface Props {
  scenario: ScenarioModel;
  yamlText: string;
  yamlError: string | null;
  onOpenRunTab: (runId: string) => void;
}

type LaunchStatus = "idle" | "saving" | "launching" | "running" | "done" | "failed";

export function LaunchPanel({ scenario, yamlText, yamlError, onOpenRunTab }: Props) {
  const [savePath, setSavePath] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [status, setStatus] = useState<LaunchStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const logRef = useRef<HTMLPreElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // keep savePath in sync with scenario.name
  useEffect(() => {
    if (scenario.name) setSavePath(`scenarios/${scenario.name}.yaml`);
  }, [scenario.name]);

  // auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const handleSave = async (): Promise<string | null> => {
    setStatus("saving");
    setError(null);
    setValidationErrors([]);
    const relPath = savePath.startsWith("scenarios/") ? savePath.slice("scenarios/".length) : savePath;
    try {
      const val = await validateScenario(yamlText);
      if (!val.ok) {
        setValidationErrors(val.errors.map((e: { loc: string[]; msg: string }) => `${e.loc.join(".")}: ${e.msg}`));
        setStatus("idle");
        return null;
      }
      await saveScenario(relPath, yamlText, overwrite);
      return relPath;
    } catch (e) {
      setError(String(e));
      setStatus("idle");
      return null;
    }
  };

  const handleLaunch = async () => {
    setRunId(null);
    setLogLines([]);
    const relPath = await handleSave();
    if (!relPath) return;

    setStatus("launching");
    try {
      const { job_id } = await launchScenario(relPath);
      setStatus("running");
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(job_id);
          setLogLines(job.log_tail);
          if (job.run_id) setRunId(job.run_id);
          if (job.status === "done") { setStatus("done"); stopPolling(); }
          if (job.status === "failed") { setStatus("failed"); setError(`exit code ${job.returncode}`); stopPolling(); }
        } catch (_) { /* keep polling */ }
      }, 1000);
    } catch (e) {
      setError(String(e));
      setStatus("idle");
    }
  };

  const busy = status === "saving" || status === "launching" || status === "running";
  const canAct = !yamlError && !busy;

  return (
    <div className="section">
      <h4>Save &amp; Launch</h4>

      <div className="controls" style={{ flexWrap: "wrap", alignItems: "center" }}>
        <label style={{ flex: 1 }}>
          Save path
          <input
            type="text"
            value={savePath}
            onChange={(e) => setSavePath(e.target.value)}
            style={{ width: "100%", marginLeft: 6 }}
            placeholder="scenarios/my_scenario.yaml"
          />
        </label>
        <label>
          <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
          {" "}overwrite
        </label>
      </div>

      <div className="controls" style={{ marginTop: 6 }}>
        <button className="btn-toggle" disabled={!canAct} onClick={handleSave as () => void}>
          Save
        </button>
        <button className="btn-toggle" disabled={!canAct} onClick={handleLaunch}>
          {status === "running" ? "Running…" : status === "saving" || status === "launching" ? "Starting…" : "Launch"}
        </button>
        <span className={`pill ${status === "done" ? "attn" : ""}`} style={{ marginLeft: 4 }}>
          {status}
        </span>
      </div>

      {validationErrors.length > 0 && (
        <div className="err" style={{ marginTop: 6 }}>
          {validationErrors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {error && <div className="err" style={{ marginTop: 4 }}>{error}</div>}

      {logLines.length > 0 && (
        <pre ref={logRef} className="log-tail">
          {logLines.join("\n")}
        </pre>
      )}

      {runId && status === "done" && (
        <button className="btn-toggle" style={{ marginTop: 6 }} onClick={() => onOpenRunTab(runId)}>
          Open run tab → {runId}
        </button>
      )}
    </div>
  );
}
