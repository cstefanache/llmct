import { useEffect, useState } from "react";
import { getLlmValidate, LlmValidateResult, LlmValidationEntry } from "./api";

function ValidationEntry({ entry, index }: { entry: LlmValidationEntry; index: number }) {
  const [showTemplates, setShowTemplates] = useState(false);

  return (
    <div className="section" style={{ borderLeft: entry.error ? "3px solid #b00" : "3px solid #16a34a" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>#{index} {entry.model}</span>
        {entry.provider && <span className="pill">{entry.provider}</span>}
        {entry.error
          ? <span className="pill" style={{ background: "#fee2e2", color: "#b00" }}>error</span>
          : <span className="pill" style={{ background: "#dcfce7", color: "#166534" }}>ok</span>}
      </div>

      {entry.error && (
        <pre style={{ fontSize: 11, color: "#b00", background: "#fff5f5", padding: 8, borderRadius: 4, overflow: "auto", maxHeight: 200, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
          {entry.error}
        </pre>
      )}

      {entry.output && (
        <div>
          <div style={{ fontSize: 12, color: "#555", marginBottom: 4 }}>Output</div>
          <pre style={{ fontSize: 12, background: "#f9fafb", padding: 8, borderRadius: 4, overflow: "auto", maxHeight: 400, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {entry.output}
          </pre>
        </div>
      )}

      <button
        className="btn-toggle"
        style={{ marginTop: 8, fontSize: 11 }}
        onClick={() => setShowTemplates((v) => !v)}
      >
        {showTemplates ? "hide templates" : "show templates"}
      </button>

      {showTemplates && (
        <div style={{ marginTop: 8 }}>
          {entry.rendered_system_prompt && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: "#555", marginBottom: 2 }}>System prompt (rendered)</div>
              <pre style={{ fontSize: 11, background: "#f9fafb", padding: 6, borderRadius: 4, overflow: "auto", maxHeight: 200, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {entry.rendered_system_prompt}
              </pre>
            </div>
          )}
          {entry.rendered_prompt && (
            <div>
              <div style={{ fontSize: 11, color: "#555", marginBottom: 2 }}>Prompt (rendered)</div>
              <pre style={{ fontSize: 11, background: "#f9fafb", padding: 6, borderRadius: 4, overflow: "auto", maxHeight: 200, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {entry.rendered_prompt}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function LlmValidateTab({ runId }: { runId: string }) {
  const [result, setResult] = useState<LlmValidateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null);
    setError(null);
    getLlmValidate(runId).then(setResult).catch((e) => setError(String(e)));
  }, [runId]);

  if (error) return <div className="err">{error}</div>;
  if (!result) return <div className="empty">loading…</div>;

  const validations = result.validations ?? [];

  return (
    <div>
      <div className="section">
        <h4>LLM Validation — {validations.length} result{validations.length !== 1 ? "s" : ""}</h4>
        <div className="muted">
          Frontier-model evaluations captured at run time. Each entry shows the validator model's output for this prompt.
        </div>
      </div>
      {validations.map((entry, i) => (
        <ValidationEntry key={i} entry={entry} index={i} />
      ))}
      {validations.length === 0 && (
        <div className="empty">no validation entries in llm_validate.json</div>
      )}
    </div>
  );
}
