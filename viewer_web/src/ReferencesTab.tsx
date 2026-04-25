import { useEffect, useState } from "react";

export function ReferencesTab({ runId }: { runId: string }) {
  const [labels, setLabels] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/tree`)
      .then((r) => r.json())
      .then((tree) => {
        const grp = tree.children.find((g: any) => g.type === "references");
        setLabels(grp ? grp.children.map((c: any) => c.label) : []);
      })
      .catch((e) => setError(String(e)));
  }, [runId]);

  if (error) return <div className="err">{error}</div>;
  if (!labels) return <div className="empty">loading…</div>;
  if (!labels.length) return <div className="empty">no references in this run</div>;

  return (
    <div className="section">
      <h4>Reference labels ({labels.length})</h4>
      <ul>{labels.map((l) => <li key={l}>{l}</li>)}</ul>
    </div>
  );
}
