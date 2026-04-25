import { useEffect, useState } from "react";

export function ConversationSnapshotsTab({ runId }: { runId: string }) {
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runId)}/tree`)
      .then((r) => r.json())
      .then((tree) => {
        const grp = tree.children.find((g: any) => g.type === "conversation_snapshots");
        setData(grp ?? null);
      })
      .catch((e) => setError(String(e)));
  }, [runId]);

  if (error) return <div className="err">{error}</div>;
  if (!data) return <div className="empty">no conversation snapshots in this run</div>;

  return (
    <div className="section">
      <h4>Conversation snapshots ({data.children.length})</h4>
      <table className="metrics-table">
        <thead>
          <tr><th>#</th><th>role</th><th>preview</th><th>attention?</th></tr>
        </thead>
        <tbody>
          {data.children.map((n: any) => (
            <tr key={n.name}>
              <td>{n.index}</td>
              <td>{n.role}</td>
              <td>{n.label.slice(5)}</td>
              <td>{n.has_attention ? "yes" : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
