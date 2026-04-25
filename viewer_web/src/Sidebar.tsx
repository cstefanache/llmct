import { useEffect, useState } from "react";
import { RunSummary, TreeNode, TreeGroup, RunTree, listRuns, getTree, NpzRef } from "./api";

export interface Selection {
  kind: "run" | "group" | "npz";
  runId: string;
  groupType?: TreeGroup["type"];
  npz?: TreeNode;
}

export interface SidebarProps {
  selected: Selection | null;
  selectedNpz: NpzRef[];
  onSelect: (s: Selection) => void;
  onToggleNpz: (ref: NpzRef, node: TreeNode) => void;
}

export function Sidebar({ selected, selectedNpz, onSelect, onToggleNpz }: SidebarProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [openRuns, setOpenRuns] = useState<Set<string>>(new Set());
  const [openGroups, setOpenGroups] = useState<Set<string>>(new Set());
  const [trees, setTrees] = useState<Record<string, RunTree>>({});

  useEffect(() => {
    listRuns().then(setRuns).catch(console.error);
  }, []);

  const toggleRun = async (id: string) => {
    const next = new Set(openRuns);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
      if (!trees[id]) {
        try {
          const t = await getTree(id);
          setTrees((prev) => ({ ...prev, [id]: t }));
        } catch (e) {
          console.error(e);
        }
      }
    }
    setOpenRuns(next);
  };

  const toggleGroup = (key: string) => {
    const next = new Set(openGroups);
    next.has(key) ? next.delete(key) : next.add(key);
    setOpenGroups(next);
  };

  const isChecked = (runId: string, node: TreeNode) =>
    selectedNpz.some((n) => n.run_id === runId && n.kind === node.kind && n.name === node.name);

  return (
    <aside className="sidebar">
      <h3>Runs</h3>
      <div className="tree">
        {runs.map((r) => {
          const isOpen = openRuns.has(r.id);
          const runSelected = selected?.kind === "run" && selected.runId === r.id;
          return (
            <div key={r.id}>
              <div
                className={`node ${runSelected ? "selected" : ""}`}
                onClick={() => {
                  toggleRun(r.id);
                  onSelect({ kind: "run", runId: r.id });
                }}
              >
                <span className="chev">{isOpen ? "▾" : "▸"}</span>
                <span>{r.id}</span>
              </div>
              {isOpen && trees[r.id]?.children.map((group) => {
                const gkey = `${r.id}::${group.type}`;
                const gOpen = openGroups.has(gkey);
                const gSelected = selected?.kind === "group" && selected.runId === r.id && selected.groupType === group.type;
                return (
                  <div key={gkey}>
                    <div
                      className={`node indent-1 ${gSelected ? "selected" : ""}`}
                      onClick={() => {
                        toggleGroup(gkey);
                        onSelect({ kind: "group", runId: r.id, groupType: group.type });
                      }}
                    >
                      <span className="chev">{gOpen ? "▾" : "▸"}</span>
                      <span>{group.type} <span className="muted">({group.children.length})</span></span>
                    </div>
                    {gOpen && group.children.map((n) => {
                      const ref: NpzRef = { run_id: r.id, kind: n.kind, name: n.name };
                      const npzSel = selected?.kind === "npz" && selected.runId === r.id && selected.npz?.name === n.name;
                      return (
                        <div
                          key={`${gkey}::${n.name}`}
                          className={`node leaf indent-2 ${npzSel ? "selected" : ""}`}
                          onClick={(e) => {
                            if ((e.target as HTMLElement).tagName === "INPUT") return;
                            onSelect({ kind: "npz", runId: r.id, npz: n });
                          }}
                        >
                          <span className="chev" />
                          <input
                            type="checkbox"
                            className="check"
                            checked={isChecked(r.id, n)}
                            onChange={() => onToggleNpz(ref, n)}
                          />
                          <span title={n.label}>{n.label}</span>
                          {n.has_attention === true && <span className="pill attn">attn</span>}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
