import { useMemo, useState } from "react";
import { Sidebar, Selection } from "./Sidebar";
import { NpzRef, NpzMeta, TreeNode, getNpzMeta } from "./api";
import { RunOverviewTab } from "./RunOverviewTab";
import { ConversationSnapshotsTab } from "./ConversationSnapshotsTab";
import { ReferencesTab } from "./ReferencesTab";
import { NpzAttentionTab } from "./NpzAttentionTab";
import { PairCompareTab } from "./PairCompareTab";
import { MultiCompareTab } from "./MultiCompareTab";
import { pairTabLabel } from "./refLabels";

const SOURCES = ["hidden_in", "hidden_out", "attn_out", "mlp_down_out", "qkv_last"] as const;

const MULTI_TAB_ID = "multi";

type ActiveTabKey =
  | { kind: "run"; runId: string }
  | { kind: "group"; runId: string; groupType: "conversation_snapshots" | "references" | "tensors" }
  | { kind: "npz"; ref: NpzRef }
  | { kind: "pair"; a: NpzRef; b: NpzRef }
  | { kind: "multi"; refs: NpzRef[] };

function tabId(t: ActiveTabKey): string {
  if (t.kind === "run") return `run:${t.runId}`;
  if (t.kind === "group") return `group:${t.runId}:${t.groupType}`;
  if (t.kind === "npz") return `npz:${t.ref.run_id}:${t.ref.kind}:${t.ref.name}`;
  if (t.kind === "multi") return MULTI_TAB_ID;
  return `pair:${t.a.run_id}:${t.a.kind}:${t.a.name}::${t.b.run_id}:${t.b.kind}:${t.b.name}`;
}

function tabLabel(t: ActiveTabKey): string {
  if (t.kind === "run") return `run ${t.runId.slice(0, 18)}…`;
  if (t.kind === "group") return t.groupType;
  if (t.kind === "npz") return `${t.ref.kind}:${t.ref.name}`;
  if (t.kind === "multi") return `compare all (${t.refs.length})`;
  return pairTabLabel(t.a, t.b);
}

export function App() {
  const [selected, setSelected] = useState<Selection | null>(null);
  const [selectedNpz, setSelectedNpz] = useState<NpzRef[]>([]);
  const [npzMetas, setNpzMetas] = useState<Record<string, NpzMeta>>({});
  const [showKeys, setShowKeys] = useState(false);
  const [sources, setSources] = useState<string[]>(["hidden_out"]);
  const [tabs, setTabs] = useState<ActiveTabKey[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [pinnedTabs, setPinnedTabs] = useState<Set<string>>(new Set());

  const metaKey = (ref: NpzRef) => `${ref.run_id}|${ref.kind}|${ref.name}`;

  const addTab = (t: ActiveTabKey) => {
    const id = tabId(t);
    setTabs((prev) => prev.find((x) => tabId(x) === id) ? prev : [...prev, t]);
    setActiveTab(id);
  };

  const openSelection = (t: ActiveTabKey) => {
    const id = tabId(t);
    setTabs((prev) => {
      const pinned = prev.filter((x) => pinnedTabs.has(tabId(x)));
      if (pinned.find((x) => tabId(x) === id)) return pinned;
      return [...pinned, t];
    });
    setActiveTab(id);
  };

  const closeTab = (id: string) => {
    setTabs((prev) => prev.filter((t) => tabId(t) !== id));
    setPinnedTabs((prev) => { const next = new Set(prev); next.delete(id); return next; });
    if (activeTab === id) setActiveTab(null);
  };

  const togglePin = (id: string) => {
    setPinnedTabs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const onSelect = (s: Selection) => {
    setSelected(s);
    if (s.kind === "run") openSelection({ kind: "run", runId: s.runId });
    else if (s.kind === "group" && s.groupType) openSelection({ kind: "group", runId: s.runId, groupType: s.groupType });
    else if (s.kind === "npz" && s.npz) openSelection({ kind: "npz", ref: { run_id: s.runId, kind: s.npz.kind, name: s.npz.name } });
  };

  const onToggleNpz = (ref: NpzRef, node: TreeNode) => {
    const rk = metaKey(ref);
    const isAdding = !selectedNpz.find((p) => p.run_id === ref.run_id && p.kind === ref.kind && p.name === ref.name);
    if (isAdding) {
      getNpzMeta(ref).then((m) => setNpzMetas((prev) => ({ ...prev, [rk]: m }))).catch(console.error);
    } else {
      setNpzMetas((prev) => { const next = { ...prev }; delete next[rk]; return next; });
    }

    setSelectedNpz((prev) => {
      const has = prev.find((p) => p.run_id === ref.run_id && p.kind === ref.kind && p.name === ref.name);
      const next = has ? prev.filter((p) => p !== has) : [...prev, ref];

      // open a per-npz tab on first check
      if (!has) addTab({ kind: "npz", ref });

      // when 2+ selected, generate pair tabs for any new pair not already present
      if (next.length >= 2) {
        for (let i = 0; i < next.length; i++) {
          for (let j = i + 1; j < next.length; j++) {
            const a = next[i];
            const b = next[j];
            // simple policy: always add a pair tab; the backend/frontend will show per-source metrics
            const id = tabId({ kind: "pair", a, b });
            setTabs((p) => p.find((t) => tabId(t) === id) ? p : [...p, { kind: "pair", a, b }]);
          }
        }
        // singleton "compare all" tab — replace its refs to track current selection
        setTabs((p) => {
          const without = p.filter((t) => t.kind !== "multi");
          return [...without, { kind: "multi", refs: next }];
        });
      } else {
        // dropped below 2 selected — drop the multi tab if present
        setTabs((p) => p.filter((t) => t.kind !== "multi"));
        setPinnedTabs((pp) => {
          if (!pp.has(MULTI_TAB_ID)) return pp;
          const nx = new Set(pp); nx.delete(MULTI_TAB_ID); return nx;
        });
        setActiveTab((cur) => (cur === MULTI_TAB_ID ? null : cur));
      }

      return next;
    });
  };

  const toggleSource = (src: string) => {
    setSources((prev) => (prev.includes(src) ? prev.filter((s) => s !== src) : [...prev, src]));
  };

  const renderTab = (t: ActiveTabKey) => {
    if (t.kind === "run") return <RunOverviewTab runId={t.runId} />;
    if (t.kind === "group") {
      if (t.groupType === "conversation_snapshots") return <ConversationSnapshotsTab runId={t.runId} />;
      if (t.groupType === "references") return <ReferencesTab runId={t.runId} />;
      return <div className="empty">tensors group — pick a step file from the sidebar</div>;
    }
    if (t.kind === "npz") return <NpzAttentionTab npz={t.ref} />;
    if (t.kind === "multi") return <MultiCompareTab refs={t.refs} sources={sources} />;
    return <PairCompareTab a={t.a} b={t.b} sources={sources} />;
  };

  const active = useMemo(() => tabs.find((t) => tabId(t) === activeTab) ?? null, [tabs, activeTab]);

  return (
    <div className="app">
      <Sidebar
        selected={selected}
        selectedNpz={selectedNpz}
        onSelect={onSelect}
        onToggleNpz={onToggleNpz}
      />
      <main className="main">
        <div className="header">
          <span className="title">Activation Lab</span>
          {SOURCES.map((src) => (
            <label key={src}>
              <input
                type="checkbox"
                checked={sources.includes(src)}
                onChange={() => toggleSource(src)}
              />
              {src}
            </label>
          ))}
          <span className="muted">{selectedNpz.length} npz selected</span>
          {selectedNpz.length > 0 && (
            <button className="btn-toggle" onClick={() => setShowKeys((v) => !v)}>
              {showKeys ? "hide keys" : "show keys"}
            </button>
          )}
        </div>

        {showKeys && selectedNpz.length > 0 && (
          <div className="keys-panel">
            {selectedNpz.map((ref) => {
              const rk = metaKey(ref);
              const meta = npzMetas[rk];
              return (
                <div key={rk} className="keys-entry">
                  <span className="keys-label">{ref.kind}:{ref.name}</span>
                  {meta ? (
                    <div className="keys-list">
                      {meta.keys.map((k) => <span key={k} className="key-chip">{k}</span>)}
                    </div>
                  ) : (
                    <span className="muted">loading…</span>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div className="tab-area">
          <div className="tabs">
            {tabs.map((t) => {
              const id = tabId(t);
              const pinned = pinnedTabs.has(id);
              return (
                <div
                  key={id}
                  className={`tab ${id === activeTab ? "active" : ""} ${pinned ? "pinned" : ""}`}
                  onClick={() => setActiveTab(id)}
                >
                  <span
                    className="pin"
                    title={pinned ? "Unpin tab" : "Pin tab"}
                    onClick={(e) => { e.stopPropagation(); togglePin(id); }}
                  >{pinned ? "●" : "○"}</span>
                  {tabLabel(t)}
                  <span className="close" onClick={(e) => { e.stopPropagation(); closeTab(id); }}>×</span>
                </div>
              );
            })}
          </div>

          <div className="tabpanel">
            {active ? renderTab(active) : <div className="empty">select something from the sidebar</div>}
          </div>
        </div>
      </main>
    </div>
  );
}
