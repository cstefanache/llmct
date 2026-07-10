import { useEffect, useMemo, useRef, useState } from "react";
import {
  LensLayer, LensToken, LocalModel, SteerBaseResponse, SteerEdit, SteerKind, SteerLens,
  SteerMessage, SteerPosition, SteerRunResponse, SteerSaveResponse,
  createSteerSession, getSteerLens, listLocalModels, runSteer, runSteerBase, saveSteer,
} from "./api";
import { barBg, pct, showToken } from "./LensCompareTab";
import { LayerChips, LayerRangeSlider } from "./LayerFilter";

const ROLES: SteerMessage["role"][] = ["system", "user", "assistant"];
const KIND_HINT: Record<SteerKind, string> = {
  point: "edit the residual at THIS position only — a local next-token nudge (weak on the answer for mid-prompt tokens)",
  broadcast: "add this direction at EVERY position + every generated token — strong, global push toward the target token",
  swap: "interpolate this token's INPUT embedding toward the target — the model genuinely reads a blended token here",
};
const STEER_COLOR = "rgba(220,38,38,0.22)";
const BASE_COLOR = "rgba(37,99,235,0.22)";

// ─── config bar: model + message editor ───────────────────────────────────────

function ConfigBar({
  onRun, running,
}: {
  onRun: (model: Record<string, unknown>, messages: SteerMessage[], maxNewTokens: number) => void;
  running: boolean;
}) {
  const [models, setModels] = useState<LocalModel[]>([]);
  const [modelId, setModelId] = useState("");
  const [dtype, setDtype] = useState<"float16" | "bfloat16" | "float32">("float16");
  const [maxNewTokens, setMaxNewTokens] = useState(40);
  const [messages, setMessages] = useState<SteerMessage[]>([
    { role: "user", content: "Think of a sport. Answer in one word." },
  ]);

  useEffect(() => {
    listLocalModels().then((m) => {
      setModels(m);
      if (m.length && !modelId) setModelId(m[0].model_id);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setMsg = (i: number, patch: Partial<SteerMessage>) =>
    setMessages((prev) => prev.map((m, j) => (j === i ? { ...m, ...patch } : m)));
  const addMsg = () =>
    setMessages((prev) => [...prev, { role: "user", content: "" }]);
  const delMsg = (i: number) =>
    setMessages((prev) => prev.filter((_, j) => j !== i));

  return (
    <div className="steer-config">
      <div className="controls">
        <label>
          model{" "}
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            style={{ width: 260 }}
            disabled={!models.length}
          >
            {!models.length && <option value="">no local models found</option>}
            {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id}</option>)}
          </select>
        </label>
        <label>
          dtype{" "}
          <select value={dtype} onChange={(e) => setDtype(e.target.value as typeof dtype)}>
            <option value="float16">float16</option>
            <option value="bfloat16">bfloat16</option>
            <option value="float32">float32</option>
          </select>
        </label>
        <label>
          max new tokens{" "}
          <input
            type="number" min={1} max={512} value={maxNewTokens}
            onChange={(e) => setMaxNewTokens(Math.max(1, Math.min(512, +e.target.value || 1)))}
            style={{ width: 64 }}
          />
        </label>
        <button
          className="btn-toggle steer-run"
          disabled={running || !modelId}
          onClick={() => onRun({ id: modelId, dtype, device: "auto" }, messages, maxNewTokens)}
        >
          {running ? "running…" : "▶ Run"}
        </button>
      </div>
      <div className="steer-messages">
        {messages.map((m, i) => (
          <div key={i} className="steer-msg">
            <select value={m.role} onChange={(e) => setMsg(i, { role: e.target.value as SteerMessage["role"] })}>
              {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <textarea
              value={m.content} rows={m.role === "system" ? 3 : 2}
              onChange={(e) => setMsg(i, { content: e.target.value })}
              placeholder={`${m.role} message…`}
            />
            <button className="steer-msg-del" title="remove message" onClick={() => delMsg(i)}>×</button>
          </div>
        ))}
        <button className="btn-toggle" onClick={addMsg}>+ add message</button>
      </div>
    </div>
  );
}

// ─── left conversation panel ───────────────────────────────────────────────────

function ConversationPanel({
  base, selectedPos, onSelect, steer,
}: {
  base: SteerBaseResponse;
  selectedPos: number | null;
  onSelect: (abs: number) => void;
  steer: SteerRunResponse | null;
}) {
  const prompt = base.positions.filter((p) => p.section === "prompt");
  const generated = base.positions.filter((p) => p.section === "generated");
  const lastPrompt = base.prompt_len - 1;

  const chip = (p: SteerPosition) => {
    const next = base.positions.find((q) => q.abs === p.abs + 1);
    return (
      <span
        key={p.abs}
        className={`steer-tok${p.abs === selectedPos ? " sel" : ""}${p.abs === lastPrompt ? " frontier" : ""}`}
        title={`pos ${p.abs} · ${p.section}[${p.rel}] · id=${p.token_id}${next ? ` · generates "${next.token}"` : ""}${p.abs === lastPrompt ? " · generates the first answer word" : ""}`}
        onClick={() => onSelect(p.abs)}
      >
        {showToken(p.token, p.token_id)}
      </span>
    );
  };

  return (
    <div className="steer-conv">
      <div className="steer-conv-sec">
        <div className="steer-conv-h">prompt <span className="muted">({prompt.length} tok)</span></div>
        <div className="steer-tokrow">{prompt.map(chip)}</div>
      </div>
      <div className="steer-conv-sec">
        <div className="steer-conv-h">answer — base <span className="muted">({generated.length} tok)</span></div>
        <div className="steer-tokrow">{generated.map(chip)}</div>
      </div>
      {steer && (
        <div className="steer-conv-sec">
          <div className="steer-conv-h steer-conv-h--steered">
            answer — steered
            {steer.diverged_from_base_at != null && (
              <span className="muted"> · diverges at pos {steer.diverged_from_base_at}</span>
            )}
          </div>
          <div className="steer-answer-steered">{steer.generated_text || <span className="muted">∅</span>}</div>
        </div>
      )}
    </div>
  );
}

// ─── clickable per-layer lens table ─────────────────────────────────────────────

function LensGrid({
  lens, layers, topN, color, target, onPick,
}: {
  lens: SteerLens;
  layers: number[];
  topN: number;
  color: string;
  target: { layer: number; token_id: number } | null;
  onPick?: (layer: number, tok: LensToken) => void;
}) {
  const byLayer = useMemo(() => {
    const m = new Map<number, LensLayer>();
    lens.layers.forEach((l) => m.set(l.layer, l));
    return m;
  }, [lens]);
  const present = layers.filter((l) => byLayer.has(l));
  const rows = Array.from({ length: topN }, (_, r) => r);

  return (
    <div className="lens-scroll">
      <table className="lens-table">
        <thead>
          <tr>
            <th className="lens-rank-h">#</th>
            {present.map((l) => <th key={l}>L{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <td className="lens-rank">{r + 1}</td>
              {present.map((l) => {
                const tok = byLayer.get(l)!.predicted[r];
                if (!tok) return <td key={l} className="lens-cell lens-empty">—</td>;
                const isTarget = !!target && target.layer === l && target.token_id === tok.id;
                return (
                  <td
                    key={l}
                    className={`lens-cell${onPick ? " steer-pickable" : ""}${isTarget ? " steer-target" : ""}`}
                    style={{ background: barBg(tok.prob, color) }}
                    title={`${tok.token === null ? "" : JSON.stringify(tok.token) + " "}id=${tok.id} · ${pct(tok.prob)}${onPick ? " · click to steer" : ""}`}
                    onClick={onPick ? () => onPick(l, tok) : undefined}
                  >
                    <span className="lens-tok">{showToken(tok.token, tok.id)}</span>
                    <span className="lens-prob">{pct(tok.prob)}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── main view ───────────────────────────────────────────────────────────────

export function SteeringView() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [base, setBase] = useState<SteerBaseResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [selectedPos, setSelectedPos] = useState<number | null>(null);
  const [topN, setTopN] = useState(10);
  const [baseLens, setBaseLens] = useState<SteerLens | null>(null);
  const [lensLoading, setLensLoading] = useState(false);

  const [hiddenLayers, setHiddenLayers] = useState<Set<number>>(new Set());
  const lastClicked = useRef<number | null>(null);

  const [target, setTarget] = useState<{ layer: number; token_id: number; token: string | null } | null>(null);
  const [kind, setKind] = useState<SteerKind>("point");
  const [mode, setMode] = useState<"relative" | "absolute">("relative");
  const [value, setValue] = useState(0.5);
  const [scope, setScope] = useState<"layer" | "all">("layer");

  const [steer, setSteer] = useState<SteerRunResponse | null>(null);
  const [steering, setSteering] = useState(false);
  const [saved, setSaved] = useState<SteerSaveResponse | null>(null);
  const [saving, setSaving] = useState(false);

  // sensible default value per mechanism/mode (swap weight → full swap; prob/relative → mid)
  useEffect(() => { setValue(kind === "swap" ? 1.0 : 0.5); }, [mode, kind]);

  const runBase = async (model: Record<string, unknown>, messages: SteerMessage[], maxNewTokens: number) => {
    setRunning(true); setErr(null); setBase(null); setBaseLens(null);
    setSelectedPos(null); setTarget(null); setSteer(null); setSaved(null);
    try {
      const { session_id } = await createSteerSession(model, messages, { max_new_tokens: maxNewTokens, do_sample: false });
      setSessionId(session_id);
      const b = await runSteerBase(session_id);
      setBase(b);
      // default selection: last prompt token (its lens = first generated prediction)
      const defaultPos = b.prompt_len - 1;
      setSelectedPos(defaultPos);
    } catch (e) {
      setErr(String(e));
    } finally {
      setRunning(false);
    }
  };

  // load lens whenever selection / topN changes
  useEffect(() => {
    if (!sessionId || selectedPos == null) return;
    let cancelled = false;
    setLensLoading(true); setTarget(null); setSteer(null); setSaved(null);
    getSteerLens(sessionId, selectedPos, topN)
      .then((l) => { if (!cancelled) setBaseLens(l); })
      .catch((e) => { if (!cancelled) setErr(String(e)); })
      .finally(() => { if (!cancelled) setLensLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId, selectedPos, topN]);

  // soft-swap only applies to prompt tokens; revert to point if the selection is a generated token
  const selIsPrompt = base && selectedPos != null && selectedPos < base.prompt_len;
  useEffect(() => {
    if (kind === "swap" && base && selectedPos != null && !selIsPrompt) setKind("point");
  }, [kind, base, selectedPos, selIsPrompt]);

  const allLayers = useMemo(() => {
    const s = new Set<number>();
    baseLens?.layers.forEach((l) => s.add(l.layer));
    return [...s].sort((a, b) => a - b);
  }, [baseLens]);
  const shownLayers = allLayers.filter((l) => !hiddenLayers.has(l));

  const toggleLayer = (layer: number, shift: boolean) => {
    setHiddenLayers((prev) => {
      const next = new Set(prev);
      const willHide = !next.has(layer);
      const from = shift && lastClicked.current !== null ? lastClicked.current : layer;
      const lo = Math.min(from, layer), hi = Math.max(from, layer);
      for (const l of allLayers) if (l >= lo && l <= hi) { willHide ? next.add(l) : next.delete(l); }
      return next;
    });
    lastClicked.current = layer;
  };
  const rangeFrom = shownLayers.length ? shownLayers[0] : (allLayers[0] ?? 0);
  const rangeTo = shownLayers.length ? shownLayers[shownLayers.length - 1] : (allLayers[allLayers.length - 1] ?? 0);
  const setRange = (from: number, to: number) =>
    setHiddenLayers(new Set(allLayers.filter((l) => l < from || l > to)));

  const doSteer = async () => {
    if (!sessionId || selectedPos == null || !target) return;
    let edit: SteerEdit;
    if (kind === "swap") {
      edit = {
        token_id: target.token_id, position: selectedPos,
        scope: "all", layers: [], mode: "absolute", value, kind: "swap",
      };
    } else {
      const layers = scope === "layer" ? [target.layer] : shownLayers;
      if (!layers.length) { setErr("no layers selected for 'all' scope"); return; }
      edit = { token_id: target.token_id, position: selectedPos, scope, layers, mode, value, kind };
    }
    setSteering(true); setErr(null); setSaved(null);
    try {
      setSteer(await runSteer(sessionId, edit, topN));
    } catch (e) {
      setErr(String(e));
    } finally {
      setSteering(false);
    }
  };

  const doSave = async () => {
    if (!sessionId) return;
    setSaving(true); setErr(null);
    try {
      setSaved(await saveSteer(sessionId));
    } catch (e) {
      setErr(String(e));
    } finally {
      setSaving(false);
    }
  };

  const selPos = base && selectedPos != null ? base.positions[selectedPos] : null;

  return (
    <div className="steer-view">
      <ConfigBar onRun={runBase} running={running} />
      {err && <div className="err">{err}</div>}

      {!base && !running && (
        <div className="empty">configure a model + prompt above and hit Run to start steering</div>
      )}

      {base && (
        <div className="steer-body">
          <div className="steer-left">
            <ConversationPanel base={base} selectedPos={selectedPos} onSelect={setSelectedPos} steer={steer} />
          </div>

          <div className="steer-main">
            {selPos && (() => {
              const next = base.positions.find((p) => p.abs === selPos.abs + 1);
              const lastPrompt = base.prompt_len - 1;
              const isFrontier = selPos.abs === lastPrompt;
              const isAnswer = selPos.section === "generated";
              return (
                <div className="steer-selinfo">
                  <div>
                    steering position <b>{selPos.abs}</b> ({selPos.section}[{selPos.rel}]) — token{" "}
                    <code>{showToken(selPos.token, selPos.token_id)}</code>
                    {next && <> → generates <code>{showToken(next.token, next.token_id)}</code></>}.
                    Its per-layer next-token lens is below; click a cell to pick the token to steer.
                  </div>
                  {isFrontier ? (
                    <div className="steer-hint steer-hint--ok">
                      This is the last prompt token — it produces the <b>first answer word</b>, so steering
                      it changes the answer directly.
                    </div>
                  ) : isAnswer ? (
                    <div className="steer-hint steer-hint--ok">
                      This is an answer token — steering it changes this word and everything the model
                      generates after it.
                    </div>
                  ) : (
                    <div className="steer-hint steer-hint--warn">
                      This is a mid-prompt token. Steering it only changes what token follows <i>it</i>; it
                      reaches the final answer weakly, via attention, so the answer often won't change. To
                      change the answer, click the last prompt token (
                      <button className="steer-jump" onClick={() => setSelectedPos(lastPrompt)}>pos {lastPrompt}</button>
                      ) or an answer token.
                    </div>
                  )}
                </div>
              );
            })()}

            {allLayers.length > 0 && (
              <>
                <div className="controls">
                  <label>
                    Token depth{" "}
                    <input
                      type="number" min={1} max={30} value={topN}
                      onChange={(e) => setTopN(Math.max(1, Math.min(30, +e.target.value || 1)))}
                      style={{ width: 52 }}
                    />
                  </label>
                  {lensLoading && <span className="muted">computing lens… (first call loads the model)</span>}
                </div>
                <LayerChips
                  layers={allLayers} hidden={hiddenLayers}
                  onToggle={toggleLayer} onShowAll={() => setHiddenLayers(new Set())}
                  onHideAll={() => setHiddenLayers(new Set(allLayers))}
                />
                <LayerRangeSlider
                  min={allLayers[0]} max={allLayers[allLayers.length - 1]}
                  from={rangeFrom} to={rangeTo} onChange={setRange}
                />
              </>
            )}

            {baseLens && (
              <div className="section">
                <h4>Base lens {steer ? "" : "· click a token cell to steer it"}</h4>
                <LensGrid
                  lens={baseLens} layers={shownLayers} topN={topN} color={BASE_COLOR}
                  target={target} onPick={(layer, tok) => setTarget({ layer, token_id: tok.id, token: tok.token })}
                />
              </div>
            )}

            {target && (
              <div className="steer-controls section">
                <div className="steer-controls-row">
                  <span>mechanism</span>
                  <span className="steer-mode">
                    <button className={`btn-toggle ${kind === "point" ? "lens-mode-on" : ""}`} onClick={() => setKind("point")}>point</button>
                    <button className={`btn-toggle ${kind === "broadcast" ? "lens-mode-on" : ""}`} onClick={() => setKind("broadcast")}>broadcast</button>
                    <button
                      className={`btn-toggle ${kind === "swap" ? "lens-mode-on" : ""}`}
                      disabled={!selIsPrompt}
                      title={selIsPrompt ? "soft token swap" : "soft-swap applies to prompt tokens only"}
                      onClick={() => setKind("swap")}
                    >soft-swap</button>
                  </span>
                  <span className="muted steer-hint">{KIND_HINT[kind]}</span>
                </div>

                {kind === "swap" ? (
                  <>
                    <div className="steer-controls-row">
                      <span>
                        swap <code>{showToken(selPos?.token ?? null, selPos?.token_id ?? 0)}</code> →{" "}
                        <code>{showToken(target.token, target.token_id)}</code>
                      </span>
                    </div>
                    <div className="steer-controls-row">
                      <input
                        type="range" min={0} max={1} step={0.01}
                        value={value} onChange={(e) => setValue(+e.target.value)}
                        className="steer-slider"
                      />
                      <span className="steer-val">w = {value.toFixed(2)}</span>
                      <button className="btn-toggle steer-run" disabled={steering} onClick={doSteer}>
                        {steering ? "swapping…" : "▶ Run swap"}
                      </button>
                      <span className="muted steer-hint">0 = original token · 1 = fully the target token</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="steer-controls-row">
                      <span>
                        target token <code>{showToken(target.token, target.token_id)}</code>
                        {" "}at{" "}
                        {scope === "layer" ? <b>layer {target.layer}</b> : <b>all shown layers ({shownLayers.length})</b>}
                        {kind === "broadcast" && <> · <span className="muted">every position</span></>}
                      </span>
                      <span className="steer-mode">
                        <button className={`btn-toggle ${mode === "relative" ? "lens-mode-on" : ""}`} onClick={() => setMode("relative")}>relative</button>
                        <button className={`btn-toggle ${mode === "absolute" ? "lens-mode-on" : ""}`} onClick={() => setMode("absolute")}>absolute</button>
                      </span>
                      <span className="steer-mode">
                        <button className={`btn-toggle ${scope === "layer" ? "lens-mode-on" : ""}`} onClick={() => setScope("layer")}>this layer</button>
                        <button className={`btn-toggle ${scope === "all" ? "lens-mode-on" : ""}`} onClick={() => setScope("all")}>all layers</button>
                      </span>
                    </div>
                    <div className="steer-controls-row">
                      <input
                        type="range"
                        min={mode === "relative" ? -1 : 0} max={1} step={0.01}
                        value={value} onChange={(e) => setValue(+e.target.value)}
                        className="steer-slider"
                      />
                      <span className="steer-val">
                        {mode === "relative"
                          ? `t = ${value >= 0 ? "+" : ""}${value.toFixed(2)}`
                          : `p = ${(value * 100).toFixed(0)}%`}
                      </span>
                      <button className="btn-toggle steer-run" disabled={steering} onClick={doSteer}>
                        {steering ? "steering…" : "▶ Run steer"}
                      </button>
                      <span className="muted steer-hint">
                        {kind === "broadcast"
                          ? "α solved at the answer frontier, then added everywhere — high targets can destabilize output"
                          : mode === "relative"
                            ? "−1 fully suppress · 0 unchanged · +1 drive to 1.0"
                            : "absolute target probability over the full vocab"}
                      </span>
                    </div>
                  </>
                )}
              </div>
            )}

            {steer && (
              <div className="section">
                <h4>Steered result <span className="muted">· {steer.kind}</span></h4>
                {steer.kind === "swap" && steer.swap ? (
                  <div className="steer-hint steer-hint--ok">
                    soft-swapped <code>{steer.swap.from_token}</code> → <code>{steer.swap.to_token}</code> at
                    weight <b>{steer.swap.weight.toFixed(2)}</b>
                    {steer.diverged_from_base_at != null
                      ? <> · answer diverges at pos {steer.diverged_from_base_at}</>
                      : <> · answer unchanged</>}
                  </div>
                ) : (
                  <SolvedTable solved={steer.solved} />
                )}
                <h4 style={{ marginTop: 12 }}>
                  Steered lens <span className="muted">· at pos {steer.lens_position}</span>
                </h4>
                <LensGrid lens={steer.lens} layers={shownLayers} topN={topN} color={STEER_COLOR} target={target} />
                <div className="steer-save-row">
                  <button className="btn-toggle" disabled={saving} onClick={doSave}>
                    {saving ? "saving…" : "💾 Save base + steered snapshots"}
                  </button>
                  {saved && (
                    <span className="muted">
                      saved <code>{saved.human_label}</code> → open <b>Comparison</b> and pick{" "}
                      <code>{saved.base_run_id}</code> vs <code>{saved.steered_run_id}</code>
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function SolvedTable({ solved }: { solved: SteerRunResponse["solved"] }) {
  if (!solved.length) return <div className="muted">no layers solved</div>;
  return (
    <div className="lens-scroll">
      <table className="lens-table steer-solved">
        <thead>
          <tr><th>layer</th><th>p₀</th><th>target</th><th>achieved</th><th>α</th><th></th></tr>
        </thead>
        <tbody>
          {solved.map((s) => (
            <tr key={s.layer}>
              <td>L{s.layer}</td>
              <td>{pct(s.p0)}</td>
              <td>{pct(s.target_prob)}</td>
              <td>{pct(s.achieved_prob)}</td>
              <td>{s.alpha.toFixed(2)}</td>
              <td>{s.saturated && <span className="steer-sat" title="target unreachable — alpha capped">saturated</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
