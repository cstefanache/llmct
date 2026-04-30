import { useEffect, useState } from "react";
import { listLocalModels, LocalModel } from "../api";
import { defaultPrompt, Message, PromptConfig, ScenarioModel } from "./scenarioModel";

interface Props {
  scenario: ScenarioModel;
  onChange: (s: ScenarioModel) => void;
  disabled?: boolean;
}

export function ScenarioForm({ scenario, onChange, disabled }: Props) {
  const [localModels, setLocalModels] = useState<LocalModel[]>([]);

  useEffect(() => {
    listLocalModels().then(setLocalModels).catch(() => {});
  }, []);

  const set = <K extends keyof ScenarioModel>(key: K, val: ScenarioModel[K]) =>
    onChange({ ...scenario, [key]: val });

  const setModel = (patch: Partial<ScenarioModel["model"]>) =>
    set("model", { ...scenario.model, ...patch });

  const setGeneration = (patch: Partial<ScenarioModel["generation"]>) =>
    set("generation", { ...scenario.generation, ...patch });

  const setCapture = (patch: Partial<ScenarioModel["capture"]>) =>
    set("capture", { ...scenario.capture, ...patch });

  const setOutput = (patch: Partial<ScenarioModel["output"]>) =>
    set("output", { ...scenario.output, ...patch });

  const updatePrompt = (pi: number, patch: Partial<PromptConfig>) => {
    const next = scenario.prompt.map((p, idx) => idx === pi ? { ...p, ...patch } : p);
    set("prompt", next);
  };

  const addPrompt = () => set("prompt", [...scenario.prompt, defaultPrompt()]);

  const removePrompt = (pi: number) => {
    if (scenario.prompt.length <= 1) return;
    set("prompt", scenario.prompt.filter((_, idx) => idx !== pi));
  };

  const updateMessage = (pi: number, mi: number, patch: Partial<Message>) => {
    const msgs = scenario.prompt[pi].messages.map((m, idx) => idx === mi ? { ...m, ...patch } : m);
    updatePrompt(pi, { messages: msgs });
  };

  const addMessage = (pi: number) =>
    updatePrompt(pi, { messages: [...scenario.prompt[pi].messages, { role: "user", content: "" }] });

  const removeMessage = (pi: number, mi: number) =>
    updatePrompt(pi, { messages: scenario.prompt[pi].messages.filter((_, idx) => idx !== mi) });

  const addRefState = () =>
    set("reference_states", [
      ...scenario.reference_states,
      { label: "", messages: [{ role: "user", content: "" }] },
    ]);

  const removeRefState = (i: number) =>
    set("reference_states", scenario.reference_states.filter((_, idx) => idx !== i));

  const updateRefState = (i: number, patch: Partial<ScenarioModel["reference_states"][0]>) => {
    const refs = scenario.reference_states.map((r, idx) => idx === i ? { ...r, ...patch } : r);
    set("reference_states", refs);
  };

  const updateRefMsg = (ri: number, mi: number, patch: Partial<Message>) => {
    const refs = scenario.reference_states.map((r, idx) => {
      if (idx !== ri) return r;
      const msgs = r.messages.map((m, j) => j === mi ? { ...m, ...patch } : m);
      return { ...r, messages: msgs };
    });
    set("reference_states", refs);
  };

  const layersIsAll = scenario.capture.layers === "all";
  const layersText = layersIsAll
    ? ""
    : (scenario.capture.layers as number[]).join(", ");

  const knownIds = new Set(localModels.map((m) => m.model_id));
  const idInList = knownIds.has(scenario.model.id);

  return (
    <div className="scenario-form" style={{ opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? "none" : undefined }}>

      {/* Identity */}
      <div className="section">
        <h4>Identity</h4>
        <div className="form-row">
          <label>Name</label>
          <input
            type="text"
            value={scenario.name}
            onChange={(e) => set("name", e.target.value)}
            placeholder="my_scenario"
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* Model */}
      <div className="section">
        <h4>Model</h4>
        <div className="form-row">
          <label>
            Model id
            {localModels.length === 0 && <span className="muted"> — no local HF models found, enter manually</span>}
          </label>
          {localModels.length > 0 && (
            <select
              value={idInList ? scenario.model.id : ""}
              onChange={(e) => { if (e.target.value) setModel({ id: e.target.value }); }}
              style={{ width: "100%" }}
            >
              <option value="">— select a locally cached model —</option>
              {localModels.map((m) => (
                <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
              ))}
            </select>
          )}
          {(!idInList || localModels.length === 0) && (
            <input
              type="text"
              value={scenario.model.id}
              onChange={(e) => setModel({ id: e.target.value })}
              placeholder="Qwen/Qwen2.5-3B-Instruct"
              style={{ width: "100%", marginTop: localModels.length > 0 ? 4 : 0 }}
            />
          )}
        </div>
        <div className="controls" style={{ flexWrap: "wrap" }}>
          <label>
            dtype
            <select value={scenario.model.dtype} onChange={(e) => setModel({ dtype: e.target.value as ScenarioModel["model"]["dtype"] })}>
              <option>float16</option>
              <option>bfloat16</option>
              <option>float32</option>
            </select>
          </label>
          <label>
            device
            <select value={scenario.model.device} onChange={(e) => setModel({ device: e.target.value as ScenarioModel["model"]["device"] })}>
              <option>auto</option>
              <option>mps</option>
              <option>cpu</option>
            </select>
          </label>
          <label>
            <input type="checkbox" checked={scenario.model.trust_remote_code} onChange={(e) => setModel({ trust_remote_code: e.target.checked })} />
            {" "}trust remote code
          </label>
        </div>
        <div className="form-row">
          <label>GGUF file <span className="muted">(optional)</span></label>
          <input
            type="text"
            value={scenario.model.gguf_file ?? ""}
            onChange={(e) => setModel({ gguf_file: e.target.value || null })}
            placeholder="path/to/model.gguf"
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* Prompts */}
      <div className="section">
        <h4>Prompts <span className="muted">(each runs independently with its own npz output)</span></h4>
        {scenario.prompt.map((prompt, pi) => (
          <div key={pi} className="ref-state-block">
            <div className="controls" style={{ justifyContent: "space-between" }}>
              <strong>Prompt {pi + 1}</strong>
              <div className="controls">
                <label>
                  <input type="checkbox" checked={prompt.run_at_each_message}
                    onChange={(e) => updatePrompt(pi, { run_at_each_message: e.target.checked })} />
                  {" "}run at each message
                </label>
                <button
                  className="btn-toggle"
                  onClick={() => removePrompt(pi)}
                  disabled={scenario.prompt.length <= 1}
                  title={scenario.prompt.length <= 1 ? "At least one prompt is required" : "Remove prompt"}
                >
                  remove
                </button>
              </div>
            </div>
            {prompt.messages.map((msg, mi) => (
              <div key={mi} className="message-row" style={{ paddingLeft: 12 }}>
                <select
                  value={msg.role}
                  onChange={(e) => updateMessage(pi, mi, { role: e.target.value as Message["role"] })}
                  style={{ width: 90, flexShrink: 0 }}
                >
                  <option>system</option>
                  <option>user</option>
                  <option>assistant</option>
                </select>
                <textarea
                  value={msg.content}
                  onChange={(e) => updateMessage(pi, mi, { content: e.target.value })}
                  rows={2}
                  style={{ flex: 1, fontFamily: "monospace", fontSize: 12 }}
                />
                <button className="btn-toggle" onClick={() => removeMessage(pi, mi)} title="Remove message" style={{ flexShrink: 0 }}>×</button>
              </div>
            ))}
            <button className="btn-toggle" onClick={() => addMessage(pi)} style={{ marginLeft: 12, marginTop: 4 }}>+ message</button>
          </div>
        ))}
        <button className="btn-toggle" onClick={addPrompt} style={{ marginTop: 6 }}>+ prompt</button>
      </div>

      {/* Generation */}
      <div className="section">
        <h4>Generation</h4>
        <div className="controls" style={{ flexWrap: "wrap" }}>
          <label>
            max new tokens
            <input type="number" min={1} value={scenario.generation.max_new_tokens}
              onChange={(e) => setGeneration({ max_new_tokens: parseInt(e.target.value) || 1 })}
              style={{ width: 70 }} />
          </label>
          <label>
            seed
            <input type="number" value={scenario.generation.seed}
              onChange={(e) => setGeneration({ seed: parseInt(e.target.value) ?? 42 })}
              style={{ width: 70 }} />
          </label>
          <label>
            <input type="checkbox" checked={scenario.generation.do_sample}
              onChange={(e) => setGeneration({ do_sample: e.target.checked })} />
            {" "}do sample
          </label>
          {scenario.generation.do_sample && (
            <label>
              temperature
              <input type="number" step={0.1} min={0} value={scenario.generation.temperature}
                onChange={(e) => setGeneration({ temperature: parseFloat(e.target.value) || 1 })}
                style={{ width: 70 }} />
            </label>
          )}
          <label>
            top_k
            <input type="number" min={1} value={scenario.generation.top_k ?? ""}
              placeholder="none"
              onChange={(e) => setGeneration({ top_k: e.target.value ? parseInt(e.target.value) : null })}
              style={{ width: 70 }} />
          </label>
          <label>
            top_p
            <input type="number" step={0.05} min={0} max={1} value={scenario.generation.top_p ?? ""}
              placeholder="none"
              onChange={(e) => setGeneration({ top_p: e.target.value ? parseFloat(e.target.value) : null })}
              style={{ width: 70 }} />
          </label>
        </div>
      </div>

      {/* Capture */}
      <div className="section">
        <h4>Capture</h4>
        <div className="controls" style={{ flexWrap: "wrap" }}>
          {(["hidden_states", "attention_weights", "qkv", "mlp", "logits"] as const).map((key) => (
            <label key={key}>
              <input type="checkbox" checked={scenario.capture[key] as boolean}
                onChange={(e) => setCapture({ [key]: e.target.checked })} />
              {" "}{key.replace(/_/g, " ")}
            </label>
          ))}
        </div>
        <div className="controls" style={{ flexWrap: "wrap", marginTop: 6 }}>
          <label>
            top k probs
            <input type="number" min={0} value={scenario.capture.top_k_probs}
              onChange={(e) => setCapture({ top_k_probs: parseInt(e.target.value) ?? 20 })}
              style={{ width: 60 }} />
          </label>
          <label>
            store dtype
            <select value={scenario.capture.store_dtype}
              onChange={(e) => setCapture({ store_dtype: e.target.value as ScenarioModel["capture"]["store_dtype"] })}>
              <option>float16</option>
              <option>bfloat16</option>
              <option>float32</option>
            </select>
          </label>
        </div>
        <div className="controls" style={{ marginTop: 6 }}>
          <label>layers</label>
          <label>
            <input type="radio" checked={layersIsAll}
              onChange={() => setCapture({ layers: "all" })} />
            {" "}all
          </label>
          <label>
            <input type="radio" checked={!layersIsAll}
              onChange={() => setCapture({ layers: [] })} />
            {" "}specific
          </label>
          {!layersIsAll && (
            <input type="text" value={layersText}
              placeholder="0, 5, 10, 31"
              onChange={(e) => {
                const nums = e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n));
                setCapture({ layers: nums });
              }}
              style={{ width: 160 }} />
          )}
        </div>
      </div>

      {/* Output */}
      <div className="section">
        <h4>Output</h4>
        <div className="form-row">
          <label>dir</label>
          <input type="text" value={scenario.output.dir}
            onChange={(e) => setOutput({ dir: e.target.value })}
            style={{ width: "100%" }} />
        </div>
      </div>

      {/* Reference States */}
      <div className="section">
        <h4>Reference states</h4>
        {scenario.reference_states.map((ref, ri) => (
          <div key={ri} className="ref-state-block">
            <div className="controls">
              <label>label</label>
              <input type="text" value={ref.label}
                onChange={(e) => updateRefState(ri, { label: e.target.value })}
                style={{ width: 160 }} />
              <button className="btn-toggle" onClick={() => removeRefState(ri)}>remove</button>
            </div>
            {ref.messages.map((msg, mi) => (
              <div key={mi} className="message-row" style={{ paddingLeft: 12 }}>
                <select value={msg.role}
                  onChange={(e) => updateRefMsg(ri, mi, { role: e.target.value as Message["role"] })}
                  style={{ width: 90, flexShrink: 0 }}>
                  <option>system</option>
                  <option>user</option>
                  <option>assistant</option>
                </select>
                <textarea value={msg.content} rows={1}
                  onChange={(e) => updateRefMsg(ri, mi, { content: e.target.value })}
                  style={{ flex: 1, fontFamily: "monospace", fontSize: 12 }} />
              </div>
            ))}
            <button className="btn-toggle" style={{ marginLeft: 12, marginTop: 4 }}
              onClick={() => updateRefState(ri, { messages: [...ref.messages, { role: "user", content: "" }] })}>
              + message
            </button>
          </div>
        ))}
        <button className="btn-toggle" style={{ marginTop: 6 }} onClick={addRefState}>+ reference state</button>
      </div>
    </div>
  );
}
