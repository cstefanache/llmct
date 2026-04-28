import yaml from "js-yaml";

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface PromptConfig {
  messages: Message[];
  run_at_each_message: boolean;
}

export interface ReferenceState {
  label: string;
  messages: Message[];
}

export interface ModelConfig {
  id: string;
  gguf_file?: string | null;
  dtype: "float16" | "bfloat16" | "float32";
  device: "auto" | "mps" | "cpu";
  trust_remote_code: boolean;
}

export interface GenerationConfig {
  max_new_tokens: number;
  do_sample: boolean;
  temperature: number;
  top_k?: number | null;
  top_p?: number | null;
  seed: number;
}

export interface CaptureConfig {
  hidden_states: boolean;
  attention_weights: boolean;
  qkv: boolean;
  mlp: boolean;
  logits: boolean;
  top_k_probs: number;
  layers: number[] | "all";
  store_dtype: "float16" | "bfloat16" | "float32";
}

export interface OutputConfig {
  dir: string;
  format: "json+npz";
}

export interface ScenarioModel {
  name: string;
  model: ModelConfig;
  prompt: PromptConfig;
  generation: GenerationConfig;
  capture: CaptureConfig;
  output: OutputConfig;
  reference_states: ReferenceState[];
}

export function defaultScenario(): ScenarioModel {
  return {
    name: "",
    model: { id: "", dtype: "float16", device: "auto", trust_remote_code: false, gguf_file: null },
    prompt: { messages: [{ role: "user", content: "" }], run_at_each_message: false },
    generation: { max_new_tokens: 50, do_sample: false, temperature: 1.0, top_k: null, top_p: null, seed: 42 },
    capture: { hidden_states: true, attention_weights: true, qkv: true, mlp: true, logits: true, top_k_probs: 20, layers: "all", store_dtype: "float16" },
    output: { dir: "./runs", format: "json+npz" },
    reference_states: [],
  };
}

export function scenarioToYaml(s: ScenarioModel): string {
  const obj = structuredClone(s) as unknown as Record<string, unknown>;
  // drop null/undefined optional fields
  const model = obj.model as Record<string, unknown>;
  if (!model.gguf_file) delete model.gguf_file;
  const gen = obj.generation as Record<string, unknown>;
  if (gen.top_k == null) delete gen.top_k;
  if (gen.top_p == null) delete gen.top_p;
  return yaml.dump(obj, { sortKeys: false, lineWidth: 120 });
}

export function yamlToScenario(text: string): ScenarioModel {
  const raw = yaml.load(text) as Record<string, unknown>;
  const d = defaultScenario();
  // Merge raw over defaults so every optional field is guaranteed to be present.
  const merged: ScenarioModel = {
    name: (raw.name as string) ?? d.name,
    model: { ...d.model, ...(raw.model as object ?? {}) },
    prompt: {
      ...d.prompt,
      ...(raw.prompt as object ?? {}),
      messages: ((raw.prompt as Record<string, unknown>)?.messages as ScenarioModel["prompt"]["messages"]) ?? d.prompt.messages,
    },
    generation: { ...d.generation, ...(raw.generation as object ?? {}) },
    capture: {
      ...d.capture,
      ...(raw.capture as object ?? {}),
      layers: ((raw.capture as Record<string, unknown>)?.layers ?? "all") as ScenarioModel["capture"]["layers"],
    },
    output: { ...d.output, ...(raw.output as object ?? {}) },
    reference_states: (raw.reference_states as ScenarioModel["reference_states"]) ?? [],
  };
  return merged;
}
