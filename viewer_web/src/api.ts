export type Kind = "snapshot" | "reference" | "step";

export interface NpzRef {
  run_id: string;
  kind: Kind;
  name: string;
}

export interface RunSummary {
  id: string;
  path: string;
  /** Parent scenario directory name when this run is nested under a multi-prompt scenario; null for legacy flat runs. */
  group?: string | null;
  /** Leaf folder name (e.g. "prompt_0"). For legacy flat runs, equals id. */
  label?: string;
}

export interface TreeNode {
  kind: Kind;
  name: string;
  label: string;
  role?: string;
  index?: number;
  step?: number;
  /** true = attention captured, false = not captured, null = not yet probed (see /meta) */
  has_attention: boolean | null;
}

export interface TreeGroup {
  type: "conversation_snapshots" | "references" | "tensors";
  children: TreeNode[];
}

export interface RunTree {
  run_id: string;
  children: TreeGroup[];
}

export interface NpzMeta {
  file: string;
  keys: string[];
  shapes: Record<string, number[]>;
  dtypes: Record<string, string>;
  layers: number[];
  has_attention: boolean;
  has_qk: boolean;
  has_qkv_last: boolean;
}

export interface EntropyResponse {
  layers: number[];
  entropy: number[];
}

export interface LogitStatsResponse {
  steps: number[];
  entropy: number[];
  effective_vocab: number[];
  top1_prob: number[];
}

export interface ConvergenceResponse {
  layers: number[];
  adj_layers: number[];
  adj_cosine_last: number[];
  adj_cosine_mean: number[];
  delta_norm_last: number[];
  delta_norm_mean: number[];
  effective_depth_last: number | null;
  effective_depth_mean: number | null;
}

export interface PerSourceMetrics {
  layers: number[];
  mean_cos: number[];
  min_cos: number[];
  mean_mae: number[];
  std_diff: number[];
  mean_overlap: number[];
}

export type PairMetrics = Record<string, PerSourceMetrics>;

export interface PairEntry {
  i: number;
  j: number;
  metrics: PairMetrics;
}

export interface MetricsResponse {
  pairs: PairEntry[];
}

// ---- advanced multi-ref analysis

export interface SvdRefResult {
  spectra: (number[] | null)[];
  effective_rank: (number | null)[];
  nuclear_norm: (number | null)[];
  spectral_norm: (number | null)[];
}

export interface SvdSourceResult {
  layers: number[];
  refs: SvdRefResult[];
}

export interface CkaSourceResult {
  layers: number[];
  matrices: (number | null)[][][];  // [layer][i][j]
}

export interface PcaSourceResult {
  coords: [number, number][];
  ref_indices: number[];
  layer_indices: number[];
  layers: number[];
  explained_variance: [number, number];
}

export interface AdvancedMetricsResponse {
  svd: Record<string, SvdSourceResult>;
  cka: Record<string, CkaSourceResult>;
  pca: Record<string, PcaSourceResult>;
}

export interface GroupSourceResult {
  layers: number[];
  cos_to_centroid: (number | null)[][];   // [layer][snapshot]
  divergence: (number | null)[];          // per layer
  pairwise_cos: (number | null)[][];      // (N, N) layer-averaged
  topk_jaccard: (number | null)[][];      // (N, N) layer-averaged
  linkage: [number, number, number, number][];  // SciPy-style linkage rows
}

export type GroupAnalysisResponse = Record<string, GroupSourceResult>;

export async function compareGroup(
  refs: NpzRef[], sources: string[],
): Promise<GroupAnalysisResponse> {
  const r = await fetch("/api/compare/group", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refs, sources }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function compareAdvanced(
  refs: NpzRef[], sources: string[],
): Promise<AdvancedMetricsResponse> {
  const r = await fetch("/api/compare/advanced", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refs, sources }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getRunLogitStats(runId: string): Promise<LogitStatsResponse> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/logit_stats`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getNpzConvergence(
  ref: NpzRef,
  source: string = "hidden_out",
): Promise<ConvergenceResponse> {
  const r = await fetch(
    `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/convergence?source=${encodeURIComponent(source)}`,
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listRuns(): Promise<RunSummary[]> {
  const r = await fetch("/api/runs");
  return r.json();
}

export async function getRun(runId: string): Promise<any> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!r.ok) throw new Error(`run ${runId} not found`);
  return r.json();
}

export async function getTree(runId: string): Promise<RunTree> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/tree`);
  if (!r.ok) throw new Error(`tree ${runId}`);
  return r.json();
}

export async function getNpzMeta(ref: NpzRef): Promise<NpzMeta> {
  const r = await fetch(
    `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/meta`,
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getEntropy(ref: NpzRef): Promise<EntropyResponse> {
  const r = await fetch(
    `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/entropy`,
  );
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function attentionPngUrl(ref: NpzRef, layer: string): string {
  return `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/attention.png?layer=${layer}`;
}

export function qkPngUrl(ref: NpzRef, layer: string): string {
  return `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/qk.png?layer=${layer}`;
}

export function qkvPngUrl(ref: NpzRef): string {
  return `/api/runs/${encodeURIComponent(ref.run_id)}/npz/${ref.kind}/${encodeURIComponent(ref.name)}/qkv.png`;
}

export async function compareMetrics(refs: NpzRef[], sources: string[]): Promise<MetricsResponse> {
  const r = await fetch("/api/compare/metrics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refs, sources }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function comparePairHeatmap(
  a: NpzRef, b: NpzRef,
  source: "attention" | "qk" | "qkv",
  layer: string,
  variant: "a" | "b" | "abs_diff" | "sq_err" | "hadamard" | "ratio",
): Promise<string> {
  const r = await fetch("/api/compare/heatmap.png", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ a, b, source, layer, variant }),
  });
  if (!r.ok) throw new Error(await r.text());
  const blob = await r.blob();
  return URL.createObjectURL(blob);
}

// ---- configurator

export interface ScenarioFileSummary { path: string; name: string; mtime: number; }

export async function listScenarios(): Promise<ScenarioFileSummary[]> {
  const r = await fetch("/api/scenarios");
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getScenarioFile(path: string): Promise<string> {
  const r = await fetch(`/api/scenarios/file?path=${encodeURIComponent(path)}`);
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  return data.yaml as string;
}

export async function getRunScenarioYaml(runId: string): Promise<string> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/scenario.yaml`);
  if (!r.ok) throw new Error(await r.text());
  return r.text();
}

export interface ValidationResult { ok: boolean; errors: { loc: string[]; msg: string }[]; }

export async function validateScenario(yamlText: string): Promise<ValidationResult> {
  const r = await fetch("/api/scenarios/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml: yamlText }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function saveScenario(path: string, yamlText: string, overwrite: boolean): Promise<void> {
  const r = await fetch("/api/scenarios/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, yaml: yamlText, overwrite }),
  });
  if (!r.ok) throw new Error(await r.text());
}

export interface LocalModel { model_id: string; }

export async function listLocalModels(): Promise<LocalModel[]> {
  const r = await fetch("/api/models/local");
  if (!r.ok) return [];
  return r.json();
}

export async function launchScenario(path: string): Promise<{ job_id: string }> {
  const r = await fetch("/api/runs/launch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export interface JobStatus {
  status: "queued" | "running" | "done" | "failed";
  returncode: number | null;
  started_at: string;
  finished_at: string;
  log_tail: string[];
  run_id: string | null;
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---- report download

export type ReportKind = "run" | "npz" | "pair" | "multi";

export interface ReportRequest {
  kind: ReportKind;
  run_id?: string;
  ref?: NpzRef;
  a?: NpzRef;
  b?: NpzRef;
  refs?: NpzRef[];
  sources?: string[];
}

export function openReport(req: ReportRequest): void {
  const params = btoa(JSON.stringify(req));
  window.open(`/api/report/view?params=${encodeURIComponent(params)}`, "_blank");
}
