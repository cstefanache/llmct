export type Kind = "snapshot" | "reference" | "step";

export interface NpzRef {
  run_id: string;
  kind: Kind;
  name: string;
}

export interface RunSummary { id: string; path: string; }

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
