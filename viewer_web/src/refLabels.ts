import { NpzRef } from "./api";
import { getLabelOverride } from "./labelStore";

/**
 * Auto-generated label from the ref hierarchy:
 *   {run_id without timestamp suffix} / {kind} / {name}
 * Note: several snapshots (e.g. the same npz across prompt_0/prompt_3/…) can
 * collapse to the same auto label — that's why manual overrides exist.
 */
export function defaultLabel(r: NpzRef): string {
  return `${r.run_id.slice(0, -21)} / ${r.kind} / ${r.name}`;
}

/** Display label: the user's custom override if set, otherwise the auto label. */
export function baseLabel(r: NpzRef): string {
  return getLabelOverride(r) ?? defaultLabel(r);
}

/** Build labels for a collection of refs. */
export function makeRefLabels(refs: NpzRef[]): string[] {
  return refs.map(baseLabel);
}

/** Label for a pair tab. */
export function pairTabLabel(a: NpzRef, b: NpzRef): string {
  return `pair: ${a.run_id} / ${baseLabel(a)} ↔ ${b.run_id} / ${baseLabel(b)}`;
}
