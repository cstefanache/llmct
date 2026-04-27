import { NpzRef } from "./api";

/**
 * Label for a single ref using 3-part hierarchy:
 *   {run_id[0:15]} / {kind[0:10]} / {name[0:10]}
 */
export function baseLabel(r: NpzRef): string {
  return `${r.run_id.slice(0, -21)} / ${r.kind} / ${r.name}`;
}

/** Build labels for a collection of refs. */
export function makeRefLabels(refs: NpzRef[]): string[] {
  return refs.map(baseLabel);
}

/** Label for a pair tab. */
export function pairTabLabel(a: NpzRef, b: NpzRef): string {
  return `pair: ${a.run_id} / ${baseLabel(a)} ↔ ${b.run_id} / ${baseLabel(b)}`;
}
