import { NpzRef } from "./api";

/**
 * Label for a single ref using 3-part hierarchy:
 *   {run_id[0:15]} / {kind[0:10]} / {name[0:10]}
 */
export function baseLabel(r: NpzRef): string {
  return `${r.run_id.slice(0, 20)} / ${r.kind.slice(0, 2)} / ${r.name.slice(0, 2)}`;
}

/** Build labels for a collection of refs. */
export function makeRefLabels(refs: NpzRef[]): string[] {
  return refs.map(baseLabel);
}

/** Label for a pair tab. */
export function pairTabLabel(a: NpzRef, b: NpzRef): string {
  return `pair: ${baseLabel(a)} ↔ ${baseLabel(b)}`;
}
