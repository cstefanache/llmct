import { useSyncExternalStore } from "react";
import { NpzRef } from "./api";

// Persistent per-snapshot label overrides. Keyed by run_id|kind|name so a
// custom label sticks to one exact snapshot (a specific prompt's npz), which is
// what disambiguates snapshots whose auto-generated labels collide.

const STORAGE_KEY = "activation_lab.labelOverrides.v1";

function load(): Record<string, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

let overrides: Record<string, string> = load();
const listeners = new Set<() => void>();

function emit() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
  } catch {
    /* ignore quota / private-mode failures — labels just won't persist */
  }
  listeners.forEach((l) => l());
}

export function refKey(r: NpzRef): string {
  return `${r.run_id}|${r.kind}|${r.name}`;
}

export function getLabelOverride(r: NpzRef): string | undefined {
  return overrides[refKey(r)];
}

/** Set (non-empty) or clear (empty/whitespace) the custom label for a ref. */
export function setLabelOverride(r: NpzRef, label: string): void {
  const key = refKey(r);
  const trimmed = label.trim();
  if (trimmed) {
    if (overrides[key] === trimmed) return;
    overrides = { ...overrides, [key]: trimmed };
  } else {
    if (!(key in overrides)) return;
    const next = { ...overrides };
    delete next[key];
    overrides = next;
  }
  emit();
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot(): Record<string, string> {
  return overrides;
}

/** Subscribe a component so it re-renders whenever any override changes. */
export function useLabelOverrides(): Record<string, string> {
  return useSyncExternalStore(subscribe, getSnapshot);
}
