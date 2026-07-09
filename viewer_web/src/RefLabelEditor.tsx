import { useEffect, useState } from "react";
import { NpzRef } from "./api";
import { defaultLabel } from "./refLabels";
import { refKey, setLabelOverride, useLabelOverrides } from "./labelStore";

/**
 * Editable list of the compared snapshots. Each row shows the series color and
 * lets the user type a custom label. Labels are committed on blur / Enter and
 * persist in localStorage, so every chart legend updates to match. The color
 * swatch mirrors the chart palette so a row maps to a line at a glance.
 */
export function RefLabelEditor({ refs, colors }: { refs: NpzRef[]; colors: string[] }) {
  useLabelOverrides(); // re-render this editor when overrides change elsewhere
  return (
    <div className="ref-label-editor">
      {refs.map((r, i) => (
        <RefLabelRow key={refKey(r)} r={r} color={colors[i % colors.length]} />
      ))}
    </div>
  );
}

function RefLabelRow({ r, color }: { r: NpzRef; color: string }) {
  const overrides = useLabelOverrides();
  const override = overrides[refKey(r)] ?? "";
  const auto = defaultLabel(r);
  const [draft, setDraft] = useState(override);

  // Keep the input in sync if the override is changed/reset elsewhere.
  useEffect(() => setDraft(override), [override]);

  const commit = () => setLabelOverride(r, draft);

  return (
    <div className="ref-label-row">
      <span className="ref-label-swatch" style={{ background: color }} />
      <input
        className="ref-label-input"
        value={draft}
        placeholder={auto}
        title={`${auto}\n(${r.run_id})`}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") setDraft(override);
        }}
      />
      {override && (
        <button
          className="ref-label-reset"
          title="Reset to auto label"
          onClick={() => setLabelOverride(r, "")}
        >
          ↺
        </button>
      )}
    </div>
  );
}
