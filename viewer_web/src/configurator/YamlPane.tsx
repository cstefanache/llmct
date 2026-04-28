interface YamlPaneProps {
  yaml: string;
  onChange: (text: string) => void;
  error: string | null;
}

export function YamlPane({ yaml, onChange, error }: YamlPaneProps) {
  return (
    <div className="yaml-pane">
      <div className="yaml-pane-label">YAML</div>
      <textarea
        className="yaml-textarea"
        value={yaml}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
      />
      {error && <div className="err yaml-error">{error}</div>}
    </div>
  );
}
