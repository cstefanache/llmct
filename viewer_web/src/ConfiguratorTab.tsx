import { useCallback, useEffect, useRef, useState } from "react";
import { getRunScenarioYaml, getScenarioFile, listRuns, listScenarios, RunSummary, ScenarioFileSummary } from "./api";
import { LaunchPanel } from "./configurator/LaunchPanel";
import { ScenarioForm } from "./configurator/ScenarioForm";
import { defaultScenario, ScenarioModel, scenarioToYaml, yamlToScenario } from "./configurator/scenarioModel";
import { YamlPane } from "./configurator/YamlPane";

interface Props {
  onOpenRunTab: (runId: string) => void;
}

export function ConfiguratorTab({ onOpenRunTab }: Props) {
  const [scenario, setScenario] = useState<ScenarioModel>(defaultScenario);
  const [yamlText, setYamlText] = useState(() => scenarioToYaml(defaultScenario()));
  const [yamlError, setYamlError] = useState<string | null>(null);

  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [scenarioFiles, setScenarioFiles] = useState<ScenarioFileSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    listRuns().then(setRuns).catch(console.error);
    listScenarios().then(setScenarioFiles).catch(console.error);
  }, []);

  const applyYaml = useCallback((text: string) => {
    setYamlText(text);
    setLoadError(null);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      try {
        const parsed = yamlToScenario(text);
        setScenario(parsed);
        setYamlError(null);
      } catch (e) {
        setYamlError(String(e));
      }
    }, 280);
  }, []);

  const applyScenario = useCallback((s: ScenarioModel) => {
    setScenario(s);
    setYamlError(null);
    try {
      setYamlText(scenarioToYaml(s));
    } catch (_) { /* keep old yaml */ }
  }, []);

  const handleCloneRun = async (runId: string) => {
    if (!runId) return;
    setLoadError(null);
    try {
      const text = await getRunScenarioYaml(runId);
      const parsed = yamlToScenario(text);
      setScenario(parsed);
      setYamlText(text);
      setYamlError(null);
    } catch (e) {
      setLoadError(`Failed to load run: ${e}`);
    }
  };

  const handleCloneFile = async (path: string) => {
    if (!path) return;
    setLoadError(null);
    try {
      const text = await getScenarioFile(path);
      const parsed = yamlToScenario(text);
      setScenario(parsed);
      setYamlText(text);
      setYamlError(null);
    } catch (e) {
      setLoadError(`Failed to load scenario: ${e}`);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 900 }}>

      {/* Clone bar */}
      <div className="section" style={{ marginBottom: 0 }}>
        <div className="controls" style={{ flexWrap: "wrap" }}>
          <label>
            Clone from run
            <select style={{ marginLeft: 6, maxWidth: 280 }} defaultValue=""
              onChange={(e) => handleCloneRun(e.target.value)}>
              <option value="">— select a run —</option>
              {[...runs].reverse().map((r) => (
                <option key={r.id} value={r.id}>{r.id}</option>
              ))}
            </select>
          </label>
          <label>
            Clone from file
            <select style={{ marginLeft: 6, maxWidth: 280 }} defaultValue=""
              onChange={(e) => handleCloneFile(e.target.value)}>
              <option value="">— select a scenario —</option>
              {scenarioFiles.map((f) => (
                <option key={f.path} value={f.path}>{f.path}</option>
              ))}
            </select>
          </label>
        </div>
        {loadError && <div className="err" style={{ marginTop: 4 }}>{loadError}</div>}
      </div>

      {/* Side-by-side editor */}
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        <div style={{ flex: "0 0 420px", minWidth: 0 }}>
          <ScenarioForm
            scenario={scenario}
            onChange={applyScenario}
            disabled={!!yamlError}
          />
        </div>
        <div style={{ flex: 1, minWidth: 0, position: "sticky", top: 0 }}>
          <YamlPane yaml={yamlText} onChange={applyYaml} error={yamlError} />
        </div>
      </div>

      {/* Launch */}
      <LaunchPanel
        scenario={scenario}
        yamlText={yamlText}
        yamlError={yamlError}
        onOpenRunTab={onOpenRunTab}
      />
    </div>
  );
}
