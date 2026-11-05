import { useEffect, useState } from "react";
import { IS_DEMO, demoApi } from "../api";

/** "claude-haiku-5-5" -> "Claude Haiku 5.5"; "fake" -> "FakeLLM". */
export function modelLabel(id: string): string {
  if (id === "fake") return "FakeLLM";
  const m = /^claude-([a-z]+)-(\d+)-(\d+)$/.exec(id);
  if (!m) return id;
  return `Claude ${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}.${m[3]}`;
}

/** Small badge: which model produced the precomputed answers. Renders nothing outside the demo. */
export default function DemoBadge() {
  const [info, setInfo] = useState<{ generated_by: string; generated_at: string } | null>(null);
  useEffect(() => {
    if (IS_DEMO) void demoApi.info().then(setInfo);
  }, []);
  if (!IS_DEMO || !info) return null;
  const model = modelLabel(info.generated_by);
  return (
    <span className="demobadge" title={`Vorberechnete Antworten von ${model}, Stand ${info.generated_at}`}>
      Demo · {model} · vorberechnet
    </span>
  );
}
