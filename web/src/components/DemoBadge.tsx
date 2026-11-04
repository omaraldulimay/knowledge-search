import { useEffect, useState } from "react";
import { IS_DEMO, demoApi } from "../api";

/** Small badge: which model produced the precomputed answers. Renders nothing outside the demo. */
export default function DemoBadge() {
  const [info, setInfo] = useState<{ generated_by: string; generated_at: string } | null>(null);
  useEffect(() => {
    if (IS_DEMO) void demoApi.info().then(setInfo);
  }, []);
  if (!IS_DEMO || !info) return null;
  const model = info.generated_by === "fake" ? "FakeLLM" : info.generated_by;
  return (
    <span className="demobadge" title={`Vorberechnete Antworten von ${model}, Stand ${info.generated_at}`}>
      Demo · {model}
    </span>
  );
}
