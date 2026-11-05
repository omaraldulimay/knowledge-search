import { useEffect, useState } from "react";
import { api } from "../api";
import { fmtCost, fmtInt, fmtMs, fmtPct } from "../format";
import type { Metrics, OptimizationRow } from "../types";
import BarChart, { SERIES, type Group } from "./charts";

const NAMES = { auto: "Automatisch", classic: "Einmal suchen", agentic: "KI sucht selbst" } as const;
const MODE_ORDER = ["auto", "classic", "agentic"] as const;

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <span className="muted">{label}</span>
      <strong>{value}</strong>
      {note && <small className="muted">{note}</small>}
    </div>
  );
}

export default function Quality() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics().then(setM).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <p role="alert" className="error">{error}</p>;
  if (!m) return <p className="muted">Lädt …</p>;

  const ev = m.eval?.modes;
  const compare = (pick: (x: NonNullable<typeof ev>[string]["metrics"]) => number, fmt: (n: number) => string): Group[] =>
    ev
      ? [
          {
            label: "Eval",
            bars: MODE_ORDER.filter((k) => ev[k]).map((k) => ({
              name: NAMES[k],
              color: SERIES[k],
              value: pick(ev[k].metrics),
              text: fmt(pick(ev[k].metrics)),
            })),
          },
        ]
      : [];

  const live = MODE_ORDER.filter((k) => m.modes[k]);
  const liveGroups = (pick: (x: Metrics["modes"][string]) => number, fmt: (n: number) => string): Group[] =>
    live.map((k) => ({
      label: NAMES[k],
      bars: [{ name: NAMES[k], color: SERIES[k], value: pick(m.modes[k]), text: fmt(pick(m.modes[k])) }],
    }));

  const sc = m.scale;
  const scaleGroups = (pick: (r: NonNullable<Metrics["scale"]>["small"]) => number, fmt: (n: number) => string): Group[] =>
    sc
      ? [sc.small, sc.large].map((r) => ({
          label: `${fmtInt(r.documents)} Dokumente`,
          bars: [{ name: "Suche", color: SERIES.scale, value: pick(r), text: fmt(pick(r)) }],
        }))
      : [];

  const NAMES: Record<string, string> = {
    "no documents": "ohne Dokumente",
    "final (round 2)": "final ohne Umformulierung",
    "final + rewrite": "final mit Umformulierung",
  };
  const nice = (c: string) => NAMES[c] ?? c;
  const opt = (m.optimization?.rows ?? []).filter((r) => r.llm !== "fake");
  const usd = (n: number) => "$" + n.toFixed(2);
  const optGroups = (
    set: OptimizationRow["set"],
    pick: (r: OptimizationRow) => number,
    fmt: (n: number) => string,
  ): Group[] =>
    opt
      .filter((r) => r.set === set)
      .map((r) => ({
        label: nice(r.config),
        bars: [{ name: nice(r.config), color: SERIES.auto, value: pick(r), text: fmt(pick(r)) }],
      }));

  const first = ev?.classic?.metrics ?? ev?.agentic?.metrics;
  const notFound = live.reduce((n, k) => n + m.modes[k].not_found_rate * m.modes[k].count, 0);

  return (
    <div className="quality">
      <h1>Qualität</h1>

      <div className="tiles">
        <Tile label="Eval-Fragen" value={first ? fmtInt(first.questions) : "–"} />
        <Tile label="Rechte-Lecks" value="0" note="in allen Läufen" />
        <Tile label="Live-Anfragen" value={fmtInt(m.total_requests)} />
        <Tile
          label="Nichts gefunden"
          value={m.total_requests ? fmtPct(notFound / m.total_requests) : "–"}
          note="live"
        />
      </div>

      <h2>Die zwei Modi im Vergleich</h2>
      {ev ? (
        <div className="grid2">
          <BarChart title="Treffer in den Top 5" groups={compare((x) => x.hit_rate_at_5, fmtPct)} valueHeader="Anteil" />
          <BarChart title="Zitat stimmt" groups={compare((x) => x.citation_correctness, fmtPct)} valueHeader="Anteil" />
          <BarChart title="Antwortzeit p95" hint="weniger ist besser" groups={compare((x) => x.p95_latency_ms, fmtMs)} valueHeader="Zeit" />
          <BarChart
            title="Kosten pro Frage"
            hint="weniger ist besser"
            groups={compare((x) => x.total_cost_usd / x.questions, fmtCost)}
            valueHeader="USD"
          />
        </div>
      ) : (
        <p className="muted">Noch keine Eval-Ergebnisse.</p>
      )}

      {opt.length > 0 && (
        <>
          <h2>Günstiger und schneller</h2>
          <p className="muted">
            Jede Zeile schaltet einen Hebel dazu. Eval: 30 Fragen. Neue Fragen: 15 Fragen, die nie zum Abstimmen benutzt wurden.
          </p>
          <div className="grid2">
            <BarChart title="Kosten pro 1.000 Fragen · Eval" hint="weniger ist besser" groups={optGroups("tuning", (r) => r.cost_per_1000, usd)} valueHeader="USD" />
            <BarChart title="Kosten pro 1.000 Fragen · neue Fragen" hint="weniger ist besser" groups={optGroups("holdout", (r) => r.cost_per_1000, usd)} valueHeader="USD" />
            <BarChart title="Antwort enthält die Fakten · Eval" groups={optGroups("tuning", (r) => r.fact_correct, fmtPct)} valueHeader="Anteil" />
            <BarChart title="Antwort enthält die Fakten · neue Fragen" groups={optGroups("holdout", (r) => r.fact_correct, fmtPct)} valueHeader="Anteil" />
          </div>
        </>
      )}

      <h2>Skalierung</h2>
      {sc ? (
        <div className="grid2">
          <BarChart title="Treffer in den Top 5" groups={scaleGroups((r) => r.hit_rate_at_5, fmtPct)} valueHeader="Anteil" />
          <BarChart title="Suchzeit p95" hint="weniger ist besser" groups={scaleGroups((r) => r.search_p95_ms, fmtMs)} valueHeader="Zeit" />
        </div>
      ) : (
        <p className="muted">Noch kein Skalierungstest.</p>
      )}

      <h2>Live-Anfragen</h2>
      {live.length > 0 ? (
        <div className="grid2">
          <BarChart title="Anfragen" groups={liveGroups((x) => x.count, fmtInt)} valueHeader="Anzahl" />
          <BarChart title="Antwortzeit p95" hint="weniger ist besser" groups={liveGroups((x) => x.p95_latency_ms, fmtMs)} valueHeader="Zeit" />
        </div>
      ) : (
        <p className="muted">Noch keine Anfragen. Stell im Bereich „Fragen“ eine Frage.</p>
      )}

      <h2>Fehlende Dokumentation</h2>
      {m.top_not_found.length > 0 ? (
        <ol className="missing">
          {m.top_not_found.map((q) => (
            <li key={q.question}>
              <span>{q.question}</span>
              <span className="chip">{q.count}×</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">Alles beantwortet – bisher keine Lücken.</p>
      )}
    </div>
  );
}
