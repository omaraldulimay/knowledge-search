import { useId, useState } from "react";

export type Bar = { name: string; color: string; value: number; text: string };
export type Group = { label: string; bars: Bar[] };

/** Series colors are CSS variables (validated with the dataviz palette script, light and dark). */
export const SERIES = {
  classic: "var(--series-classic)",
  agentic: "var(--series-agentic)",
  scale: "var(--series-scale)",
};

type Props = { title: string; hint?: string; groups: Group[]; valueHeader?: string };

/** Horizontal bar chart in plain HTML/CSS with a hover/focus tooltip, a legend and a table view. */
export default function BarChart({ title, hint, groups, valueHeader = "Wert" }: Props) {
  const [view, setView] = useState<"chart" | "table">("chart");
  const id = useId();
  const max = Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value)), 1e-9);
  const series = [...new Map(groups.flatMap((g) => g.bars).map((b) => [b.name, b.color])).entries()];

  return (
    <figure className="chart">
      <figcaption>
        <div>
          <h3 id={id}>{title}</h3>
          {hint && <p className="muted">{hint}</p>}
        </div>
        <div className="seg" role="group" aria-label={`Ansicht für ${title}`}>
          {(["chart", "table"] as const).map((v) => (
            <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>
              {v === "chart" ? "Diagramm" : "Tabelle"}
            </button>
          ))}
        </div>
      </figcaption>
      {view === "chart" ? (
        <>
          {series.length > 1 && (
            <ul className="legend">
              {series.map(([name, color]) => (
                <li key={name}>
                  <i style={{ background: color }} aria-hidden="true" />
                  {name}
                </li>
              ))}
            </ul>
          )}
          <div className="bars" role="list" aria-labelledby={id}>
            {groups.map((g) => (
              <div className="bgroup" role="listitem" key={g.label}>
                {groups.length > 1 && <span className="blabel">{g.label}</span>}
                {g.bars.map((b) => (
                  <div className="brow" key={b.name} tabIndex={0} aria-label={`${g.label}, ${b.name}: ${b.text}`}>
                    <span className="btrack">
                      <span
                        className="bfill"
                        style={{ width: `${Math.max((b.value / max) * 100, 1.5)}%`, background: b.color }}
                      />
                    </span>
                    <span className="bval">{b.text}</span>
                    <span className="tip" role="tooltip">
                      {g.label !== b.name ? `${g.label} · ` : ""}
                      {b.name}: <strong>{b.text}</strong>
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </>
      ) : (
        <table className="datatable">
          <thead>
            <tr>
              <th scope="col">{groups.length > 1 ? "Gruppe" : "Reihe"}</th>
              <th scope="col">Serie</th>
              <th scope="col">{valueHeader}</th>
            </tr>
          </thead>
          <tbody>
            {groups.flatMap((g) =>
              g.bars.map((b) => (
                <tr key={g.label + b.name}>
                  <th scope="row">{g.label}</th>
                  <td>{b.name}</td>
                  <td>{b.text}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      )}
    </figure>
  );
}
