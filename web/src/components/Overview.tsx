import type { AskResult } from "../types";

export type HistoryItem = { question: string; result: AskResult };

type Props = {
  history: HistoryItem[];
  suggestions: string[];
  current: string;
  onShow: (item: HistoryItem) => void;
  onAsk: (question: string) => void;
};

/** Right column when no source is open (wide screens only): last questions and common questions. */
export default function Overview({ history, suggestions, current, onShow, onAsk }: Props) {
  return (
    <aside className="side" aria-label="Verlauf und Vorschläge">
      <section>
        <h2>Deine letzten Fragen</h2>
        {history.length === 0 ? (
          <p className="muted">Hier erscheinen die Fragen dieser Sitzung.</p>
        ) : (
          <ul className="sidelist">
            {history.map((h) => (
              <li key={h.question}>
                <button className={h.question === current ? "sideitem sideitem--on" : "sideitem"} onClick={() => onShow(h)}>
                  {h.question}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section>
        <h2>Häufig gefragt</h2>
        <ul className="sidelist">
          {suggestions.map((q) => (
            <li key={q}>
              <button className="sideitem" onClick={() => onAsk(q)}>
                {q}
              </button>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
