import { useEffect, useRef, useState } from "react";
import { IS_DEMO, api, demoApi } from "../api";
import { examplesFor, fmtCost, fmtInt, fmtMs, parseAnswer } from "../format";
import type { AskResult, Mode, Session, Step } from "../types";
import Overview, { type HistoryItem } from "./Overview";
import SourcePanel from "./SourcePanel";

const MODES: [Mode, string][] = [
  ["auto", "Automatisch"],
  ["classic", "Einmal suchen"],
  ["agentic", "KI sucht selbst"],
];

const KIND_TAG: Record<string, string> = {
  not_found: "nicht dokumentiert",
  permission: "Rechte-Test",
};

const ROUTE_LABEL: Record<string, string> = {
  classic: "einmal gesucht",
  agentic: "KI hat selbst gesucht",
  none: "ohne KI-Aufruf",
};

function stepLabel(s: Step): string {
  if (s.tool === "router") return "Router entscheidet";
  if (s.tool === "rewrite") return `Suche neu formuliert: „${String(s.args.query ?? "")}“`;
  if (s.tool === "cache") return "Antwort aus dem Cache";
  if (s.tool === "search") return `Suche „${String(s.args.query ?? "")}“`;
  if (s.tool === "read_document") return `Dokument gelesen: ${String(s.args.doc_id ?? "")}`;
  if (s.tool === "list_documents") return "Dokumente aufgelistet";
  return s.tool;
}

const HISTORY_MAX = 8;
const historyKey = (login: string) => `ks-history:${login}`;

function loadHistory(login: string): HistoryItem[] {
  try {
    const raw = sessionStorage.getItem(historyKey(login));
    const items = raw ? (JSON.parse(raw) as HistoryItem[]) : [];
    return Array.isArray(items) ? items.slice(0, HISTORY_MAX) : [];
  } catch {
    return []; // private window, blocked storage, broken JSON: start with an empty list
  }
}

function saveHistory(login: string, items: HistoryItem[]) {
  try {
    sessionStorage.setItem(historyKey(login), JSON.stringify(items));
  } catch {
    /* the history just stays in memory */
  }
}

export function Timeline({ steps }: { steps: Step[] }) {
  return (
    <ol className="timeline" aria-label="Schritte der KI">
      {steps.map((s, i) => (
        <li key={i}>
          <span>{stepLabel(s)}</span>
          <small className="muted">{s.summary}</small>
        </li>
      ))}
      <li>
        <span>Antwort</span>
      </li>
    </ol>
  );
}

/** Admin only: how the answer was found (steps, route) and what it cost. */
function TechDetails({ result }: { result: AskResult }) {
  const parts = [
    result.cached ? "aus dem Cache" : result.route ? (ROUTE_LABEL[result.route] ?? result.route) : "",
    typeof result.latency_ms === "number" ? fmtMs(result.latency_ms) : "",
    typeof result.input_tokens === "number" ? `${fmtInt(result.input_tokens + (result.output_tokens ?? 0))} Tokens` : "",
    typeof result.cost === "number" ? fmtCost(result.cost) : "",
    typeof result.ttft_ms === "number" ? `erstes Wort nach ${fmtMs(result.ttft_ms)}` : "",
  ].filter(Boolean);
  return (
    <details className="tech">
      <summary>Technische Details</summary>
      {result.steps && result.steps.length > 0 && <Timeline steps={result.steps} />}
      <p className="meta muted">{parts.join(" · ")}</p>
    </details>
  );
}

type Open = { docId: string; section: string };

export default function Ask({ session, devParams }: { session: Session; devParams: URLSearchParams | null }) {
  const [question, setQuestion] = useState(devParams?.get("q") ?? "");
  const isAdmin = session.is_admin;
  // Employees always get the automatic mode; only the admin may pick one.
  const [pickedMode, setMode] = useState<Mode>((isAdmin && (devParams?.get("mode") as Mode)) || "auto");
  const mode: Mode = isAdmin ? pickedMode : "auto";
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory(session.login));
  const [streamed, setStreamed] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [asked, setAsked] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Open | null>(null);
  const lastMarker = useRef<HTMLElement | null>(null);
  const [prepared, setPrepared] = useState<{ question: string; kind: string }[]>([]);

  useEffect(() => {
    if (IS_DEMO) void demoApi.questions().then(setPrepared);
  }, []);

  const submit = async (q = question, m = mode) => {
    if (!q.trim() || loading) return;
    setLoading(true);
    setError("");
    setOpen(null);
    setStreamed("");
    setQuestion("");
    try {
      const r = await api.askStream(q.trim(), m, (text) => setStreamed((s) => s + text));
      setResult(r);
      setAsked(q.trim());
      setHistory((h) => {
        const next = [{ question: q.trim(), result: r }, ...h.filter((x) => x.question !== q.trim())];
        saveHistory(session.login, next.slice(0, HISTORY_MAX));
        return next.slice(0, HISTORY_MAX);
      });
      if (devParams?.get("cite") && r.citations[0]) {
        setOpen({ docId: r.citations[0].doc_id, section: r.citations[0].section });
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
      setStreamed("");
    }
  };

  useEffect(() => {
    if (devParams?.get("q")) void submit(devParams.get("q")!, mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const showAgain = (item: HistoryItem) => {
    setOpen(null);
    setError("");
    setResult(item.result);
    setAsked(item.question);
    setQuestion(item.question);
  };

  const lines = result ? parseAnswer(result.answer, result.citations) : [];

  const closePanel = () => {
    setOpen(null);
    lastMarker.current?.focus();
  };

  return (
    <div className={open ? "ask ask--panel" : "ask"}>
      <div className="ask-main">
        <h1>Was möchtest du wissen?</h1>
        {IS_DEMO ? (
          <p className="demo-hint muted">Demo: vorbereitete Fragen. Live-Version läuft lokal mit Docker.</p>
        ) : (
          <form
            className="askform"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <label className="sr-only" htmlFor="q">
              Frage
            </label>
            <input
              id="q"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="z. B. Was bedeutet K3 in ELKO?"
              maxLength={1000}
              autoComplete="off"
            />
            <button className="primary" type="submit" disabled={loading || !question.trim()}>
              {loading ? "Sucht …" : "Absenden"}
            </button>
          </form>
        )}
        {isAdmin && (
        <div className="modes" role="radiogroup" aria-label="Suchmodus">
          {MODES.map(([id, label]) => (
            <label key={id} className={mode === id ? "mode mode--on" : "mode"}>
              <input type="radio" name="mode" checked={mode === id} onChange={() => {
                  setMode(id);
                  if (IS_DEMO && asked) void submit(asked, id); // compare the two modes on the same question
                }} />
              {label}
            </label>
          ))}
        </div>
        )}

        {IS_DEMO && (
          <div className="examples examples--narrow" role="group" aria-label="Vorbereitete Fragen">
            {prepared.map((p) => (
              <button
                key={p.question}
                className={asked === p.question ? "chip chip--btn chip--on" : "chip chip--btn"}
                onClick={() => {
                  setQuestion(p.question);
                  void submit(p.question);
                }}
              >
                {p.question}
                {KIND_TAG[p.kind] && <small className="muted"> · {KIND_TAG[p.kind]}</small>}
              </button>
            ))}
          </div>
        )}

        {!IS_DEMO && !result && !loading && (
          <div className="examples examples--narrow">
            {examplesFor(session.departments).map((q) => (
              <button
                key={q}
                className="chip chip--btn"
                onClick={() => {
                  setQuestion(q);
                  void submit(q);
                }}
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}

        {loading && streamed.trim() && (
          <article className="answer answer--live" aria-live="polite">
            <p className="asked muted">{question}</p>
            <p>{streamed.replace(/\[\d+(?:\s*,\s*\d+)*\]/g, "")}</p>
          </article>
        )}

        {result && !loading && (
          <article className={result.found ? "answer" : "answer answer--empty"} aria-live="polite">
            <p className="asked muted">{asked}</p>
            {result.found ? (
              <div className="answer-body">
                {lines.map((parts, i) => (
                  <p key={i}>
                    {parts.map((p, j) =>
                      p.kind === "text" ? (
                        <span key={j}>{p.text}</span>
                      ) : (
                        <button
                          key={j}
                          className="cite"
                          aria-label={`Quelle ${p.n}: ${result.citations[p.n - 1].title}`}
                          onClick={(e) => {
                            lastMarker.current = e.currentTarget;
                            const c = result.citations[p.n - 1];
                            setOpen({ docId: c.doc_id, section: c.section });
                          }}
                        >
                          {p.n}
                        </button>
                      ),
                    )}
                  </p>
                ))}
              </div>
            ) : (
              <div className="empty-note">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                  <circle cx="11" cy="11" r="6.5" />
                  <path d="M16 16l4.5 4.5M9 11h4" strokeLinecap="round" />
                </svg>
                <div>
                  <strong>Nichts gefunden</strong>
                  <p>{result.answer}</p>
                  <p className="muted">Die Frage wird für die Dokumentation vermerkt.</p>
                </div>
              </div>
            )}
            {result.found && result.citations.length > 0 && (
              <ul className="sources" aria-label="Quellen">
                {result.citations.map((c, i) => (
                  <li key={c.chunk_id}>
                    <button
                      className="source"
                      onClick={(e) => {
                        lastMarker.current = e.currentTarget;
                        setOpen({ docId: c.doc_id, section: c.section });
                      }}
                    >
                      <span className="cite cite--static">{i + 1}</span>
                      {c.title} <span className="muted">· {c.section}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {isAdmin && <TechDetails result={result} />}
          </article>
        )}
      </div>
      {open ? (
        <SourcePanel docId={open.docId} section={open.section} onClose={closePanel} />
      ) : (
        <Overview
          history={history}
          suggestions={IS_DEMO ? prepared.map((p) => p.question) : examplesFor(session.departments)}
          current={asked}
          onShow={showAgain}
          onAsk={(q) => {
            setQuestion(q);
            void submit(q);
          }}
        />
      )}
    </div>
  );
}
