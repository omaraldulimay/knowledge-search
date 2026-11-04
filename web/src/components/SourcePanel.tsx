import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { deptLabel } from "../format";
import type { DocDetail } from "../types";

/** Chunk text is markdown-ish: render pipe tables as tables, everything else as a paragraph. */
function ChunkText({ text }: { text: string }) {
  if (!text.includes("|---")) return <p>{text}</p>;
  const cells = (row: string) => row.split("|").slice(1, -1).map((c) => c.trim());
  const lines = text.split("\n").filter((l) => l.trim());
  const rows = lines.filter((l) => l.trim().startsWith("|") && !/^\|[\s|:-]+\|$/.test(l.trim()));
  const prose = lines.filter((l) => !l.trim().startsWith("|"));
  const [head, ...body] = rows.map(cells);
  return (
    <>
      {prose.map((l, i) => (
        <p key={i}>{l}</p>
      ))}
      <table className="datatable">
        <thead>
          <tr>{head?.map((c, i) => <th key={i} scope="col">{c}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

type Props = { docId: string; section?: string; onClose: () => void };

/** Side panel with the full source document; the cited section is highlighted and scrolled into view. */
export default function SourcePanel({ docId, section, onClose }: Props) {
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [error, setError] = useState("");
  const closeRef = useRef<HTMLButtonElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const hitRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setDoc(null);
    setError("");
    api
      .document(docId)
      .then(setDoc)
      .catch((e: Error) => setError(e.message));
  }, [docId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => closeRef.current?.focus(), [docId]);

  useEffect(() => {
    // scroll only the panel itself, never the page behind it
    const hit = hitRef.current;
    const panel = panelRef.current;
    if (hit && panel?.scrollTo) panel.scrollTo({ top: hit.offsetTop - (headRef.current?.offsetHeight ?? 0) - 16 });
  }, [doc, section]);

  return (
    <aside className="panel" aria-label="Quelle" ref={panelRef}>
      <div className="panel-head" ref={headRef}>
        <div>
          <h2>{doc?.title ?? "Quelle"}</h2>
          {doc && <span className="chip">{deptLabel(doc.department)}</span>}
        </div>
        <button ref={closeRef} className="closebtn" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M2 2l10 10M12 2L2 12" />
          </svg>
          Schließen
        </button>
      </div>
      <div className="panel-body">
      {error && <p className="error">{error}</p>}
      {!doc && !error && <p className="muted">Lädt …</p>}
      {doc?.chunks.map((c) => {
        const hit = section !== undefined && c.heading === section;
        return (
          <section
            key={c.chunk_id}
            className={hit ? "chunk chunk--hit" : "chunk"}
            ref={hit ? hitRef : undefined}
            aria-current={hit ? "true" : undefined}
          >
            <h3>{c.heading}</h3>
            <ChunkText text={c.text} />
          </section>
        );
      })}
      </div>
    </aside>
  );
}
