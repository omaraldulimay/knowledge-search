import { useCallback, useEffect, useState } from "react";
import { IS_DEMO, api } from "../api";
import { deptLabel } from "../format";
import type { DocItem, Session } from "../types";
import SourcePanel from "./SourcePanel";

const DEPARTMENTS = ["alle", "hr", "it", "vertrieb", "logistik"];
const DEMO_OFF = "In der Demo nicht verfügbar";
const ACCEPT = [".md", ".pdf", ".docx"];

function Upload({ onDone }: { onDone: () => void }) {
  const [department, setDepartment] = useState("alle");
  const [over, setOver] = useState(false);
  const [state, setState] = useState("");

  const send = async (files: FileList | null) => {
    if (IS_DEMO || !files?.length) return;
    const file = files[0];
    if (!ACCEPT.some((s) => file.name.toLowerCase().endsWith(s))) {
      setState("Nur .md, .pdf und .docx");
      return;
    }
    setState("Lädt hoch …");
    try {
      const d = await api.upload(file, department);
      setState(`„${d.title}“ ist indexiert.`);
      onDone();
    } catch (e) {
      setState((e as Error).message);
    }
  };

  return (
    <div className="upload" title={IS_DEMO ? DEMO_OFF : undefined}>
      <label
        className={over ? "drop drop--over" : "drop"}
        aria-disabled={IS_DEMO}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          void send(e.dataTransfer.files);
        }}
      >
        <input
          type="file"
          disabled={IS_DEMO}
          accept={ACCEPT.join(",")}
          className="sr-only"
          onChange={(e) => {
            void send(e.target.files);
            e.target.value = "";
          }}
        />
        <strong>Datei hierher ziehen</strong>
        <span className="muted">oder klicken · md, pdf, docx</span>
      </label>
      <label className="field">
        <span className="muted">Abteilung</span>
        <select disabled={IS_DEMO} value={department} onChange={(e) => setDepartment(e.target.value)}>
          {DEPARTMENTS.map((d) => (
            <option key={d} value={d}>
              {deptLabel(d)}
            </option>
          ))}
        </select>
      </label>
      <p className="muted" role="status">
        {state}
      </p>
    </div>
  );
}

export default function Documents({ session }: { session: Session }) {
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<string | null>(null);

  const load = useCallback(() => {
    api.documents().then(setDocs).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const remove = async (id: string) => {
    try {
      await api.remove(id);
      setConfirm(null);
      if (open === id) setOpen(null);
      load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const groups = DEPARTMENTS.map((d) => [d, docs.filter((x) => x.department === d)] as const).filter(
    ([, list]) => list.length > 0,
  );

  return (
    <div className={open ? "ask ask--panel" : "ask"}>
      <div className="ask-main docs-main">
        <h1>Dokumente</h1>
        <p className="muted">{docs.length} Dokumente für dich sichtbar</p>
        {session.is_admin && <Upload onDone={load} />}
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <div className="docgrid">
        {groups.map(([dept, list]) => (
          <section key={dept} className="docgroup" aria-label={deptLabel(dept)}>
            <h2>
              {deptLabel(dept)} <span className="muted count">{list.length}</span>
            </h2>
            <ul className="doclist">
              {list.map((d) => (
                <li key={d.doc_id}>
                  <button className="doclink" onClick={() => setOpen(d.doc_id)}>
                    {d.title}
                  </button>
                  {session.is_admin &&
                    (confirm === d.doc_id ? (
                      <span className="confirm">
                        <button className="danger" onClick={() => void remove(d.doc_id)}>
                          Wirklich löschen
                        </button>
                        <button className="linkbtn" onClick={() => setConfirm(null)}>
                          Abbrechen
                        </button>
                      </span>
                    ) : (
                      <button
                        className="linkbtn"
                        aria-label={`${d.title} löschen`}
                        disabled={IS_DEMO}
                        title={IS_DEMO ? DEMO_OFF : undefined}
                        onClick={() => setConfirm(d.doc_id)}
                      >
                        Löschen
                      </button>
                    ))}
                </li>
              ))}
            </ul>
          </section>
        ))}
        </div>
      </div>
      {open && <SourcePanel docId={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
