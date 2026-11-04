import { useCallback, useEffect, useState } from "react";
import { api, setToken } from "./api";
import Ask from "./components/Ask";
import Documents from "./components/Documents";
import Login from "./components/Login";
import Quality from "./components/Quality";
import type { Session } from "./types";

export type Tab = "ask" | "docs" | "quality";

// Dev only: ?as=katja logs in automatically, ?tab=docs|quality picks the screen (used for screenshots).
const devParams = import.meta.env.DEV ? new URLSearchParams(window.location.search) : null;

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [tab, setTab] = useState<Tab>((devParams?.get("tab") as Tab) || "ask");

  const logIn = useCallback((s: Session) => {
    setToken(s.token);
    setSession(s);
    setTab((t) => (t === "quality" && !s.is_admin ? "ask" : t));
  }, []);

  const logOut = () => {
    setToken(null);
    setSession(null);
    setTab("ask");
  };

  useEffect(() => {
    const as = devParams?.get("as");
    if (as) api.login(as).then(logIn).catch(() => undefined);
  }, [logIn]);

  if (!session) return <Login onLogin={logIn} />;

  const tabs: [Tab, string][] = [
    ["ask", "Fragen"],
    ["docs", "Dokumente"],
    ...(session.is_admin ? [["quality", "Qualität"] as [Tab, string]] : []),
  ];

  return (
    <div className="app">
      <header className="topbar">
        <span className="wordmark">Elbkontor</span>
        <nav aria-label="Hauptnavigation">
          {tabs.map(([id, label]) => (
            <button
              key={id}
              className="navlink"
              aria-current={tab === id ? "page" : undefined}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="who">
          <span className="chip">{session.name}</span>
          <button className="linkbtn" onClick={logOut}>
            Abmelden
          </button>
        </div>
      </header>
      <main>
        {tab === "ask" && <Ask session={session} devParams={devParams} />}
        {tab === "docs" && <Documents session={session} />}
        {tab === "quality" && session.is_admin && <Quality />}
      </main>
    </div>
  );
}
