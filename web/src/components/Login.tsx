import { useEffect, useState } from "react";
import { api } from "../api";
import { deptLabel } from "../format";
import DemoBadge from "./DemoBadge";
import type { DemoUser, Session } from "../types";

// Department chips are static demo info; the real permissions are enforced by the API.
const DEPARTMENT_OF: Record<string, string> = {
  mira: "hr",
  ole: "it",
  katja: "vertrieb",
  bent: "logistik",
  fiete: "admin",
};

export default function Login({ onLogin }: { onLogin: (s: Session) => void }) {
  const [users, setUsers] = useState<DemoUser[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .users()
      .then(setUsers)
      .catch(() => setError("Der Dienst ist gerade nicht erreichbar."));
  }, []);

  const pick = (login: string) =>
    api
      .login(login)
      .then(onLogin)
      .catch(() => setError("Anmeldung fehlgeschlagen."));

  return (
    <div className="login">
      <p className="wordmark">Elbkontor</p>
      <h1>Wissenssuche</h1>
      <p className="muted">Wer bist du?</p>
      <DemoBadge />
      <ul className="people">
        {users.map((u) => {
          const dept = DEPARTMENT_OF[u.login];
          return (
            <li key={u.login}>
              <button className="person" onClick={() => pick(u.login)}>
                <span className="avatar" aria-hidden="true">
                  {u.name[0]}
                </span>
                <span className="person-name">{u.name}</span>
                <span className="chip">{dept === "admin" ? "Administration" : deptLabel(dept ?? "")}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
    </div>
  );
}
