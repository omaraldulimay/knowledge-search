import type { AskResult, DemoUser, DocDetail, DocItem, Metrics, Mode, Session } from "./types";

let token: string | null = null;
export const setToken = (t: string | null) => {
  token = t;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (typeof init.body === "string") headers.set("Content-Type", "application/json");
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = "Etwas ist schiefgelaufen";
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : "Eingabe ungültig");
  }
  return res.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  users: () => call<DemoUser[]>("/users"),
  login: (username: string) => call<Session>("/login", post({ username })),
  ask: (question: string, mode: Mode) => call<AskResult>("/ask", post({ question, mode })),
  documents: () => call<DocItem[]>("/documents"),
  document: (id: string) => call<DocDetail>(`/documents/${id}`),
  upload: (file: File, department: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("department", department);
    return call<DocItem>("/documents", { method: "POST", body: form });
  },
  remove: (id: string) => call<{ deleted: string }>(`/documents/${id}`, { method: "DELETE" }),
  metrics: () => call<Metrics>("/metrics"),
};
