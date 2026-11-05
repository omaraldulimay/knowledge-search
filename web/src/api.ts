import { createDemoApi, type DemoData } from "./demo";
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

/** One server-sent event block ("event: x\ndata: {...}") -> [event, data]. */
export function parseSse(block: string): [string, Record<string, unknown>] | null {
  const event = /^event: (.+)$/m.exec(block)?.[1];
  const data = /^data: (.+)$/m.exec(block)?.[1];
  return event && data ? [event, JSON.parse(data)] : null;
}

/** Streams the answer: `onDelta` gets the text as it is written, the promise resolves with the full result. */
async function askStream(question: string, mode: Mode, onDelta: (text: string) => void): Promise<AskResult> {
  const headers = new Headers({ "Content-Type": "application/json" });
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch("/api/ask/stream", { method: "POST", headers, body: JSON.stringify({ question, mode }) });
  if (!res.ok) {
    let detail = "Etwas ist schiefgelaufen";
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : "Eingabe ungültig");
  }
  let done: AskResult | null = null;
  const handle = (block: string) => {
    const parsed = parseSse(block);
    if (!parsed) return;
    const [event, data] = parsed;
    if (event === "delta") onDelta(String(data.text ?? ""));
    if (event === "done") done = data as unknown as AskResult;
    if (event === "error") throw new ApiError(500, String(data.detail ?? "Antwort fehlgeschlagen"));
  };
  if (res.body) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buffer += decoder.decode(value, { stream: true });
      let cut: number;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        handle(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
      }
    }
    if (buffer.trim()) handle(buffer);
  } else {
    (await res.text()).split("\n\n").forEach(handle);
  }
  if (!done) throw new ApiError(500, "Antwort unvollständig");
  return done;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const IS_DEMO = import.meta.env.VITE_DEMO === "1";

const realApi = {
  users: () => call<DemoUser[]>("/users"),
  login: (username: string) => call<Session>("/login", post({ username })),
  ask: (question: string, mode: Mode) => call<AskResult>("/ask", post({ question, mode })),
  askStream,
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

export const demoApi = createDemoApi(async () => {
  const res = await fetch(`${import.meta.env.BASE_URL}demo/demo.json`);
  return (await res.json()) as DemoData;
});

// the static demo has no server: answers are precomputed and appear at once
export const api = IS_DEMO
  ? { ...realApi, ...demoApi, askStream: (q: string, m: Mode, _onDelta: (t: string) => void) => demoApi.ask(q, m) }
  : realApi;
