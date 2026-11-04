// Static demo: everything comes from one precomputed file (public/demo/demo.json), no server.
import type { AskResult, DemoUser, DocDetail, DocItem, Metrics, Mode, Session } from "./types";

export type DemoQuestion = {
  question: string;
  kind: "single" | "multi" | "example" | "not_found" | "permission";
  results: Record<Mode, AskResult>;
};
export type DemoData = {
  generated_by: string;
  generated_at: string;
  users: (DemoUser & { departments: string[]; is_admin: boolean })[];
  questions: Record<string, DemoQuestion[]>;
  documents: Record<string, DocItem[]>;
  contents: Record<string, Record<string, DocDetail>>;
  metrics: Metrics;
};

export class DemoError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Same surface as the real API client, answering from the precomputed data. */
export function createDemoApi(load: () => Promise<DemoData>) {
  let login = "";
  let cache: Promise<DemoData> | null = null;
  const data = () => (cache ??= load());
  const me = async () => {
    const d = await data();
    if (!login) throw new DemoError(401, "Nicht angemeldet");
    return d;
  };
  const refuse = () => Promise.reject(new DemoError(403, "In der Demo nicht verfügbar"));

  return {
    info: async () => {
      const d = await data();
      return { generated_by: d.generated_by, generated_at: d.generated_at };
    },
    users: async (): Promise<DemoUser[]> =>
      (await data()).users.map((u) => ({ login: u.login, name: u.name })),
    login: async (username: string): Promise<Session> => {
      const u = (await data()).users.find((x) => x.login === username);
      if (!u) throw new DemoError(401, "Unbekannter Benutzer");
      login = u.login;
      return { token: "demo", login: u.login, name: u.name, departments: u.departments, is_admin: u.is_admin };
    },
    questions: async () => (await me()).questions[login].map((q) => ({ question: q.question, kind: q.kind })),
    ask: async (question: string, mode: Mode): Promise<AskResult> => {
      const q = (await me()).questions[login].find((x) => x.question === question);
      if (!q) throw new DemoError(404, "Diese Frage ist in der Demo nicht vorbereitet");
      return q.results[mode];
    },
    documents: async (): Promise<DocItem[]> => (await me()).documents[login],
    document: async (id: string): Promise<DocDetail> => {
      const doc = (await me()).contents[login][id];
      if (!doc) throw new DemoError(404, "Dokument nicht gefunden");
      return doc;
    },
    metrics: async (): Promise<Metrics> => (await me()).metrics,
    upload: refuse,
    remove: refuse,
  };
}
