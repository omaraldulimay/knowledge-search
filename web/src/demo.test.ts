import { describe, expect, it } from "vitest";
import { createDemoApi, type DemoData } from "./demo";
import type { AskResult } from "./types";

const result = (answer: string, found = true): AskResult => ({
  answer, citations: [], mode: "classic", steps: [], input_tokens: 1, output_tokens: 1,
  cost: 0, latency_ms: 1, found,
});
const DATA = {
  generated_by: "fake",
  generated_at: "2026-01-01",
  users: [
    { login: "mira", name: "Mira", departments: ["alle", "hr"], is_admin: false },
    { login: "fiete", name: "Fiete", departments: ["alle"], is_admin: true },
  ],
  questions: {
    mira: [{ question: "Q1?", kind: "single", results: { auto: result("C"), classic: result("A"), agentic: result("B") } }],
    fiete: [{ question: "Q1?", kind: "single", results: { auto: result("C"), classic: result("A"), agentic: result("B") } }],
  },
  documents: { mira: [{ doc_id: "hr/x", title: "X", department: "hr" }] },
  contents: { mira: { "hr/x": { doc_id: "hr/x", title: "X", department: "hr", chunks: [] } } },
  metrics: {},
} as unknown as DemoData;

describe("demo api", () => {
  const make = () => createDemoApi(async () => DATA);

  it("answers prepared questions per mode", async () => {
    const api = make();
    await api.login("fiete");
    expect((await api.ask("Q1?", "auto")).answer).toBe("C");
    expect((await api.ask("Q1?", "classic")).answer).toBe("A");
    expect((await api.ask("Q1?", "agentic")).answer).toBe("B");
  });

  it("gives employees the automatic answer without technical details", async () => {
    const api = make();
    await api.login("mira");
    const r = await api.ask("Q1?", "classic");
    expect(r.answer).toBe("C");
    expect(r).not.toHaveProperty("cost");
    expect(r).not.toHaveProperty("steps");
    await api.login("fiete");
    expect(await api.ask("Q1?", "auto")).toHaveProperty("cost");
  });

  it("refuses free text, unknown documents, uploads and anonymous access", async () => {
    const api = make();
    await expect(api.documents()).rejects.toThrow();
    await api.login("mira");
    await expect(api.ask("Something else", "classic")).rejects.toThrow();
    await expect(api.document("it/secret")).rejects.toThrow("Dokument nicht gefunden");
    await expect(api.upload()).rejects.toThrow();
  });
});
