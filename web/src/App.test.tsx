import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { parseAnswer } from "./format";
import type { AskResult } from "./types";

const USERS = [
  { login: "katja", name: "Katja Sommer" },
  { login: "fiete", name: "Fiete Naumann" },
];
const CITATION = { doc_id: "vertrieb/provisionsmodell", title: "Provisionsmodell Vertrieb 2026", section: "Abrechnung", chunk_id: "vertrieb/provisionsmodell#3" };
const FOUND: AskResult = {
  answer: "Laut den Dokumenten:\n- Die Provision wird quartalsweise ausgezahlt. [vertrieb/provisionsmodell#Abrechnung]",
  citations: [CITATION],
  mode: "agentic",
  steps: [{ tool: "search", args: { query: "Provision" }, summary: "5 Zeilen" }],
  input_tokens: 100, output_tokens: 20, cost: 0.0003, latency_ms: 12, found: true,
};
const NOT_FOUND: AskResult = {
  ...FOUND, answer: "Dazu habe ich in den Dokumenten nichts gefunden.", citations: [], found: false,
};
const DOC = {
  doc_id: CITATION.doc_id, title: CITATION.title, department: "vertrieb",
  chunks: [
    { chunk_id: "a#0", heading: "Grundprinzip", text: "Fixgehalt plus Provision." },
    { chunk_id: "a#3", heading: "Abrechnung", text: "Quartalsweise." },
  ],
};

let askResult: AskResult = FOUND;
let isAdmin = false;

function json(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
}

beforeEach(() => {
  askResult = FOUND;
  isAdmin = false;
  sessionStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url === "/api/users") return json(USERS);
      if (url === "/api/login")
        return json({ token: "t", login: "katja", name: "Katja Sommer", departments: ["alle", "vertrieb"], is_admin: isAdmin });
      if (url === "/api/ask/stream")
        return Promise.resolve(
          new Response(
            `event: delta\ndata: {"text": "Laut den "}\n\nevent: done\ndata: ${JSON.stringify(askResult)}\n\n`,
            { status: 200 },
          ),
        );
      if (url.startsWith("/api/documents/")) return json(DOC);
      return json([]);
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

async function loginAsKatja() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(await screen.findByRole("button", { name: /Katja Sommer/ }));
  return user;
}

async function ask(user: ReturnType<typeof userEvent.setup>) {
  await user.type(await screen.findByLabelText("Frage"), "Wie wird die Provision ausgezahlt?");
  await user.click(screen.getByRole("button", { name: "Absenden" }));
}

describe("Wissenssuche", () => {
  it("logs in with one click and hides the quality tab for non-admins", async () => {
    await loginAsKatja();
    expect(await screen.findByRole("heading", { name: "Was möchtest du wissen?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Dokumente" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Qualität" })).not.toBeInTheDocument();
  });

  it("opens the source panel with the cited section highlighted", async () => {
    const user = await loginAsKatja();
    await ask(user);
    await user.click(await screen.findByRole("button", { name: /Quelle 1/ }));
    const panel = await screen.findByRole("complementary", { name: "Quelle" });
    await waitFor(() => expect(panel).toHaveTextContent("Quartalsweise."));
    expect(screen.getByText("Abrechnung").closest("section")).toHaveClass("chunk--hit");
    expect(screen.getByText("Grundprinzip").closest("section")).not.toHaveClass("chunk--hit");
  });

  it("shows a distinct calm style for not-found answers", async () => {
    askResult = NOT_FOUND;
    const user = await loginAsKatja();
    await ask(user);
    expect(await screen.findByText("Nichts gefunden")).toBeInTheDocument();
    expect(screen.getByText(NOT_FOUND.answer).closest("article")).toHaveClass("answer--empty");
  });
});

describe("Mitarbeiter- und Admin-Ansicht", () => {
  it("shows employees only question, answer and sources", async () => {
    const user = await loginAsKatja();
    expect(screen.queryByRole("radiogroup", { name: "Suchmodus" })).not.toBeInTheDocument();
    await ask(user);
    await screen.findByRole("button", { name: /Quelle 1/ });
    expect(screen.queryByText(/Tokens/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Technische Details/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Router entscheidet/)).not.toBeInTheDocument();
  });

  it("gives the admin the mode switch and collapsible technical details", async () => {
    isAdmin = true;
    const user = await loginAsKatja();
    expect(screen.getByRole("radiogroup", { name: "Suchmodus" })).toBeInTheDocument();
    await ask(user);
    await user.click(await screen.findByText("Technische Details"));
    expect(screen.getByText(/Tokens/)).toBeInTheDocument();
  });

  it("keeps the last questions of the session and shows an answer again", async () => {
    const user = await loginAsKatja();
    await ask(user);
    await screen.findByRole("button", { name: /Quelle 1/ });
    expect(sessionStorage.getItem("ks-history:katja")).toContain("Provision");
    const [fromHistory] = within(screen.getByRole("complementary", { name: "Verlauf und Vorschläge" }))
      .getAllByRole("button", { name: "Wie wird die Provision ausgezahlt?" });
    await user.click(fromHistory); // first entry of the list = the history, suggestions come after
    expect(screen.getByRole("button", { name: /Quelle 1/ })).toBeInTheDocument();
    expect(screen.getByText("Deine letzten Fragen")).toBeInTheDocument();
    expect(screen.getByText("Häufig gefragt")).toBeInTheDocument();
  });
});

describe("parseAnswer", () => {
  it("turns labels into numbered markers and drops unknown ones", () => {
    const lines = parseAnswer("- Text [vertrieb/provisionsmodell#Abrechnung] und [x/y#Z]", [CITATION]);
    expect(lines[0]).toEqual([
      { kind: "text", text: "Text " },
      { kind: "cite", n: 1 },
      { kind: "text", text: " und " },
    ]);
  });
});
