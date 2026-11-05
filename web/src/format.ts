import type { Citation } from "./types";

export const DEPARTMENT_LABEL: Record<string, string> = {
  alle: "Alle",
  hr: "Personal",
  it: "IT",
  vertrieb: "Vertrieb",
  logistik: "Logistik",
};
export const deptLabel = (d: string) => DEPARTMENT_LABEL[d] ?? d;

const de = (n: number, digits = 0) =>
  n.toLocaleString("de-DE", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const fmtMs = (ms: number) =>
  ms >= 1000 ? `${de(ms / 1000, 1)} s` : `${de(ms, ms < 10 ? 1 : 0)} ms`;
export const fmtPct = (x: number) => `${de(x * 100, 0)} %`;
export const fmtInt = (n: number) => de(n);
export const fmtCost = (usd: number) => `$${de(usd, usd < 0.01 ? 4 : 3)}`;

/** Questions to try, per department of the logged-in user. */
export const EXAMPLES: Record<string, string[]> = {
  alle: [
    "Wer gibt meinen Urlaub frei, wenn mein Teamleiter in Halle 2 krank ist?",
    "Wo ist bei Feueralarm der Sammelplatz?",
  ],
  hr: [
    "Wie viel Prozent legt die Firma bei der bAV obendrauf?",
    "Wer gibt eine Weiterbildung für 800 Euro aus dem Lerntopf frei?",
    "Wie viele Feedbackgespräche gibt es in der Probezeit?",
  ],
  it: [
    "Welche Durchwahl hat die IT-Bereitschaft am Wochenende?",
    "Wie viele Stunden darf die Wiederherstellung von ELKO höchstens dauern?",
    "Wie viele Zeichen muss mein Passwort haben?",
  ],
  vertrieb: [
    "Welche Vertragsstrafe droht bei Nordhafen Frische bei zu später Meldung?",
    "Ab welcher Marge muss der Vertriebsleiter ein Angebot freigeben?",
  ],
  logistik: [
    "Was bedeutet K3 in ELKO?",
    "Warum wurde 2025 auf Schichtmodell B umgestellt?",
    "Was tun, wenn bei Nordhafen Frische an Tor 7 die Kühlkette unterbrochen ist?",
  ],
};

export function examplesFor(departments: string[]): string[] {
  const own = departments.filter((d) => d !== "alle").flatMap((d) => EXAMPLES[d] ?? []);
  const pool = own.length > 4 ? own.filter((_, i) => i % 2 === 0) : own;
  return [...pool, ...EXAMPLES.alle].slice(0, 6);
}

export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; n: number };

const LABEL = /\[([^\]#\s]+\/[^\]#]+)#([^\]]+)\]/g;

/** Splits an answer into lines; "[doc#Section]" labels become numbered citation markers. */
export function parseAnswer(answer: string, citations: Citation[]): AnswerPart[][] {
  const index = new Map(citations.map((c, i) => [`${c.doc_id}#${c.section}`, i + 1]));
  return answer
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      const parts: AnswerPart[] = [];
      let last = 0;
      for (const m of line.matchAll(LABEL)) {
        const text = line.slice(last, m.index).replace(/^[-•]\s*/, "");
        if (text.trim()) parts.push({ kind: "text", text });
        const n = index.get(`${m[1]}#${m[2]}`);
        if (n) parts.push({ kind: "cite", n });
        last = (m.index ?? 0) + m[0].length;
      }
      const rest = line.slice(last).replace(/^[-•]\s*/, "");
      if (rest.trim()) parts.push({ kind: "text", text: rest });
      return parts;
    });
}
