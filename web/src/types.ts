export type Session = {
  token: string;
  login: string;
  name: string;
  departments: string[]; // always starts with "alle"
  is_admin: boolean;
};
export type DemoUser = { login: string; name: string };
export type Citation = { doc_id: string; title: string; section: string; chunk_id: string };
export type Step = { tool: string; args: Record<string, unknown>; summary: string };
export type Mode = "auto" | "classic" | "agentic";
export type AskResult = {
  answer: string;
  citations: Citation[];
  mode: Mode;
  steps: Step[];
  input_tokens: number;
  output_tokens: number;
  cost: number;
  latency_ms: number;
  found: boolean;
  route?: string; // mode "auto": what the router chose (classic | agentic | none)
  ttft_ms?: number | null; // time to first streamed token
  cached?: boolean; // served from the semantic cache
};
export type DocItem = { doc_id: string; title: string; department: string };
export type DocDetail = DocItem & { chunks: { chunk_id: string; heading: string; text: string }[] };
export type EvalMetrics = {
  questions: number;
  hit_rate_at_5: number;
  citation_correctness: number;
  p95_latency_ms: number;
  avg_latency_ms: number;
  total_cost_usd: number;
};
export type ScaleRun = {
  documents: number;
  hit_rate_at_5: number;
  search_p95_ms: number;
  search_p50_ms: number;
};
export type LiveMode = {
  count: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  cost: number;
  not_found_rate: number;
};
export type OptimizationRow = {
  config: string;
  set: "tuning" | "holdout";
  llm: string;
  questions: number;
  citation_correct: number;
  fact_correct: number;
  cost_per_1000: number;
  p50_ms: number;
  tokens_in: number;
  tokens_out: number;
};
export type Metrics = {
  total_requests: number;
  modes: Record<string, LiveMode>;
  top_not_found: { question: string; count: number }[];
  eval: { modes: Record<string, { metrics: EvalMetrics }> } | null;
  scale: { small: ScaleRun; large: ScaleRun } | null;
  optimization?: { rows: OptimizationRow[] } | null;
};
