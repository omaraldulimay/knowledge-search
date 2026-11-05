"""Adaptive mode router (lever 4): decide per question from retrieval signals, no LLM call.

   nothing found            -> answer the not-found sentence directly (0 model calls, 0 cost)
   confident                -> classic (one model call)
   unsure / rankers differ  -> agentic (the model searches and reads itself)

Signals come from the hybrid search that has to run anyway."""

import time
from dataclasses import dataclass, field

from ks.config import BASELINE, Config
from ks.llm import LLM, NOT_FOUND, TextCallback
from ks.rag import AskResult, Tap, classic_answer, finish, relevant_hits
from ks.search import Hit, HybridSearcher

# below this best vector score the retrieval is "unsure"; per embedder like the other thresholds
UNSURE_VECTOR = {"fastembed": 0.5, "hash": 0.2}


@dataclass(frozen=True)
class Decision:
    route: str  # "none" | "classic" | "agentic"
    reason: str
    signals: dict = field(default_factory=dict)


def signals(hits: list[Hit]) -> dict:
    """The numbers the decision is based on (also shown in the step log)."""
    found = [h for h in hits if not h.via]
    best_vec = max((h.vector_score or 0.0 for h in hits), default=0.0)
    best_bm25 = max((h.bm25_score or 0.0 for h in hits), default=0.0)
    top_rrf = found[0].chunk.doc_id if found else ""
    vec_hits = [h for h in found if h.vector_score is not None]
    top_vec = max(vec_hits, key=lambda h: h.vector_score or 0.0).chunk.doc_id if vec_hits else ""
    return {
        "best_vector": round(best_vec, 3),
        "best_bm25": round(best_bm25, 2),
        "rankers_agree": top_rrf == top_vec,  # keyword and vector search favour the same document
        "distinct_docs": len({h.chunk.doc_id for h in found}),
        "linked_docs": sum(1 for h in hits if h.via),
    }


def decide(hits: list[Hit], embedder_name: str = "fastembed") -> Decision:
    sig = signals(hits)
    if not relevant_hits(hits, embedder_name, soft=True):
        return Decision("none", "keine Treffer über der Schwelle", sig)
    if sig["best_vector"] < UNSURE_VECTOR.get(embedder_name, 0.55):
        return Decision("agentic", "unsichere Suche (schwacher bester Treffer)", sig)
    if not sig["rankers_agree"]:
        return Decision("agentic", "Stichwort- und Bedeutungssuche widersprechen sich", sig)
    return Decision("classic", "eindeutiger Treffer", sig)


def answer_auto(
    question: str, user, searcher: HybridSearcher, llm: LLM, cfg: Config = BASELINE,
    on_text: TextCallback | None = None,
) -> AskResult:  # fmt: skip
    from ks.agent import answer_agentic
    from ks.auth import allowed_departments

    started = time.perf_counter()
    tap = Tap(started, on_text)
    hits = searcher.search(
        question, allowed_departments(user), k=5, expand=cfg.glossary, graph=cfg.graph
    )
    decision = decide(hits, searcher.embedder.name)
    retrieved = [h.chunk.doc_id for h in hits]
    if decision.route == "none":
        steps = [{"tool": "router", "args": decision.signals, "summary": decision.reason}]
        result = finish(NOT_FOUND, {}, "auto", steps, (0, 0), started, llm, retrieved, route="none")
        result.ttft_ms = result.latency_ms  # the answer is there at once
        return result
    if decision.route == "classic":
        result = classic_answer(
            question, hits, searcher, llm, cfg, started, tap, soft_gate=True, route="classic"
        )
    else:
        pre_ms = (time.perf_counter() - started) * 1000  # the routing search ran first
        result = answer_agentic(question, user, searcher, llm, cfg=cfg, on_text=on_text)
        result.route = "agentic"
        result.latency_ms += pre_ms
        if result.ttft_ms is not None:
            result.ttft_ms += pre_ms
    result.mode = "auto"
    summary = f"{decision.route}: {decision.reason}"
    result.steps = [{"tool": "router", "args": decision.signals, "summary": summary}, *result.steps]
    return result
