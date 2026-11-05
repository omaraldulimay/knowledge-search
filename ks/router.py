"""Adaptive mode router (lever 4): decide per question from retrieval signals, no LLM call.

   nothing found            -> answer the not-found sentence directly (0 model calls, 0 cost)
   everything else          -> classic (one model call; glossary + document links already help)
   clearly needs several documents -> agentic (the model searches and reads itself)

Classic is the default. A weak best score does NOT escalate: the glossary-expanded search handles
colloquial wording. Only evidence that the answer spans several documents sends it to the agent.

Signals come from the hybrid search that has to run anyway."""

import re
import time
from dataclasses import dataclass, field

from ks.config import BASELINE, Config
from ks.llm import LLM, NOT_FOUND, TextCallback
from ks.rag import AskResult, Tap, classic_answer, finish, relevant_hits
from ks.search import Hit, HybridSearcher

# "Wie viel darf das kosten UND wie schnell muss ich abrechnen": two question words joined by "und"
QUESTION_WORD = r"(?:wie|was|wann|wer|wo|wohin|welche[rsmn]?|wieviel|muss|darf|kann|gibt)"
MULTI_PART = re.compile(
    rf"\b{QUESTION_WORD}\b[^?]{{6,}}?\bund\b\s+(?:\w+\s+){{0,2}}?{QUESTION_WORD}\b"
)
# two different documents both match strongly (vector score) and the two rankers disagree
STRONG_DOC = {"fastembed": 0.65, "hash": 0.3}
SECOND_DOC = {
    "fastembed": 0.65,
    "hash": 0.3,
}  # a two-part question needs a 2nd doc at least this good (same bar as STRONG_DOC: with ~50
#    documents many unrelated ones score 0.5, and the graph already puts linked documents into
#    the classic prompt, so the agent is kept for questions where two documents clearly compete)


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


def strong_docs(hits: list[Hit], embedder_name: str, bar: float | None = None) -> int:
    """Number of distinct documents whose best vector score is at least `bar`."""
    bar = bar if bar is not None else STRONG_DOC.get(embedder_name, 0.65)
    best: dict[str, float] = {}
    for h in hits:
        best[h.chunk.doc_id] = max(best.get(h.chunk.doc_id, 0.0), h.vector_score or 0.0)
    return sum(1 for v in best.values() if v >= bar)


def decide(hits: list[Hit], embedder_name: str = "fastembed", question: str = "") -> Decision:
    sig = signals(hits)
    if not relevant_hits(hits, embedder_name, soft=True):
        return Decision("none", "keine Treffer über der Schwelle", sig)
    # a two-part question only needs the agent if a second document matches too
    second = SECOND_DOC.get(embedder_name, 0.5)
    if MULTI_PART.search(question.lower()) and strong_docs(hits, embedder_name, second) >= 2:
        return Decision("agentic", "mehrteilige Frage, mehrere Dokumente passen", sig)
    if strong_docs(hits, embedder_name) >= 2 and not sig["rankers_agree"]:
        return Decision("agentic", "mehrere Dokumente passen stark", sig)
    return Decision("classic", "ein Dokument genügt", sig)


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
    decision = decide(hits, searcher.embedder.name, question)
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
