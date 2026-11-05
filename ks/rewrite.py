"""Lever 6: query rewrite for weak searches.

Colloquial questions ("Bei einer Palette steht K5, was heißt das?") share few words with the
documents. If the first search looks weak, one very small model call (about 120 tokens in, 25 out)
turns the question into search keywords; the search runs again and both result lists are merged.
Strong first searches never pay for this. FakeLLM cannot rewrite, so offline runs skip the lever."""

from ks.llm import LLM, NOT_FOUND, cost_usd
from ks.search import Hit, HybridSearcher

REWRITE_SYSTEM = (
    "Du formulierst Fragen von Mitarbeitenden in eine kurze Suchanfrage für das interne "
    "Dokumentenarchiv der Elbkontor Logistik GmbH um. Nenne nur die wichtigsten Stichwörter "
    "(Begriffe, Namen, Abkürzungen auch ausgeschrieben), höchstens 12 Wörter, ohne Satzzeichen, "
    "ohne Antwort."
)
REWRITE_MAX_TOKENS = 40
# the first search is "weak" below this best vector score (calibrated on the tuning set)
WEAK_VECTOR = {"fastembed": 0.6, "hash": 0.3}


def is_weak(hits: list[Hit], embedder_name: str) -> bool:
    best = max((h.vector_score or 0.0 for h in hits), default=0.0)
    return best < WEAK_VECTOR.get(embedder_name, 0.6)


def merge(first: list[Hit], second: list[Hit], k: int = 5) -> list[Hit]:
    """Alternate between the two result lists (original question first), no duplicate chunks."""
    merged: list[Hit] = []
    seen: set[str] = set()
    for pair in zip(first, second, strict=False):
        for hit in pair:
            if hit.chunk.chunk_id not in seen and len(merged) < k:
                seen.add(hit.chunk.chunk_id)
                merged.append(hit)
    for hit in [*first, *second]:
        if hit.chunk.chunk_id not in seen and len(merged) < k:
            seen.add(hit.chunk.chunk_id)
            merged.append(hit)
    return merged


def rewrite_query(question: str, llm: LLM) -> tuple[str, tuple[int, int, float]]:
    """(keywords, (tokens in, tokens out, cost)); empty keywords if nothing usable came back."""
    res = llm.complete(
        REWRITE_SYSTEM, question, max_tokens=REWRITE_MAX_TOKENS, think=False
    )  # fmt: skip
    usage = res.usage
    cost = cost_usd(llm, usage.input_tokens, usage.output_tokens)
    text = " ".join(res.text.split())
    if not text or text.startswith(NOT_FOUND) or len(text) > 200:
        text = ""
    return text, (usage.input_tokens, usage.output_tokens, cost)


def search_with_rewrite(
    question: str, searcher: HybridSearcher, llm: LLM, allowed, hits: list[Hit], **options
) -> tuple[list[Hit], dict | None, tuple[int, int, float]]:
    """Second search with the rewritten question. Returns (hits, step for the log, usage)."""
    keywords, usage = rewrite_query(question, llm)
    if not keywords:
        return hits, None, usage
    second = searcher.search(keywords, allowed, k=5, **options)
    step = {"tool": "rewrite", "args": {"query": keywords}, "summary": f"{len(second)} Treffer"}
    extra = [h for h in hits if h.via]  # keep linked documents found for the original question
    return merge([h for h in hits if not h.via], second) + extra, step, usage
