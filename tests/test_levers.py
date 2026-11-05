"""Round 4 levers for colloquial wording: doc2query and reranker.
Everything runs offline (hash embedder, stub models)."""

import numpy as np

from ks.auth import allowed_departments, get_user
from ks.doc2query import Doc2Query, chunk_key, generate, parse_questions
from ks.llm import LLMResponse, Usage
from ks.search import HybridSearcher


class StubLLM:
    model = "stub"
    price_in = price_out = 1.0

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=None, on_text=None, think=True):
        self.calls += 1
        return LLMResponse(
            "1. Wie viel Kohle gibt es?\n- Was kostet das so?\nquatsch", [], Usage(10, 5)
        )


def test_parse_questions_keeps_only_questions():
    text = "1. Wie viel Kohle gibt es?\n- Was kostet das so?\nkeine frage\nkurz?"
    assert parse_questions(text) == ["Wie viel Kohle gibt es?", "Was kostet das so?"]


def test_generate_is_cached_by_chunk_content(chunks):
    llm, cache = StubLLM(), {}
    generate(chunks[:3], llm, cache)
    assert llm.calls == 3 and set(cache) == {chunk_key(c) for c in chunks[:3]}
    generate(chunks[:3], llm, cache)  # nothing new: no second model call
    assert llm.calls == 3


def test_generate_stops_at_the_budget(chunks):
    import pytest

    llm = StubLLM()
    llm.price_in = 1e9
    with pytest.raises(RuntimeError, match="budget"):
        generate(chunks[:5], llm, {}, limit_usd=1.0)


def _searcher_with_questions(store, embedder, chunks, hr_question):
    hr = next(c for c in chunks if c.department == "hr")
    cache = {chunk_key(hr): [hr_question]}
    return HybridSearcher(store, embedder, doc2query=Doc2Query(chunks, cache, embedder)), hr


def test_doc2query_finds_chunk_by_generated_question(store, embedder, chunks):
    question = "Wie viel Kohle kriegt man für zeltplatzwunder?"
    searcher, hr = _searcher_with_questions(store, embedder, chunks, question)
    hits = searcher.search(question, allowed_departments(get_user("mira")), k=3, d2q=True)
    assert hits[0].chunk.chunk_id == hr.chunk_id
    plain = searcher.search(question, allowed_departments(get_user("mira")), k=3, d2q=False)
    assert plain[0].chunk.chunk_id != hr.chunk_id  # the word only exists in the generated question


def test_doc2query_never_leaks_across_departments(store, embedder, chunks):
    question = "Wie viel Kohle kriegt man für zeltplatzwunder?"
    searcher, hr = _searcher_with_questions(store, embedder, chunks, question)
    hits = searcher.search(question, allowed_departments(get_user("katja")), k=10, d2q=True)
    assert hits and all(h.chunk.department in ("alle", "vertrieb") for h in hits)
    vec = searcher.embed_query(question)
    assert searcher.doc2query.best_per_chunk(vec, {"vertrieb"}) == {}


class _ReverseReranker:
    """Stub cross-encoder that prefers the chunk that is last in the list."""

    def scores(self, query, texts):
        return np.arange(len(texts), dtype=np.float32)


def test_rerank_changes_the_order_and_keeps_permissions(store, embedder):
    searcher = HybridSearcher(store, embedder, reranker=_ReverseReranker())
    allowed = allowed_departments(get_user("katja"))
    plain = searcher.search("Reisekosten Hotel", allowed, k=5)
    reranked = searcher.search("Reisekosten Hotel", allowed, k=5, rerank=True)
    assert [h.chunk.chunk_id for h in plain] != [h.chunk.chunk_id for h in reranked]
    assert all(h.chunk.department in ("alle", "vertrieb") for h in reranked)
