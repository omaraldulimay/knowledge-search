"""Hybrid search: vector similarity + BM25 keyword search, merged with reciprocal rank fusion.

Both rankers only ever see chunks of the allowed departments (the store filters them)."""

import re
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from ks.embed import Embedder
from ks.ingest import Chunk, embedding_text
from ks.store import VectorStore

UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"})  # Gehaltsbänder ~ Gehaltsband
SUPERSEDED_PENALTY = 0.5  # score factor for outdated chunks whose newer version was not found
RRF_K = 60  # standard constant from the RRF paper; dampens the influence of top ranks
STOPWORDS = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einen", "ist", "sind", "wie", "was",
    "wer", "wo", "wann", "ich", "wir", "sie", "es", "von", "zu", "zum", "zur", "im", "in", "an",
    "auf", "für", "mit", "bei", "den", "dem", "des", "nach", "als", "wenn", "muss", "kann",
    "viel", "viele", "habe", "hat", "gibt", "darf", "mein", "meine",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lower-case words, umlauts folded, light German suffix stripping.

    Long compound words (German!) also emit a 7-letter prefix, so that "Reisekosten"
    matches "Reisekostenrichtlinie"."""
    tokens: list[str] = []
    for word in re.findall(r"\w+", text.lower().translate(UMLAUTS)):
        if word in STOPWORDS or len(word) < 2:
            continue
        stem = word
        for suffix in ("ern", "en", "er", "es", "em", "e", "n", "s"):
            if len(stem) > 5 and stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        tokens.append(stem)
        if len(stem) >= 10:
            tokens.append(stem[:7])
    return tokens


def apply_version_rule(fused: dict[str, float], by_id: dict[str, Chunk]) -> dict[str, float]:
    """Newest valid version wins. Chunks that share a `policy` key are versions of the same policy:

    1. Only the version with the newest `valid_from` stays; older versions are dropped.
    2. A chunk flagged `superseded` is dropped if a non-superseded version of the same policy
       is among the candidates.
    3. A superseded chunk without such a newer version is kept but its score is halved,
       so that outdated text never outranks a current document.
    Chunks without a policy key are not touched."""
    newest: dict[str, str] = {}
    current: set[str] = set()  # policies for which a non-superseded version is a candidate
    for cid in fused:
        c = by_id[cid]
        if c.policy:
            newest[c.policy] = max(newest.get(c.policy, ""), c.valid_from)
            if not c.superseded:
                current.add(c.policy)
    result: dict[str, float] = {}
    for cid, score in fused.items():
        c = by_id[cid]
        if c.policy and c.valid_from < newest[c.policy]:
            continue  # rule 1
        if c.policy and c.superseded:
            if c.policy in current:
                continue  # rule 2
            score *= SUPERSEDED_PENALTY  # rule 3
        result[cid] = score
    return result


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (higher is better)
    vector_score: float | None  # cosine similarity, None if not in vector candidates
    bm25_score: float | None


class HybridSearcher:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self._bm25_cache: dict[tuple[frozenset[str], int], tuple[BM25Okapi, list[Chunk]]] = {}

    def _bm25_for(self, allowed: frozenset[str]) -> tuple[BM25Okapi, list[Chunk]] | None:
        key = (allowed, self.store.count())  # new data in the store invalidates the cache
        if key not in self._bm25_cache:
            chunks = self.store.get_chunks(allowed)
            if not chunks:
                return None
            corpus = [tokenize(embedding_text(c)) for c in chunks]
            self._bm25_cache[key] = (BM25Okapi(corpus), chunks)
        return self._bm25_cache[key]

    def search(
        self, query: str, allowed_departments: Collection[str], k: int = 5, candidates: int = 20
    ) -> list[Hit]:
        allowed = frozenset(allowed_departments)
        if not allowed or not query.strip():
            return []

        # 1) vector ranking, filtered inside the store
        qvec = self.embedder.embed([query])[0]
        vector_hits = self.store.search_vector(qvec, candidates, allowed)

        # 2) keyword ranking over the allowed chunks only
        bm25_hits: list[tuple[Chunk, float]] = []
        bm25 = self._bm25_for(allowed)
        if bm25 is not None:
            index, chunks = bm25
            scores = index.get_scores(tokenize(query))
            ranked = np.argsort(-scores)[:candidates]
            bm25_hits = [(chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

        # 3) reciprocal rank fusion: score = sum over rankers of 1 / (RRF_K + rank)
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        vec_scores: dict[str, float] = {}
        kw_scores: dict[str, float] = {}
        for rank, sc in enumerate(vector_hits, start=1):
            cid = sc.chunk.chunk_id
            by_id[cid] = sc.chunk
            vec_scores[cid] = sc.score
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            by_id[chunk.chunk_id] = chunk
            kw_scores[chunk.chunk_id] = score
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        fused = apply_version_rule(fused, by_id)
        top = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [Hit(by_id[c], fused[c], vec_scores.get(c), kw_scores.get(c)) for c in top]
