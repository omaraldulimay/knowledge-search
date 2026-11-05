"""Hybrid search: vector similarity + BM25 keyword search, merged with reciprocal rank fusion.

Both rankers only ever see chunks of the allowed departments (the store filters them)."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ks.auth import DEPARTMENTS
from ks.embed import Embedder
from ks.glossary import Glossary
from ks.ingest import Chunk, embedding_text
from ks.links import LinkIndex, build_links
from ks.store import VectorStore
from ks.text import STOPWORDS, UMLAUTS, split_sentences, tokenize  # noqa: F401  (re-exported)

SUPERSEDED_PENALTY = 0.5  # score factor for outdated chunks whose newer version was not found
RRF_K = 60  # standard constant from the RRF paper; dampens the influence of top ranks
# graph expansion: look at the documents of the top GRAPH_SEEDS hits, follow links of at least
# GRAPH_MIN_WEIGHT, add at most GRAPH_MAX linked documents, and only if their best chunk is
# similar enough to the question (cosine, per embedder).
WARM_MAX_CHUNKS = 3000  # up to this size all sentence vectors are prepared up front
SENTENCES_FILE = "sentences.npz"  # sentence vectors, written next to the index at ingest time
GRAPH_SEEDS = 3
GRAPH_MIN_WEIGHT = 0.5
GRAPH_MAX = 2
GRAPH_MIN_VECTOR = {"fastembed": 0.5, "hash": 0.15}


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
    via: str = ""  # doc_id whose link brought this hit in (graph expansion), "" = found by search


class HybridSearcher:
    def __init__(
        self, store: VectorStore, embedder: Embedder, glossary: Glossary | None = None,
        links: LinkIndex | None = None,
    ) -> None:  # fmt: skip
        self.store = store
        self.embedder = embedder
        self.glossary = glossary  # optional: query expansion (see search(expand=True))
        self.links = links  # optional: document links (see search(graph=True))
        self._bm25_cache: dict[tuple[frozenset[str], int], tuple[BM25Okapi, list[Chunk]]] = {}
        self._query_vectors: dict[str, np.ndarray] = {}  # embedding memo for repeated questions

    def warm(self, chunks: list[Chunk], directory: Path | None = None, save: bool = False) -> None:
        """Embed every sentence once, up front (context compression scores sentences by meaning,
        which is slow when done cold). Vectors saved at index time are loaded from `directory`."""
        sentences = list(
            dict.fromkeys(s for c in chunks for s in [c.heading, *split_sentences(c.text)])
        )
        if directory and (directory / SENTENCES_FILE).exists():
            saved = np.load(directory / SENTENCES_FILE, allow_pickle=False)
            self._query_vectors.update(zip(saved["texts"].tolist(), saved["vectors"], strict=True))
        missing = [s for s in sentences if s not in self._query_vectors]
        for i in range(0, len(missing), 128):
            self.embed_texts(missing[i : i + 128])
        if save and directory and (missing or not (directory / SENTENCES_FILE).exists()):
            vectors = np.stack([self._query_vectors[s] for s in sentences])
            np.savez(directory / SENTENCES_FILE, texts=np.array(sentences), vectors=vectors)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embeddings with a memo: repeated questions and sentences are not embedded twice."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._query_vectors]
        if todo:
            if len(self._query_vectors) > 20000:
                self._query_vectors.clear()
            for text, vec in zip(todo, self.embedder.embed(todo), strict=True):
                self._query_vectors[text] = vec
        return np.stack([self._query_vectors[t] for t in texts])

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
        self, query: str, allowed_departments: Collection[str], k: int = 5, candidates: int = 20,
        expand: bool = False, graph: bool = False,
    ) -> list[Hit]:  # fmt: skip
        allowed = frozenset(allowed_departments)
        if not allowed or not query.strip():
            return []

        # 1) vector ranking, filtered inside the store
        qvec = self.embed_query(query)
        vector_hits = self.store.search_vector(qvec, candidates, allowed)

        # 2) keyword ranking over the allowed chunks only
        bm25_hits: list[tuple[Chunk, float]] = []
        bm25 = self._bm25_for(allowed)
        if bm25 is not None:
            index, chunks = bm25
            terms = tokenize(query)
            if expand and self.glossary:  # glossary: add synonyms / spelled-out abbreviations
                terms += tokenize(" ".join(self.glossary.expand(query)))
            scores = index.get_scores(terms)
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
        hits = [Hit(by_id[c], fused[c], vec_scores.get(c), kw_scores.get(c)) for c in top]
        if graph and self.links and hits:
            hits += self._graph_hits(qvec, hits, allowed)
        return hits

    def _graph_hits(self, qvec: np.ndarray, hits: list[Hit], allowed: frozenset[str]) -> list[Hit]:
        """Graph expansion: best chunk of documents linked to the top hits, if similar enough.
        The store applies the permission filter, so a link into a forbidden department yields
        nothing (and reveals nothing)."""
        assert self.links is not None
        seen = {h.chunk.doc_id for h in hits}
        seeds = list(dict.fromkeys(h.chunk.doc_id for h in hits))[:GRAPH_SEEDS]
        weight: dict[str, tuple[float, str]] = {}
        for seed in seeds:
            for link in self.links.linked(seed, GRAPH_MIN_WEIGHT):
                if link["to"] not in seen and link["weight"] > weight.get(link["to"], (0, ""))[0]:
                    weight[link["to"]] = (link["weight"], seed)
        wanted = sorted(weight, key=lambda d: -weight[d][0])
        if not wanted:
            return []
        floor = GRAPH_MIN_VECTOR.get(self.embedder.name, 0.45)
        found = self.store.search_documents(qvec, wanted, allowed)  # permission filter inside
        found = [sc for sc in found if sc.score >= floor]
        found.sort(key=lambda sc: -(weight[sc.chunk.doc_id][0] * sc.score))
        return [
            Hit(sc.chunk, 0.0, sc.score, None, via=weight[sc.chunk.doc_id][1])
            for sc in found[:GRAPH_MAX]
        ]


def build_searcher(
    store: VectorStore, embedder: Embedder, index_dir: Path | None = None, save: bool = False,
    warm: bool = False,
) -> HybridSearcher:  # fmt: skip
    """Searcher with glossary and link index. The saved links.json is reused if it describes
    exactly the documents in the store, otherwise the links are rebuilt (and saved if asked)."""
    chunks = store.get_chunks(DEPARTMENTS)  # the link index covers all departments
    doc_ids = {c.doc_id for c in chunks}
    links = LinkIndex.load(index_dir) if index_dir else None
    if links is None or set(links.key_terms) != doc_ids:
        links = build_links(chunks)
        if save and index_dir:
            links.save(index_dir)
    searcher = HybridSearcher(store, embedder, Glossary.load(), links)
    if warm and len(chunks) <= WARM_MAX_CHUNKS:  # small corpus: sentence vectors for all of it
        searcher.warm(chunks, index_dir, save)
    return searcher
