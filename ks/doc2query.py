"""Lever 4: doc2query. Questions that employees might ask are generated for every chunk ONCE, at
index time, and searched next to the chunk (vector and keyword). A colloquial question
("Wie viel Kohle gibt's fürs Kind?") then matches a generated colloquial question even when the
chunk itself says "Zuschuss zur Kinderbetreuung".

Only the chunk (title, heading, text) is given to the model; evaluation questions never are.
The generated questions are cached in data/doc2query.json, keyed by the hash of the chunk text,
so re-indexing does not repeat a model call and demo / CI need no API."""

import hashlib
import json
import re
from collections.abc import Collection, Iterable
from pathlib import Path

import numpy as np

from ks.ingest import Chunk, embedding_text
from ks.llm import LLM, cost_usd

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "doc2query.json"
QUESTIONS_PER_CHUNK = 4
MAX_TOKENS = 160

SYSTEM = (
    "Du hilfst beim Aufbau einer Suche im internen Wissen der Elbkontor Logistik GmbH. Zu einem "
    f"Textabschnitt schreibst du {QUESTIONS_PER_CHUNK} kurze Fragen, die Kolleginnen und Kollegen "
    "im Alltag stellen würden und die dieser Abschnitt beantwortet. Schreibe locker und "
    "umgangssprachlich, wie im Flurgespräch oder Chat: Alltagswörter und Synonyme statt der "
    "Fachbegriffe des Textes (z. B. Chef statt Teamleitung, Kohle statt Vergütung, Rechner statt "
    "Laptop), Abkürzungen, Halbsätze. Schreibe nicht die Sätze des Textes ab. Eine Frage pro "
    "Zeile, ohne Nummerierung, ohne weiteren Text."
)


def chunk_key(chunk: Chunk) -> str:
    """Stable key of a chunk's content (changes when the chunk text changes)."""
    return hashlib.sha1(embedding_text(chunk).encode()).hexdigest()


def load_cache(path: Path = CACHE_FILE) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache(cache: dict[str, list[str]], path: Path = CACHE_FILE) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0, sort_keys=True), "utf-8")


def parse_questions(text: str) -> list[str]:
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in text.splitlines())
    return [ln for ln in lines if ln.endswith("?") and 8 <= len(ln) <= 200][:QUESTIONS_PER_CHUNK]


def generate(
    chunks: Iterable[Chunk],
    llm: LLM,
    cache: dict[str, list[str]],
    limit_usd: float = 0.03,
) -> float:
    """Fill `cache` for chunks without an entry. Returns the cost in USD; stops at the limit."""
    spent = 0.0
    for chunk in chunks:
        key = chunk_key(chunk)
        if key in cache:
            continue
        if spent > limit_usd:
            raise RuntimeError(f"doc2query budget exceeded: ${spent:.4f} > ${limit_usd}")
        user = f"Dokument: {chunk.title}\nAbschnitt: {chunk.heading}\n\n{chunk.text}"
        res = llm.complete(SYSTEM, user, max_tokens=MAX_TOKENS, think=False)
        cache[key] = parse_questions(res.text)
        spent += cost_usd(llm, res.usage.input_tokens, res.usage.output_tokens)
    return spent


class Doc2Query:
    """The generated questions of all chunks as one matrix (one row per question). The searcher
    uses it like a second vector index; the department filter is applied to the chunk."""

    def __init__(self, chunks: Collection[Chunk], cache: dict[str, list[str]], embedder) -> None:
        self.questions: dict[str, list[str]] = {}  # chunk_id -> questions
        rows: list[tuple[Chunk, str]] = []
        for chunk in chunks:
            qs = cache.get(chunk_key(chunk), [])
            if qs:
                self.questions[chunk.chunk_id] = qs
                rows += [(chunk, q) for q in qs]
        self._chunks = [c for c, _ in rows]
        # questions are compared with questions, so they are embedded like queries (e5: "query: ")
        embed = getattr(embedder, "embed_queries", embedder.embed)
        self._matrix = (
            embed([q for _, q in rows]) if rows else np.zeros((0, embedder.dim), np.float32)
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def best_per_chunk(
        self, qvec: np.ndarray, allowed: Collection[str]
    ) -> dict[str, tuple[Chunk, float]]:
        """chunk_id -> (chunk, best cosine of its questions); only allowed departments."""
        if not len(self):
            return {}
        sims = self._matrix @ qvec
        best: dict[str, tuple[Chunk, float]] = {}
        for chunk, sim in zip(self._chunks, sims, strict=True):
            if chunk.department in allowed and (
                chunk.chunk_id not in best or sim > best[chunk.chunk_id][1]
            ):
                best[chunk.chunk_id] = (chunk, float(sim))
        return best

    def text_for(self, chunk_id: str) -> str:
        return " ".join(self.questions.get(chunk_id, []))
