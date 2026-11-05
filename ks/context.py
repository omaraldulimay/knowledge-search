"""Context compression: send the model only what it needs.

1. Every chunk is cut down to the sentences that match the question (plus its heading).
2. Chunks of the same section are merged, repeated sentences (chunk overlap) are dropped.
3. Chunks get short keys [1], [2], ... in the prompt. The model cites the keys; code maps them
   back to (document, section, chunk id). Short keys save input AND output tokens."""

import re
from collections.abc import Iterable

from ks.ingest import Chunk
from ks.llm import KEY_RE
from ks.search import Hit
from ks.text import tokenize

MAX_SENTENCES = 2  # sentences kept per section
MAX_CHARS = 320  # characters kept per section
MAX_SECTIONS = 4  # sections in one prompt
SURE_RANKS = 3  # the best hits are kept even if no word matches (paraphrased questions)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def label(chunk: Chunk) -> str:
    """Long citation label [doc_id#Abschnitt] as used in the answers and the UI."""
    return f"{chunk.doc_id}#{chunk.heading}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 2]


def best_sentences(
    text: str, query_terms: set[str], max_sentences: int = MAX_SENTENCES,
    max_chars: int = MAX_CHARS, seen: set[str] | None = None, need_match: bool = False,
) -> str:  # fmt: skip
    """The sentences of `text` that share the most words with the question, in original order.
    Score = number of different question words in the sentence (+0.3 if it contains a number).
    Sentences already in `seen` (overlap between chunks) are skipped. If no sentence matches
    any word: the opening sentence (a paraphrased question), or "" when `need_match` is set."""
    seen = seen if seen is not None else set()
    sentences = [s for s in split_sentences(text) if s.lower() not in seen]
    if not sentences:
        return ""
    scored = []
    for i, sentence in enumerate(sentences):
        hits = len(query_terms & set(tokenize(sentence)))
        scored.append((hits + (0.3 if hits and re.search(r"\d", sentence) else 0), -i, i))
    best = [i for score, _, i in sorted(scored, reverse=True)[:max_sentences] if score >= 1]
    if not best:
        if need_match:
            return ""
        best = [0]
    chosen, used = [], 0
    for i in sorted(best):
        if chosen and used + len(sentences[i]) > max_chars:
            break
        chosen.append(sentences[i])
        used += len(sentences[i])
    seen.update(s.lower() for s in chosen)
    return " ".join(chosen)


class KeyMap:
    """Short key -> chunk, in the order the chunks were shown to the model."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def add(self, chunk: Chunk) -> str:
        for key, known in self.chunks.items():
            if known.chunk_id == chunk.chunk_id:
                return key
        key = str(len(self.chunks) + 1)
        self.chunks[key] = chunk
        return key

    def by_label(self) -> dict[str, Chunk]:
        """The same chunks under their long labels (the model may still write those)."""
        return {label(c): c for c in self.chunks.values()}

    def restore(self, answer: str) -> str:
        """Replace [1] / [1, 2] by the long labels [doc#Abschnitt]; unknown numbers vanish."""

        def swap(match: re.Match) -> str:
            keys = re.findall(r"\d+", match.group(1))
            return "".join(
                f"[{label(self.chunks[k])}]" for k in dict.fromkeys(keys) if k in self.chunks
            )

        return KEY_RE.sub(swap, answer)


def build_sections(
    hits: Iterable[Hit], query_terms: set[str], keys: KeyMap, max_sections: int = MAX_SECTIONS
) -> list[tuple[str, Chunk, str]]:
    """(key, first chunk, compressed text) per section, best hits first. Chunks of one section
    are merged. Weak hits (rank > SURE_RANKS, or pulled in by a link) need a matching word,
    otherwise they are dropped; at most `max_sections` sections are sent."""
    seen: set[str] = set()
    merged: dict[tuple[str, str], list[tuple[int, Hit]]] = {}
    for rank, hit in enumerate(hits):
        merged.setdefault((hit.chunk.doc_id, hit.chunk.heading), []).append((rank, hit))
    sections = []
    for parts in merged.values():
        rank, first = parts[0]
        weak = rank >= SURE_RANKS or bool(first.via)
        text = best_sentences(
            "\n".join(h.chunk.text for _, h in parts), query_terms, seen=seen, need_match=weak
        )
        if text:
            sections.append((rank, first.chunk, text))
    sections = sections[:max_sections]
    return [(keys.add(chunk), chunk, text) for _, chunk, text in sections]


def build_prompt(question: str, sections: list[tuple[str, Chunk, str]]) -> str:
    blocks = "\n".join(f"[{key}] {c.heading}\n{text}" for key, c, text in sections)
    return f"Frage: {question}\n\n{blocks}"
