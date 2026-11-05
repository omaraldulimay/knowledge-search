"""Context compression: send the model only what it needs.

1. Every chunk is cut down to the sentences that match the question (plus its heading).
2. Chunks of the same section are merged, repeated sentences (chunk overlap) are dropped.
3. Chunks get short keys [1], [2], ... in the prompt. The model cites the keys; code maps them
   back to (document, section, chunk id). Short keys save input AND output tokens."""

import re
from collections.abc import Iterable

from ks.ingest import Chunk
from ks.llm import KEY_RE
from ks.search import Hit, HybridSearcher
from ks.text import split_sentences, tokenize

MAX_SENTENCES = 2  # sentences kept per section
MAX_CHARS = 350  # characters kept per section by meaning alone
FACT_RANKS = 2  # only the best sections keep every sentence with a number
MAX_CHARS_FACTS = 700  # hard cap per section once fact sentences and neighbours are added
FACT_RE = re.compile(r"\d")  # numbers, amounts, dates, deadlines, percentages all contain a digit
MAX_SECTIONS = 4  # sections in one prompt
SURE_RANKS = 3  # the best hits are kept even if no word matches (paraphrased questions)
SENTENCE_FLOOR = {"fastembed": 0.45, "hash": 0.1}  # weak sections below this are dropped
WORD_BONUS = 0.1  # score added per question word found in a sentence


def label(chunk: Chunk) -> str:
    """Long citation label [doc_id#Abschnitt] as used in the answers and the UI."""
    return f"{chunk.doc_id}#{chunk.heading}"


class Scorer:
    """Scores sentences against one question: embedding similarity (meaning) plus a small bonus
    for every question word the sentence contains (exact terms such as "E5" or "VPN")."""

    def __init__(self, searcher: HybridSearcher, question: str, terms: set[str]) -> None:
        self.searcher = searcher
        self.terms = terms
        self.vector = searcher.embed_query(question)
        self.floor = SENTENCE_FLOOR.get(searcher.embedder.name, 0.4)

    def score(self, sentences: list[str]) -> list[float]:
        sims = self.searcher.embed_texts(sentences) @ self.vector
        words = [len(self.terms & set(tokenize(s))) for s in sentences]
        return [float(sim) + WORD_BONUS * min(n, 3) for sim, n in zip(sims, words, strict=True)]


def best_sentences(
    text: str, scorer: Scorer, max_sentences: int | None = None, max_chars: int | None = None,
    seen: set[str] | None = None, need_match: bool = False, heading: str = "",
    keep_facts: bool = False,
) -> str:  # fmt: skip
    """The sentences of `text` that fit the question best (see Scorer), in original order.
    The heading is scored too ("Evakuierung von Besuchern" helps its section). Sentences already
    in `seen` (overlap between chunks) are skipped. With `need_match` a section whose best
    sentence is below the similarity floor yields nothing.

    With `keep_facts` (the prompt for the model) the cut is gentler, because numbers decide an
    answer: every sentence with a digit (amount, limit, date, deadline, percentage) stays, and so
    does the direct neighbour of each chosen sentence (a condition often sits next to its rule).
    List items and table rows are separate "sentences", so they are kept whole."""
    max_sentences, max_chars = max_sentences or MAX_SENTENCES, max_chars or MAX_CHARS
    seen = seen if seen is not None else set()
    sentences = [s for s in split_sentences(text) if s.lower() not in seen]
    if not sentences:
        return ""
    scores = scorer.score(sentences)
    if heading:  # a matching heading lifts every sentence of its section a little
        bonus = max(scorer.score([heading])[0] - scorer.floor, 0) * 0.5
        scores = [s + bonus for s in scores]
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    if need_match and scores[ranked[0]] < scorer.floor:
        return ""
    picked, used = [], 0
    for i in ranked[:max_sentences]:
        if picked and used + len(sentences[i]) > max_chars:
            break
        picked.append(i)
        used += len(sentences[i])
    named = bool(scorer.terms & set(tokenize(heading)))  # the heading names a question word
    if keep_facts and (scores[ranked[0]] >= scorer.floor or named):  # real matches only
        # priority: best sentences, then sentences with numbers, then neighbours; hard char cap
        extra = [i for i in range(len(sentences)) if FACT_RE.search(sentences[i])]
        extra += [j for i in picked for j in (i - 1, i + 1) if 0 <= j < len(sentences)]
        for i in extra:
            if i not in picked and used + len(sentences[i]) <= MAX_CHARS_FACTS:
                picked.append(i)
                used += len(sentences[i])
    chosen = [sentences[i] for i in sorted(picked)]
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
    hits: Iterable[Hit], scorer: Scorer, keys: KeyMap, max_sections: int | None = None
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
        weak = (rank >= SURE_RANKS or bool(first.via)) and not (
            scorer.terms & set(tokenize(first.chunk.heading))  # a question word in the heading
        )
        text = best_sentences(
            "\n".join(h.chunk.text for _, h in parts),
            scorer,
            seen=seen,
            need_match=weak,
            heading=first.chunk.heading,
            keep_facts=rank < FACT_RANKS or bool(scorer.terms & set(tokenize(first.chunk.heading))),
        )
        if text:
            sections.append((rank, first.chunk, text))
    sections = sections[: max_sections or MAX_SECTIONS]
    return [(keys.add(chunk), chunk, text) for _, chunk, text in sections]


def build_prompt(question: str, sections: list[tuple[str, Chunk, str]]) -> str:
    blocks = "\n".join(f"[{key}] {c.heading}\n{text}" for key, c, text in sections)
    return f"Frage: {question}\n\n{blocks}"
