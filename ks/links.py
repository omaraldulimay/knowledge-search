"""Document links ("graph-lite"): a small JSON index of which documents belong together.

Built once at ingest, no LLM. Three sources, from strong to weak:
  (a) explicit references: a document names another one ("siehe ...", or its title / file name),
  (b) key terms: the top TF-IDF words of every document (also used to seed the glossary),
  (c) rare shared key terms: two documents that both use a word almost nobody else uses.
At search time the links pull in the best chunk of a linked document (see search.py).
The index covers ALL departments; permissions are applied when it is used, never here."""

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ks.ingest import Chunk
from ks.text import UMLAUTS, tokenize

LINKS_FILE = "links.json"
KEY_TERMS = 15  # key terms kept per document
RARE_MAX_DOCS = 3  # a key term counts as "rare" if at most this many documents use it
MAX_LINKS = 4  # links kept per document
MIN_WEIGHT = 0.35  # weaker links are dropped
REVERSE_FACTOR = 0.8  # a reference A -> B also links B -> A, a bit weaker
HEAD_WEIGHT = 0.7  # mentioning only the first word of a title ("VPN", "Gefahrgut") is weaker
SIEHE_RE = re.compile(
    r"\b(?:siehe|vgl\.?|s\.\s?auch|finden Sie in|beschreibt die)\s+([^.;:\n]{3,80})", re.I
)


def norm(text: str) -> str:
    """Lower case, umlauts folded, only letters/digits separated by single spaces."""
    return " ".join(re.findall(r"\w+", text.lower().translate(UMLAUTS)))


@dataclass
class LinkIndex:
    links: dict[str, list[dict]] = field(default_factory=dict)  # doc_id -> [{to, weight, why}]
    key_terms: dict[str, list[str]] = field(default_factory=dict)  # doc_id -> top terms

    def linked(self, doc_id: str, min_weight: float = MIN_WEIGHT) -> list[dict]:
        return [link for link in self.links.get(doc_id, []) if link["weight"] >= min_weight]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        data = {"links": self.links, "key_terms": self.key_terms}
        (directory / LINKS_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "LinkIndex | None":
        path = directory / LINKS_FILE
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["links"], data["key_terms"])


def _key_terms(docs: dict[str, str]) -> dict[str, list[str]]:
    """Top TF-IDF terms per document (term frequency x log inverse document frequency)."""
    counts = {d: Counter(tokenize(text)) for d, text in docs.items()}
    df: Counter = Counter()
    for c in counts.values():
        df.update(c.keys())
    n = len(docs)
    result = {}
    for d, c in counts.items():
        scored = {t: (1 + math.log(f)) * math.log(n / df[t]) for t, f in c.items() if len(t) >= 4}
        top = sorted(scored, key=lambda t: (-scored[t], t))[:KEY_TERMS]
        result[d] = top
    return result


def _title_forms(doc_id: str, title: str) -> list[str]:
    """Strings that count as a mention of this document: title, title without parentheses,
    the acronym in parentheses and the file name ("vpn-einrichtung" -> "vpn einrichtung")."""
    forms = {norm(title), norm(re.sub(r"\(.*?\)", "", title)), norm(doc_id.split("/", 1)[1])}
    acronym = re.search(r"\(([A-Z]{3,})\)", title)
    if acronym:
        forms.add(norm(acronym.group(1)))
    return [f for f in forms if len(f) >= 3]


def _head_word(title: str) -> str:
    """First word of a title if it is distinctive: "Gefahrgut-Handling" -> "gefahrgut"."""
    first = norm(re.sub(r"\(.*?\)", "", title)).split(" ")[0]
    return first if len(first) >= 6 or (len(first) >= 3 and title[:3].isupper()) else ""


def _explicit(
    docs: dict[str, str], titles: dict[str, str]
) -> dict[tuple[str, str], tuple[float, str]]:
    """(a) document A mentions document B -> {(A, B): (weight, reason)}."""
    found: dict[tuple[str, str], tuple[float, str]] = {}
    forms = {d: _title_forms(d, t) for d, t in titles.items()}
    title_tokens = {d: set(tokenize(re.sub(r"\(.*?\)", "", t))) for d, t in titles.items()}
    for a, text in docs.items():
        normal = " " + norm(text) + " "
        for b in docs:
            if a == b:
                continue
            if any(f" {f} " in normal for f in forms[b]):
                found[(a, b)] = (1.0, f"nennt '{titles[b]}'")
            elif (head := _head_word(titles[b])) and f" {head}" in normal:
                found[(a, b)] = (HEAD_WEIGHT, f"nennt '{head}'")
        for match in SIEHE_RE.finditer(text):  # "siehe Passwortrichtlinie", "beschreibt die ..."
            words = set(tokenize(match.group(1)))
            for b, tt in title_tokens.items():
                if (
                    b != a
                    and (a, b) not in found
                    and tt & words
                    and len(tt & words) >= min(2, len(tt))
                ):
                    found[(a, b)] = (1.0, f"verweist ('{match.group(0).strip()[:40]}')")
    return found


def build_links(chunks: Iterable[Chunk]) -> LinkIndex:
    """Build the link index from all chunks (all departments)."""
    docs: dict[str, str] = defaultdict(str)
    titles: dict[str, str] = {}
    for c in chunks:
        docs[c.doc_id] += c.text + "\n"
        titles[c.doc_id] = c.title
    docs = dict(docs)
    key_terms = _key_terms({d: titles[d] + " " + titles[d] + " " + t for d, t in docs.items()})

    weights: dict[tuple[str, str], tuple[float, str]] = {}

    def add(a: str, b: str, weight: float, why: str) -> None:
        if weight > weights.get((a, b), (0.0, ""))[0]:
            weights[(a, b)] = (weight, why)

    for (a, b), (weight, why) in _explicit(docs, titles).items():  # (a) explicit references
        add(a, b, weight, why)
        add(b, a, weight * REVERSE_FACTOR, why)

    stems = {d: {t[:7] for t in terms} for d, terms in key_terms.items()}  # compound words
    df: Counter = Counter()
    for terms in stems.values():
        df.update(terms)
    n = len(docs)
    by_stem: dict[str, list[str]] = defaultdict(list)  # rare stem -> documents that use it
    for doc, doc_stems in stems.items():
        for stem in doc_stems:
            if 2 <= df[stem] <= RARE_MAX_DOCS:
                by_stem[stem].append(doc)
    shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for stem, owners in by_stem.items():  # (c) rare shared key terms
        for i, a in enumerate(sorted(owners)):
            for b in sorted(owners)[i + 1 :]:
                shared[(a, b)].append(stem)
    for (a, b), words in shared.items():
        weight = min(0.9, sum(math.log(n / df[t]) for t in words) / 10)  # 2 rare words ~ 0.6
        why = "gemeinsame Begriffe: " + ", ".join(sorted(words)[:4])
        add(a, b, weight, why)
        add(b, a, weight, why)

    links: dict[str, list[dict]] = {}
    for (a, b), (weight, why) in weights.items():
        if weight >= MIN_WEIGHT:
            links.setdefault(a, []).append({"to": b, "weight": round(weight, 3), "why": why})
    for a in links:
        links[a] = sorted(links[a], key=lambda link: -link["weight"])[:MAX_LINKS]
    return LinkIndex(links, key_terms)
