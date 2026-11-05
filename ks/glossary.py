"""Glossary: synonyms and abbreviations used to widen the keyword query (no LLM).

data/glossary.yaml lists groups of words that mean the same ("bAV" = "betriebliche Altersvorsorge").
If a query contains one word of a group, the other words are added to the BM25 query.
The `auto` section is seeded from the documents (`python -m ks.cli seed-glossary`),
the `curated` section is maintained by hand."""

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from ks.ingest import Chunk
from ks.text import tokenize

GLOSSARY_FILE = Path(__file__).resolve().parent.parent / "data" / "glossary.yaml"
ACRONYM_RE = re.compile(r"([A-ZÄÖÜ][\w-]+(?: [\wäöüÄÖÜß-]+){0,3}) \(([A-ZÄÖÜ]{2,6})\)")


class Glossary:
    def __init__(self, groups: Iterable[Iterable[str]] = ()) -> None:
        # each term is stored as its token set, e.g. "betriebliche Altersvorsorge" -> {betrieb, ...}
        self.groups = [[str(t) for t in g] for g in groups]
        self._tokens = [[frozenset(tokenize(t)) for t in g] for g in self.groups]

    @classmethod
    def load(cls, path: Path = GLOSSARY_FILE) -> "Glossary":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([*data.get("curated", []), *data.get("auto", [])])

    def expand(self, query: str) -> list[str]:
        """Extra search words for `query` (empty list if no glossary term occurs in it)."""
        words = set(tokenize(query))
        extra: list[str] = []
        for group, token_sets in zip(self.groups, self._tokens, strict=True):
            hit = [i for i, ts in enumerate(token_sets) if ts and ts <= words]
            if not hit:
                continue
            for i, term in enumerate(group):
                if i not in hit and not token_sets[i] <= words:
                    extra.append(term)
        return extra


def seed_from_chunks(chunks: Iterable[Chunk], key_terms: dict[str, list[str]]) -> list[list[str]]:
    """Automatic seed, two sources:
    1. every "Name (ABC)" in the documents is a group [ABC, Name];
    2. key terms of one document where one contains the other ("rabatt" / "rabattstaffel")
       are variants of the same idea."""
    groups: dict[str, list[str]] = {}
    for c in chunks:
        for text in (c.title, c.text):
            for name, abbr in ACRONYM_RE.findall(text):
                groups.setdefault(abbr, [abbr, name.strip()])
    for terms in key_terms.values():
        for short in terms:
            if len(short) == 7:  # 7-letter prefixes are tokenizer artifacts, not real words
                continue
            longer = [
                t
                for t in terms
                if t.startswith(short) and len(t) >= len(short) + 4 and len(short) >= 6
            ]
            if longer:
                groups.setdefault(short, [short, *longer[:2]])
    return sorted(groups.values())
