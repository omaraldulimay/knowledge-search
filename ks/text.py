"""Tokenizer shared by keyword search, key-term extraction and sentence scoring."""

import re

UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"})  # Gehaltsbänder ~ Gehaltsband
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
