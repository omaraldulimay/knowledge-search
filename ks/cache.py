"""Semantic answer cache (lever 5).

Key = question embedding + the exact set of departments the user may read. An answer is reused
only if the cosine similarity is >= 0.95 AND the department set is identical, so an answer that
was built from HR documents can never be served to a user without HR rights.
The answer mode is part of the scope too (an "agentic" request is not answered by a "classic"
cache entry). A hit costs nothing: no search, no model call."""

import time
from dataclasses import replace

import numpy as np

from ks.rag import AskResult

THRESHOLD = 0.95
MAX_PER_SCOPE = 500  # oldest entries are dropped first


class SemanticCache:
    def __init__(self, threshold: float = THRESHOLD, calibrate=None) -> None:
        self.threshold = threshold
        self.calibrate = calibrate or (lambda x: x)  # embedder cosine -> MiniLM scale
        # one list per scope (department set, mode): [(unit vector, answer)]
        self._entries: dict[tuple[frozenset[str], str], list[tuple[np.ndarray, AskResult]]] = {}
        self.hits = 0
        self.misses = 0

    def lookup(
        self, qvec: np.ndarray, allowed: frozenset[str], mode: str = ""
    ) -> tuple[AskResult, float] | None:
        """(cached answer, similarity) for a paraphrase asked in the same permission scope."""
        started = time.perf_counter()
        entries = self._entries.get(
            (frozenset(allowed), mode), []
        )  # other scopes are never looked at
        if entries:
            sims = self.calibrate(np.stack([v for v, _ in entries]) @ _unit(qvec))
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                self.hits += 1
                ms = (time.perf_counter() - started) * 1000
                steps = [{"tool": "cache", "args": {}, "summary": f"Treffer ({sims[best]:.2f})"}]
                hit = replace(
                    entries[best][1], input_tokens=0, output_tokens=0, cost=0.0, latency_ms=ms,
                    ttft_ms=ms, cached=True, steps=steps,
                )  # fmt: skip
                return hit, float(sims[best])
        self.misses += 1
        return None

    def put(
        self, qvec: np.ndarray, allowed: frozenset[str], result: AskResult, mode: str = ""
    ) -> None:
        entries = self._entries.setdefault((frozenset(allowed), mode), [])
        entries.append((_unit(qvec), result))
        del entries[:-MAX_PER_SCOPE]

    def clear(self) -> None:
        """Documents changed: cached answers may be outdated."""
        self._entries.clear()


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec
