"""Embedders turn text into unit-length vectors. Two implementations:

- FastEmbedEmbedder: real multilingual model, runs locally (optional extra "local").
- HashEmbedder: deterministic bag-of-words hashing, no download - used in tests.
"""

import atexit
import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

# 384 dimensions, supports German
FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# Cosine scores of different models live on different scales (MiniLM: unrelated ~0.1, a good match
# ~0.7; e5: unrelated ~0.7). All thresholds in this code base (relevance gate, router, rewrite)
# were set on the MiniLM scale, so other models map their cosine onto it with a straight line
# score = a * cosine + b. The two anchor points (median top-1 cosine of answerable and of
# unanswerable tuning questions) are measured once with `python -m ks.cli calibrate`.
CALIBRATION: dict[str, tuple[float, float]] = {}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised, dtype float32."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Hashing trick: every word is hashed into one of `dim` buckets. Weak but deterministic."""

    name = "hash"
    spec = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                # md5 instead of hash(): Python's hash() changes between runs
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
        return _normalise(out)


# Models that were trained with a text prefix for queries and for documents. Without it they
# work noticeably worse. (query prefix, document prefix)
PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
    "Qwen/Qwen3-Embedding-0.6B-Q": (
        "Instruct: Given a question of an employee, retrieve the passage that answers it\nQuery:",
        "",
    ),
}


class FastEmbedEmbedder:
    """Local multilingual embeddings via fastembed (ONNX, no GPU, no API key).

    `embed` is for documents (chunks, sentences); `embed_queries` is for questions and adds the
    query prefix of models that need one (e5, Gemma, Qwen3)."""

    name = "fastembed"

    def __init__(self, model_name: str = FASTEMBED_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('fastembed missing: pip install -e ".[local]"') from exc
        self.model_name = model_name
        self.spec = "fastembed" if model_name == FASTEMBED_MODEL else f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        self._query_prefix, self._doc_prefix = PREFIXES.get(model_name, ("", ""))
        self.dim = len(next(iter(self._model.embed(["Test"]))))
        self.slope, self.offset = CALIBRATION.get(model_name, (1.0, 0.0))
        # disk cache of vectors (key = hash of the exact text): re-indexing and ablation variants
        # never embed the same text twice. Lives in a gitignored folder.
        cache_dir = os.environ.get("KS_EMBED_CACHE", ".ks_cache")
        slug = re.sub(r"[^\w.-]", "_", model_name)
        self._cache_file = Path(cache_dir) / f"{slug}.npz"
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        if self._cache_file.exists():
            saved = np.load(self._cache_file, allow_pickle=False)
            self._cache = dict(zip(saved["keys"].tolist(), saved["vectors"], strict=True))
        atexit.register(self.flush)

    def flush(self) -> None:
        if self._dirty and self._cache:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self._cache_file, keys=np.array(list(self._cache)),
                vectors=np.stack(list(self._cache.values())),
            )  # fmt: skip
            self._dirty = False

    def calibrate(self, cosine):
        """Cosine similarity -> score on the MiniLM scale (works on floats and arrays)."""
        return self.slope * cosine + self.offset

    def _run(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        keys = [hashlib.sha1((prefix + t).encode()).hexdigest() for t in texts]
        todo = {k: prefix + t for k, t in zip(keys, texts, strict=True) if k not in self._cache}
        if todo:
            raw = np.array(list(self._model.embed(list(todo.values()))), dtype=np.float32)
            self._cache.update(zip(todo, _normalise(raw), strict=True))
            self._dirty = True
        return np.stack([self._cache[k] for k in keys])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, self._doc_prefix)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(texts, self._query_prefix)


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder()
    if name.startswith("fastembed:"):  # "fastembed:<model>" = another local model
        return FastEmbedEmbedder(name.split(":", 1)[1])
    raise ValueError(f"unknown embedder: {name}")
