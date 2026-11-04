"""Embedders turn text into unit-length vectors. Two implementations:

- FastEmbedEmbedder: real multilingual model, runs locally (optional extra "local").
- HashEmbedder: deterministic bag-of-words hashing, no download - used in tests.
"""

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

# 384 dimensions, supports German
FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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


class FastEmbedEmbedder:
    """Local multilingual embeddings via fastembed (ONNX, no GPU, no API key)."""

    name = "fastembed"
    dim = 384

    def __init__(self, model_name: str = FASTEMBED_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('fastembed missing: pip install -e ".[local]"') from exc
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        return _normalise(vectors)


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name}")
