"""Lever 2: local cross-encoder reranker.

Vector and keyword search compare the question and a chunk separately. A cross-encoder reads both
together and scores how well the chunk answers the question, which is better at colloquial
wording. It runs on the CPU (fastembed, ONNX), is free and adds some latency, so it only looks at
the best RERANK_POOL candidates of the hybrid search."""

import os
from collections.abc import Sequence

import numpy as np

RERANK_MODEL = os.environ.get("KS_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_POOL = int(os.environ.get("KS_RERANK_POOL", "8"))  # hybrid candidates that are reranked


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('fastembed missing: pip install -e ".[local]"') from exc
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def scores(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """One relevance score per text (higher is better; scale depends on the model)."""
        return np.array(list(self._model.rerank(query, list(texts))), dtype=np.float32)
