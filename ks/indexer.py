"""Incremental indexing: only new or changed documents are embedded, deleted ones are removed.

The manifest (`manifest.json` next to the saved index) maps doc_id -> content hash. Comparing it
with the hashes of the files on disk tells us exactly what to do - no timestamps involved."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ks.embed import Embedder
from ks.ingest import Document, chunk_documents, embedding_text
from ks.store import VectorStore

MANIFEST = "manifest.json"
BATCH_SIZE = 128


@dataclass(frozen=True)
class IndexStats:
    added: int
    changed: int
    removed: int
    unchanged: int
    chunks_embedded: int
    seconds: float


def load_manifest(directory: Path) -> dict[str, str]:
    path = directory / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_manifest(directory: Path, manifest: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def embed_in_batches(
    embedder: Embedder, texts: Sequence[str], batch_size: int = BATCH_SIZE,
    progress: Callable[[str], None] | None = None,
):  # fmt: skip
    """Embed texts batch by batch (bounded memory) and report progress."""
    import numpy as np

    parts = []
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        parts.append(embedder.embed(texts[i : i + batch_size]))
        done = min(i + batch_size, len(texts))
        if progress and (done == len(texts) or (i // batch_size) % 10 == 0):
            rate = done / max(time.perf_counter() - start, 1e-9)
            progress(f"  embedded {done}/{len(texts)} chunks ({rate:.0f}/s)")
    return np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)


def sync_index(
    docs: Sequence[Document], store: VectorStore, embedder: Embedder, manifest: dict[str, str],
    batch_size: int = BATCH_SIZE, progress: Callable[[str], None] | None = None,
) -> IndexStats:  # fmt: skip
    """Bring `store` and `manifest` in line with `docs`. Mutates both."""
    start = time.perf_counter()
    current = {d.doc_id: d for d in docs}
    removed = [i for i in manifest if i not in current]
    changed = [i for i, d in current.items() if i in manifest and manifest[i] != d.content_hash]
    added = [i for i in current if i not in manifest]

    store.remove_documents(removed + changed)  # changed docs are re-added below
    todo = [current[i] for i in added + changed]
    chunks = chunk_documents(todo)
    vectors = embed_in_batches(
        embedder, [embedding_text(c) for c in chunks], batch_size, progress
    )
    store.add(chunks, vectors)

    for doc_id in removed:
        del manifest[doc_id]
    for doc in todo:
        manifest[doc.doc_id] = doc.content_hash
    return IndexStats(
        added=len(added),
        changed=len(changed),
        removed=len(removed),
        unchanged=len(current) - len(added) - len(changed),
        chunks_embedded=len(chunks),
        seconds=time.perf_counter() - start,
    )


def upsert_documents(
    docs: Sequence[Document], store: VectorStore, embedder: Embedder, manifest: dict[str, str],
) -> IndexStats:  # fmt: skip
    """Add or replace just these documents (upload), leaving everything else untouched."""
    start = time.perf_counter()
    existing = [d.doc_id for d in docs if d.doc_id in manifest]
    store.remove_documents(existing)
    chunks = chunk_documents(list(docs))
    store.add(chunks, embed_in_batches(embedder, [embedding_text(c) for c in chunks]))
    for doc in docs:
        manifest[doc.doc_id] = doc.content_hash
    return IndexStats(
        added=len(docs) - len(existing), changed=len(existing), removed=0, unchanged=0,
        chunks_embedded=len(chunks), seconds=time.perf_counter() - start,
    )  # fmt: skip
