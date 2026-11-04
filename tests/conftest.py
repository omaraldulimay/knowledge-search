from pathlib import Path

import pytest

from ks.embed import HashEmbedder
from ks.ingest import chunk_documents, embedding_text, load_documents
from ks.search import HybridSearcher
from ks.store import MemoryStore

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="session")
def documents():
    return load_documents(DOCS_DIR)


@pytest.fixture(scope="session")
def chunks(documents):
    return chunk_documents(documents)


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def store(chunks, embedder):
    s = MemoryStore()
    s.add(chunks, embedder.embed([embedding_text(c) for c in chunks]))
    return s


@pytest.fixture(scope="session")
def searcher(store, embedder):
    return HybridSearcher(store, embedder)
