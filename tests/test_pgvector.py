"""Runs only when PG_DSN points to a Postgres with the pgvector extension."""

import os

import pytest

from ks.auth import allowed_departments, get_user
from ks.embed import HashEmbedder
from ks.ingest import embedding_text
from ks.search import HybridSearcher
from ks.store import PgVectorStore

pytestmark = pytest.mark.skipif(not os.environ.get("PG_DSN"), reason="PG_DSN not set")


@pytest.fixture
def pg_store(chunks):
    embedder = HashEmbedder()
    store = PgVectorStore(os.environ["PG_DSN"], embedder.dim)
    store.init_schema(reset=True)
    store.add(chunks, embedder.embed([embedding_text(c) for c in chunks]))
    yield store
    store.init_schema(reset=True)


def test_count_and_permission_filter(pg_store, chunks):
    assert pg_store.count() == len(chunks)
    allowed = allowed_departments(get_user("katja"))
    assert all(c.department in allowed for c in pg_store.get_chunks(allowed))
    assert pg_store.get_document_chunks("hr/gehaltsbaender", allowed) == []


def test_hybrid_search_never_returns_hr_for_vertrieb(pg_store):
    searcher = HybridSearcher(pg_store, HashEmbedder())
    allowed = allowed_departments(get_user("katja"))
    hits = searcher.search("Gehaltsband Entgeltgruppe E5", allowed, k=10)
    assert hits and all(h.chunk.department != "hr" for h in hits)


def test_finds_expected_document(pg_store):
    searcher = HybridSearcher(pg_store, HashEmbedder())
    allowed = allowed_departments(get_user("ole"))
    hits = searcher.search("VPN Elbtunnel Client einrichten", allowed, k=3)
    assert "it/vpn-einrichtung" in [h.chunk.doc_id for h in hits]
