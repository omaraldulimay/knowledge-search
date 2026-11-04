"""Offline tests: generator, newest-version rule, incremental indexing."""

import importlib.util
import shutil
from pathlib import Path

from ks.auth import allowed_departments, get_user
from ks.embed import HashEmbedder
from ks.indexer import sync_index
from ks.ingest import Chunk, load_documents, parse_front_matter
from ks.search import HybridSearcher, apply_version_rule
from ks.store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gen_corpus", ROOT / "scripts" / "gen_corpus.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _corpus(tmp_path: Path, n: int = 50):
    shutil.copytree(ROOT / "data" / "docs", tmp_path / "real")
    gen.generate(n, seed=1, out=tmp_path / "gen")
    return load_documents(tmp_path / "real") + load_documents(tmp_path / "gen")


def test_generator_is_deterministic_and_bounded(tmp_path):
    gen.generate(60, seed=7, out=tmp_path / "a")
    gen.generate(60, seed=7, out=tmp_path / "b")
    files_a = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*.md"))
    files_b = sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*.md"))
    assert files_a == files_b and len(files_a) == 60
    assert all(
        (tmp_path / "a" / f).read_text() == (tmp_path / "b" / f).read_text() for f in files_a
    )
    assert any("stand-2019" in f.name for f in files_a)  # distractors present


def test_front_matter_is_parsed_and_stripped(tmp_path):
    docs = {d.doc_id: d for d in _corpus(tmp_path)}
    old = docs["alle/reisekostenrichtlinie-stand-2019"]
    assert (
        old.superseded and old.valid_from == "2019-01-01" and old.policy == "reisekostenrichtlinie"
    )
    assert not old.text.startswith("---")
    new = docs["alle/reisekostenrichtlinie"]
    assert not new.superseded and new.valid_from == "2026-01-01"
    assert parse_front_matter("kein front matter") == ({}, "kein front matter")


def test_newest_version_wins_against_distractor(tmp_path):
    docs = _corpus(tmp_path)
    embedder = HashEmbedder()
    store = MemoryStore()
    sync_index(docs, store, embedder, {})
    searcher = HybridSearcher(store, embedder)
    query = "Hotel Übernachtung Obergrenze Euro pro Nacht Reisekosten"
    allowed = allowed_departments(get_user("mira"))
    # without the rule the outdated version is among the candidates ...
    raw = searcher.store.search_vector(embedder.embed([query])[0], 20, allowed)
    assert any(s.chunk.doc_id.startswith("alle/reisekostenrichtlinie-stand") for s in raw)
    # ... with the rule only the current policy is returned
    ids = [h.chunk.doc_id for h in searcher.search(query, allowed, k=5)]
    assert "alle/reisekostenrichtlinie" in ids
    assert not any("-stand-" in i for i in ids)


def _chunk(cid, policy, valid_from, superseded):
    return Chunk(cid, "d/" + cid, "T", "alle", "H", "x", 0, valid_from, superseded, policy)


def test_version_rule_unit():
    old = _chunk("old", "p", "2019-01-01", True)
    new = _chunk("new", "p", "2026-01-01", False)
    other = _chunk("faq", "", "", False)
    by_id = {c.chunk_id: c for c in (old, new, other)}
    assert set(apply_version_rule({"old": 1.0, "new": 0.5, "faq": 0.1}, by_id)) == {"new", "faq"}
    # only the outdated version found: kept, but penalised
    assert apply_version_rule({"old": 1.0}, by_id) == {"old": 0.5}


def test_incremental_indexing(tmp_path):
    shutil.copytree(ROOT / "data" / "docs", tmp_path / "docs")
    embedder, store, manifest = HashEmbedder(), MemoryStore(), {}
    first = sync_index(load_documents(tmp_path / "docs"), store, embedder, manifest)
    assert first.added == 35 and first.chunks_embedded == store.count()
    total = store.count()

    again = sync_index(load_documents(tmp_path / "docs"), store, embedder, manifest)
    assert (again.added, again.changed, again.removed, again.chunks_embedded) == (0, 0, 0, 0)

    changed = tmp_path / "docs" / "it" / "mfa.md"
    changed.write_text(changed.read_text() + "\n\n## Neu\n\nZusätzlicher Abschnitt.\n")
    (tmp_path / "docs" / "hr" / "elternzeit.md").unlink()
    (tmp_path / "docs" / "hr" / "neu.md").write_text("# Neu\n\n## A\n\nNeuer Text.\n")
    third = sync_index(load_documents(tmp_path / "docs"), store, embedder, manifest)
    assert (third.added, third.changed, third.removed) == (1, 1, 1)
    ids = {c.doc_id for c in store.get_chunks({"alle", "hr", "it", "vertrieb", "logistik"})}
    assert "hr/elternzeit" not in ids and "hr/neu" in ids
    assert any(
        "Zusätzlicher Abschnitt" in c.text for c in store.get_document_chunks("it/mfa", {"it"})
    )
    assert store.count() != total
