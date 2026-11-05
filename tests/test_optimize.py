"""Cost and speed levers: links, glossary, compression, router, cache, streaming. Offline."""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ks.api import create_app
from ks.auth import allowed_departments, get_user
from ks.cache import SemanticCache
from ks.config import AUTO, BASELINE, COMPRESSION, GRAPH
from ks.context import KeyMap, Scorer, best_sentences, build_sections
from ks.embed import HashEmbedder
from ks.glossary import Glossary, seed_from_chunks
from ks.links import LinkIndex, build_links
from ks.llm import NOT_FOUND, FakeLLM
from ks.optimize import build_workload, facts_ok, paraphrase
from ks.pipeline import answer
from ks.rag import AskResult, answer_classic
from ks.router import answer_auto, decide
from ks.search import Hit, HybridSearcher
from ks.text import tokenize

ALL = {"alle", "hr", "it", "vertrieb", "logistik"}


@pytest.fixture(scope="module")
def link_index(chunks):
    return build_links(chunks)


@pytest.fixture
def linked(store, embedder, link_index):
    return HybridSearcher(store, embedder, Glossary.load(), link_index)


class ForbiddenLLM(FakeLLM):
    """Fails the test if the model is called."""

    def complete(self, *args, **kwargs):
        raise AssertionError("the model must not be called")

    chat = complete


# ---------------------------------------------------------------- links


def test_links_have_explicit_and_shared_term_sources(link_index):
    targets = {link["to"]: link["why"] for link in link_index.links["alle/onboarding"]}
    assert "it/mfa" in targets and "nennt" in targets["it/mfa"]  # (a) names the MFA document
    assert link_index.key_terms["it/vpn-einrichtung"]  # (b) key terms exist
    assert any(
        "gemeinsame Begriffe" in link["why"] for v in link_index.links.values() for link in v
    )
    assert all(0 < link["weight"] <= 1 for v in link_index.links.values() for link in v)


def test_link_index_json_roundtrip(link_index, tmp_path):
    link_index.save(tmp_path)
    loaded = LinkIndex.load(tmp_path)
    assert loaded is not None and loaded.links == link_index.links
    assert LinkIndex.load(tmp_path / "missing") is None


def test_graph_expansion_adds_linked_document(linked):
    # homeoffice mentions the VPN: its best chunk joins the hits via the link
    q = "Wie arbeite ich im Homeoffice auf interne Systeme zu?"
    plain = linked.search(q, allowed_departments(get_user("ole")), k=5)
    graph = linked.search(q, allowed_departments(get_user("ole")), k=5, graph=True)
    assert len(graph) >= len(plain) and not any(h.via for h in plain)
    assert all(h.via for h in graph[len(plain) :])


def test_graph_expansion_respects_permissions(linked, store, embedder):
    """Links point from allowed documents into hr documents; a vertrieb user must never get them."""
    katja = allowed_departments(get_user("katja"))
    pulled = False
    for q in [
        "Verhaltenskodex Konsequenzen Kündigung Verstoß",
        "Wie lange dauert die Elternzeit und was ist mit dem Urlaub?",
        "Gefahrgut Zulage Gehaltsband Entgeltgruppe",
        "Betriebliche Altersvorsorge Zuschuss Probezeit",
    ]:
        hits = linked.search(q, katja, k=5, graph=True)
        assert not any(h.chunk.department == "hr" for h in hits), q
    mira = allowed_departments(get_user("mira"))  # hr user: the same links DO lead to hr
    for q in ["Urlaub in der Elternzeit"]:
        pulled |= any(h.chunk.department == "hr" for h in linked.search(q, mira, k=5, graph=True))
    assert pulled
    # the store itself refuses documents of other departments
    vec = embedder.embed(["Kündigungsfrist"])[0]
    assert store.search_documents(vec, ["hr/kuendigungsfristen"], katja) == []
    assert store.search_documents(vec, ["hr/kuendigungsfristen"], mira)


def test_forbidden_link_target_also_blocked_in_classic_answer(linked):
    res = answer_classic(
        "Wie hoch ist die Kündigungsfrist laut Verhaltenskodex?",
        get_user("katja"), linked, FakeLLM(), cfg=GRAPH,
    )  # fmt: skip
    assert not any(d.startswith("hr/") for d in res.retrieved_doc_ids)
    assert not any(c.doc_id.startswith("hr/") for c in res.citations)


# ---------------------------------------------------------------- glossary


def test_glossary_expands_abbreviations_and_synonyms():
    g = Glossary.load()
    assert "betriebliche Altersvorsorge" in g.expand("Was zahlt die Firma zur bAV?")
    assert "Homeoffice" in g.expand("Wie oft darf ich HO machen?")
    assert "Ferien" in g.expand("Wie viel Urlaub habe ich")
    assert g.expand("Wie ist das Wetter") == []


def test_glossary_raises_keyword_score(linked):
    ole = allowed_departments(get_user("mira"))
    q = "Wie viel legt die Firma bei der bAV obendrauf?"
    plain = max(h.bm25_score or 0 for h in linked.search(q, ole))
    wide = max(h.bm25_score or 0 for h in linked.search(q, ole, expand=True))
    assert wide > plain


def test_auto_seed_finds_acronyms(chunks, link_index):
    groups = seed_from_chunks(chunks, link_index.key_terms)
    assert any(g[0] == "MFA" for g in groups)


# ---------------------------------------------------------------- compression


def test_best_sentences_picks_matching_sentence_and_skips_repeats(linked):
    text = "Der Hund bellt. Die Probezeit dauert sechs Monate. Das Wetter ist schön."
    scorer = Scorer(
        linked, "Wie lange dauert die Probezeit?", set(tokenize("lange dauert Probezeit"))
    )
    seen: set[str] = set()
    assert best_sentences(text, scorer, 1, seen=seen) == "Die Probezeit dauert sechs Monate."
    assert best_sentences(text, scorer, 1, seen=seen) != "Die Probezeit dauert sechs Monate."


def test_sentence_split_keeps_dates_together():
    from ks.context import split_sentences

    parts = split_sentences("Urlaub verfällt bis zum 31. März des Folgejahres. Danach ist er weg.")
    assert parts == ["Urlaub verfällt bis zum 31. März des Folgejahres.", "Danach ist er weg."]


def test_keymap_restores_long_labels(chunks):
    keys = KeyMap()
    k1, k2 = keys.add(chunks[0]), keys.add(chunks[1])
    assert (k1, k2) == ("1", "2") and keys.add(chunks[0]) == "1"
    label = f"{chunks[0].doc_id}#{chunks[0].heading}"
    assert keys.restore("Ja [1]. Nein [1, 9].") == f"Ja [{label}]. Nein [{label}]."


def test_compressed_prompt_is_shorter_and_citations_resolve(linked):
    q = "Welche Kündigungsfrist gilt in der Probezeit?"
    mira = get_user("mira")
    base = answer_classic(q, mira, linked, FakeLLM(), cfg=BASELINE)
    small = answer_classic(q, mira, linked, FakeLLM(), cfg=COMPRESSION)
    assert small.input_tokens < base.input_tokens * 0.7
    assert small.citations and "hr/kuendigungsfristen" in {c.doc_id for c in small.citations}
    assert "[1]" not in small.answer  # short keys are mapped back to [doc#section] in code


def test_sections_are_limited_and_deduplicated(linked):
    mira = allowed_departments(get_user("mira"))
    hits = linked.search("Probezeit Kündigung", mira, k=5)
    scorer = Scorer(linked, "Probezeit Kündigung", set(tokenize("Probezeit Kündigung")))
    sections = build_sections(hits, scorer, KeyMap())
    assert 1 <= len(sections) <= 5
    texts = [t for _, _, t in sections]
    assert len(set(texts)) == len(texts)


# ---------------------------------------------------------------- router


def _hit(chunk, vec, bm25, via=""):
    return Hit(chunk, 0.03, vec, bm25, via)


def test_router_decisions(chunks):
    a, b = chunks[0], next(c for c in chunks if c.doc_id != chunks[0].doc_id)
    assert decide([], "fastembed").route == "none"
    assert decide([_hit(a, 0.3, 2.0)], "fastembed").route == "none"  # both signals weak
    assert decide([_hit(a, 0.8, 12.0), _hit(b, 0.5, 5.0)], "fastembed").route == "classic"
    assert decide([_hit(a, 0.48, 12.0)], "fastembed").route == "agentic"  # unsure retrieval
    split = [_hit(a, 0.6, 5.0), _hit(b, 0.7, None)]  # keyword and vector search disagree
    assert decide(split, "fastembed").route == "agentic"


def test_auto_answers_nothing_without_any_model_call(linked):
    for question in ["", "xyzzy qwertz plugh"]:  # no word and no meaning in common with the docs
        res = answer_auto(question, get_user("ole"), linked, ForbiddenLLM(), cfg=AUTO)
        assert res.route == "none" and res.answer == NOT_FOUND
        assert res.cost == 0 and res.input_tokens == 0 and res.citations == []


def test_auto_runs_classic_or_agentic_with_valid_citations(linked):
    res = answer_auto(
        "Wie viel darf ein Hotel im Vertrieb pro Nacht kosten?",
        get_user("katja"), linked, FakeLLM(), cfg=AUTO,
    )  # fmt: skip
    assert res.mode == "auto" and res.route in ("classic", "agentic")
    assert res.steps[0]["tool"] == "router" and res.found and res.citations


def test_lean_agent_reads_one_section(linked):
    from ks.agent import ToolBox

    box = ToolBox(linked, ALL, "Wie lang ist die Probezeit?", AUTO)
    section = box.read_document("hr/kuendigungsfristen", "Probezeit")
    whole = ToolBox(linked, ALL).read_document("hr/kuendigungsfristen")
    assert 0 < len(section) < len(whole) / 2
    outline = box.read_document("hr/kuendigungsfristen")  # no section: best parts + outline
    assert len(outline) < len(whole) and "Weitere Abschnitte" in outline


# ---------------------------------------------------------------- cache


def _result(text="Antwort"):
    return AskResult(text, [], "auto", [], 100, 20, 0.001, 900.0, True)


def test_cache_hit_needs_similar_question_and_same_departments():
    cache, vec = SemanticCache(), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hr = frozenset({"alle", "hr"})
    cache.put(vec, hr, _result("nur HR-Wissen"), "auto")
    hit, sim = cache.lookup(vec * 2, hr, "auto")  # same direction: cosine 1.0
    assert hit.cached and hit.cost == 0 and hit.input_tokens == 0 and hit.answer == "nur HR-Wissen"
    near = np.array([0.99, 0.05, 0.0], dtype=np.float32)
    assert cache.lookup(near, hr, "auto") is not None  # cosine ~0.9987
    far = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    assert cache.lookup(far, hr, "auto") is None  # cosine 0.8 < 0.95
    # never across permission scopes: identical question, other department set -> miss
    assert cache.lookup(vec, frozenset({"alle", "vertrieb"}), "auto") is None
    assert cache.lookup(vec, frozenset({"alle", "hr", "it"}), "auto") is None  # not even a superset
    assert cache.lookup(vec, hr, "classic") is None  # other mode
    cache.clear()
    assert cache.lookup(vec, hr, "auto") is None


def test_pipeline_cache_never_leaks_hr_answer_to_vertrieb(linked):
    from ks.config import CACHED

    cache = SemanticCache()
    q = "Welche Kündigungsfrist gilt in der Probezeit?"
    first = answer(q, get_user("mira"), linked, FakeLLM(), "auto", CACHED, cache)
    again = answer(q, get_user("mira"), linked, FakeLLM(), "auto", CACHED, cache)
    assert not first.cached and again.cached and again.cost == 0
    assert again.answer == first.answer and again.latency_ms < 50
    other = answer(q, get_user("katja"), linked, FakeLLM(), "auto", CACHED, cache)
    assert not other.cached  # different department set: computed from scratch
    assert not any(c.doc_id.startswith("hr/") for c in other.citations)


def test_workload_has_about_thirty_percent_repeats():
    qs = [{"id": f"q{i}", "question": f"Frage {i}?", "user": "mira"} for i in range(30)]
    work = build_workload(qs)
    assert work == build_workload(qs)  # deterministic
    repeats = [w for w in work if w["repeat_of"]]
    assert 0.25 <= len(repeats) / len(work) <= 0.35
    firsts = {w["id"]: i for i, w in enumerate(work) if not w["repeat_of"]}
    assert all(firsts[w["repeat_of"]] < i for i, w in enumerate(work) if w["repeat_of"])
    assert paraphrase("Wie viel Urlaub?", 5) == "wie viel urlaub"


def test_facts_ok_normalises_numbers():
    assert facts_ok("Das Budget beträgt 1.200 Euro im Jahr.", ["1200 euro"])
    assert facts_ok("20 % Zuschuss", ["20 prozent"])
    assert not facts_ok("Etwas anderes", ["20 prozent", "zwei wochen|2 wochen"])


# ---------------------------------------------------------------- API: streaming + cache


@pytest.fixture
def optimized_client(tmp_path, monkeypatch):
    from tests.conftest import DOCS_DIR

    monkeypatch.setenv("KS_METRICS_DB", ":memory:")
    monkeypatch.delenv("KS_CONFIG", raising=False)  # default = every lever on
    docs = tmp_path / "docs"
    shutil.copytree(DOCS_DIR, docs)
    app = create_app(
        embedder=HashEmbedder(), llm=FakeLLM(), docs_dir=docs, index_dir=tmp_path / "idx"
    )
    with TestClient(app) as c:
        yield c


def _auth(client, user):
    token = client.post("/login", json={"username": user}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        kind, data = block.split("\n", 1)
        out.append((kind.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_ask_stream_sends_deltas_then_the_full_result(optimized_client):
    q = "Welche Kündigungsfrist gilt in der Probezeit?"
    res = optimized_client.post(
        "/ask/stream",
        json={"question": q, "mode": "classic"},
        headers=_auth(optimized_client, "mira"),
    )
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _events(res.text)
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and kinds.count("delta") >= 1 and "error" not in kinds
    done = events[-1][1]
    assert done["found"] and done["citations"] and done["ttft_ms"] is not None
    streamed = "".join(p["text"] for k, p in events if k == "delta")
    assert streamed.strip().startswith("Laut den Dokumenten")


def test_stream_requires_login_and_default_mode_is_auto(optimized_client):
    q = {"question": "Wie viel Urlaub habe ich?"}
    assert optimized_client.post("/ask/stream", json=q).status_code == 401
    res = optimized_client.post("/ask", json=q, headers=_auth(optimized_client, "ole")).json()
    assert res["mode"] == "auto"


def test_api_cache_serves_repeat_but_not_across_users(optimized_client):
    q = {"question": "Welche Kündigungsfrist gilt in der Probezeit?", "mode": "classic"}
    mira, katja = _auth(optimized_client, "mira"), _auth(optimized_client, "katja")
    first = optimized_client.post("/ask", json=q, headers=mira).json()
    again = optimized_client.post("/ask", json=q, headers=mira).json()
    assert not first["cached"] and again["cached"] and again["cost"] == 0
    other = optimized_client.post("/ask", json=q, headers=katja).json()
    assert not other["cached"]
    assert not any(c["doc_id"].startswith("hr/") for c in other["citations"])
