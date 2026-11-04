import pytest

from ks.auth import USERS, allowed_departments, get_user
from ks.embed import HashEmbedder
from ks.store import MemoryStore

# (question, user, expected doc_id within the top 3). Keyword-style questions: with the toy
# HashEmbedder the BM25 part carries the ranking.
QUESTIONS = [
    ("Wie viele Urlaubstage habe ich Sonderurlaub Hochzeit", "mira", "alle/urlaubsantrag"),
    ("VPN Elbtunnel Client einrichten Server", "ole", "it/vpn-einrichtung"),
    ("Passwortrichtlinie Passwortmanager Mindestlänge Zeichen", "ole", "it/passwortrichtlinie"),
    (
        "Rabattstaffel Großkunden Jahresumsatz Prozent",
        "katja",
        "vertrieb/rabattstaffel-grosskunden",
    ),
    ("Gehaltsband Entgeltgruppe E5 Teamleitung", "mira", "hr/gehaltsbaender"),
    ("Kündigungsfrist nach Betriebszugehörigkeit Monate", "mira", "hr/kuendigungsfristen"),
    ("Gefahrgut Lagerung Halle 4 ADR", "bent", "logistik/gefahrgut-handling"),
    ("Zollanmeldung ATLAS Handelsrechnung Zolltarifnummer", "bent", "logistik/zollabwicklung"),
    (
        "Hotel Übernachtung Obergrenze Euro pro Nacht Reisekosten",
        "mira",
        "alle/reisekostenrichtlinie",
    ),
    ("Angebot nachfassen Marge Vier-Augen-Prinzip", "katja", "vertrieb/angebotsprozess"),
]


@pytest.mark.parametrize(("question", "user", "expected"), QUESTIONS)
def test_search_finds_expected_document(searcher, question, user, expected):
    hits = searcher.search(question, allowed_departments(get_user(user)), k=3)
    assert expected in [h.chunk.doc_id for h in hits], [h.chunk.chunk_id for h in hits]


def test_vertrieb_user_never_gets_hr_chunks(searcher, store, embedder):
    allowed = allowed_departments(get_user("katja"))
    assert allowed == {"alle", "vertrieb"}
    hr_queries = [
        "Gehaltsband Entgeltgruppe E5 Jahresbrutto",
        "Kündigungsfrist Betriebszugehörigkeit",
        "Elternzeit Zuschuss",
        "Bewerbungsprozess Stellenfreigabe",
        "Probezeit Feedbackgespräche",
    ]
    for q in hr_queries:
        hits = searcher.search(q, allowed, k=20, candidates=500)
        assert hits
        assert all(h.chunk.department in allowed for h in hits), q
        assert not any(h.chunk.doc_id.startswith("hr/") for h in hits)
    # same on the store level: even asking for every chunk never returns hr
    qvec = embedder.embed(["Gehaltsbänder"])[0]
    assert all(s.chunk.department != "hr" for s in store.search_vector(qvec, 10_000, allowed))
    assert all(c.department != "hr" for c in store.get_chunks(allowed))
    assert store.get_document_chunks("hr/gehaltsbaender", allowed) == []


def test_filter_is_applied_before_ranking(searcher):
    """If filtering happened after ranking, top-k of an hr-heavy query would shrink below k."""
    hits = searcher.search("Gehaltsband Entgeltgruppe Gehalt", {"alle", "vertrieb"}, k=5)
    assert len(hits) == 5


def test_empty_allowed_set_returns_nothing(searcher):
    assert searcher.search("Urlaub", set(), k=5) == []


def test_hr_user_sees_hr(searcher):
    hits = searcher.search(
        "Gehaltsbänder Entgeltgruppe Jahresbrutto", allowed_departments(get_user("mira")), k=3
    )
    assert "hr/gehaltsbaender" in [h.chunk.doc_id for h in hits]


def test_every_user_sees_alle_only_plus_own():
    for user in USERS.values():
        assert allowed_departments(user) == {"alle", *user.departments}
    assert "hr" not in allowed_departments(get_user("bent"))


def test_memory_store_roundtrip(tmp_path, store):
    store.save(tmp_path, "hash")
    loaded = MemoryStore.load(tmp_path)
    assert loaded.count() == store.count()
    assert MemoryStore.saved_embedder_name(tmp_path) == "hash"
    q = HashEmbedder().embed(["Urlaub"])[0]
    a = store.search_vector(q, 3, {"alle"})
    b = loaded.search_vector(q, 3, {"alle"})
    assert [x.chunk.chunk_id for x in a] == [x.chunk.chunk_id for x in b]
