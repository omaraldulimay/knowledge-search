"""API tests with TestClient: offline, HashEmbedder, FakeLLM, sample docs copied to a temp dir."""

import shutil

import pytest
from fastapi.testclient import TestClient

from ks.api import NOT_FOUND_DETAIL, create_app
from ks.embed import HashEmbedder
from ks.llm import NOT_FOUND, FakeLLM

HOTEL = "Wie viel darf ein Hotel im Vertrieb pro Nacht kosten, und wie schnell muss ich abrechnen?"
NOTHING = "Wie funktioniert die Parkplatzvergabe in der Tiefgarage?"
NEW_DOC = (
    "# Dachterrassen-Reservierung\n\n## Ablauf\n\n"
    "Die Dachterrasse am Elbkai wird über das Formular DT-77 reserviert. "
    "Die Reservierung muss mindestens drei Werktage vorher eingehen.\n"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.conftest import DOCS_DIR

    monkeypatch.setenv("KS_METRICS_DB", ":memory:")
    monkeypatch.setenv("KS_CONFIG", "baseline")  # these tests cover the unoptimised behaviour
    docs = tmp_path / "docs"
    shutil.copytree(DOCS_DIR, docs)
    app = create_app(
        embedder=HashEmbedder(), llm=FakeLLM(), docs_dir=docs, index_dir=tmp_path / "idx"
    )
    with TestClient(app) as c:
        yield c


def auth(client, user):
    token = client.post("/login", json={"username": user}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_is_public(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0


def test_login(client):
    res = client.post("/login", json={"username": "Katja"})
    assert res.status_code == 200
    body = res.json()
    assert body["login"] == "katja" and body["departments"] == ["alle", "vertrieb"]
    assert not body["is_admin"] and body["token"]
    assert client.post("/login", json={"username": "fiete"}).json()["is_admin"]
    assert client.post("/login", json={"username": "nobody"}).status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/ask"),
        ("get", "/documents"),
        ("get", "/documents/alle/onboarding"),
        ("post", "/documents"),
        ("delete", "/documents/alle/onboarding"),
        ("get", "/metrics"),
    ],
)
def test_endpoints_require_a_valid_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert getattr(client, method)(path, headers=bad).status_code == 401


def test_token_signed_with_other_secret_is_rejected(client):
    from itsdangerous import URLSafeTimedSerializer

    forged = URLSafeTimedSerializer("another-secret").dumps({"u": "fiete"})
    res = client.get("/documents", headers={"Authorization": f"Bearer {forged}"})
    assert res.status_code == 401


@pytest.mark.parametrize("mode", ["classic", "agentic"])
def test_ask_both_modes(client, mode):
    res = client.post("/ask", json={"question": HOTEL, "mode": mode}, headers=auth(client, "katja"))
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == mode and body["found"]
    assert body["citations"]
    assert any(c["doc_id"] == "vertrieb/vertrieb-reisekosten" for c in body["citations"])


def test_employees_get_no_technical_details_but_the_admin_does(client):
    from ks.api import ADMIN_ONLY_FIELDS

    employee = client.post("/ask", json={"question": HOTEL}, headers=auth(client, "katja")).json()
    assert employee["answer"] and employee["citations"]
    assert not set(ADMIN_ONLY_FIELDS) & set(employee)  # no tokens, cost, latency, steps
    admin = client.post("/ask", json={"question": HOTEL}, headers=auth(client, "fiete")).json()
    assert set(ADMIN_ONLY_FIELDS) <= set(admin) and admin["steps"]
    stream = client.post("/ask/stream", json={"question": HOTEL}, headers=auth(client, "katja"))
    done = [b for b in stream.text.split("\n\n") if b.startswith("event: done")][0]
    assert "cost" not in done and "latency_ms" not in done
    # the request log still records everything for the quality dashboard
    assert client.get("/metrics", headers=auth(client, "fiete")).json()["total_requests"] >= 3


def test_ask_not_found_and_validation(client):
    h = auth(client, "bent")
    body = client.post("/ask", json={"question": NOTHING, "mode": "classic"}, headers=h).json()
    assert body["answer"] == NOT_FOUND and not body["found"]
    assert (
        client.post("/ask", json={"question": "x", "mode": "magic"}, headers=h).status_code == 422
    )
    assert client.post("/ask", json={"question": ""}, headers=h).status_code == 422


@pytest.mark.parametrize("mode", ["classic", "agentic"])
def test_ask_never_leaks_hr(client, mode):
    res = client.post(
        "/ask", json={"question": "Wie hoch ist das Gehaltsband E5?", "mode": mode},
        headers=auth(client, "katja"),
    )  # fmt: skip
    body = res.json()
    assert all(not c["doc_id"].startswith("hr/") for c in body["citations"])
    assert all(not d.startswith("hr/") for d in body.get("retrieved_doc_ids", []))


def test_documents_list_is_filtered(client):
    katja = client.get("/documents", headers=auth(client, "katja")).json()
    assert {d["department"] for d in katja} == {"alle", "vertrieb"}
    assert not any(d["doc_id"].startswith("hr/") for d in katja)
    fiete = client.get("/documents", headers=auth(client, "fiete")).json()
    assert any(d["doc_id"] == "hr/gehaltsbaender" for d in fiete)
    assert len(fiete) == 35


def test_get_document_forbidden_equals_missing(client):
    h = auth(client, "katja")
    ok = client.get("/documents/vertrieb/provisionsmodell", headers=h)
    assert ok.status_code == 200 and ok.json()["chunks"]
    forbidden = client.get("/documents/hr/gehaltsbaender", headers=h)
    missing = client.get("/documents/hr/does-not-exist", headers=h)
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json() == {"detail": NOT_FOUND_DETAIL}


def test_non_admin_cannot_upload_or_delete(client):
    h = auth(client, "katja")
    files = {"file": ("neu.md", NEW_DOC.encode(), "text/markdown")}
    assert client.post("/documents", files=files, headers=h).status_code == 403
    # same answer for existing, forbidden and missing documents: nothing is revealed
    for doc_id in ("vertrieb/provisionsmodell", "hr/gehaltsbaender", "hr/missing"):
        assert client.delete(f"/documents/{doc_id}", headers=h).status_code == 403
    fiete = auth(client, "fiete")
    assert client.get("/documents/hr/gehaltsbaender", headers=fiete).status_code == 200


def test_admin_delete_removes_document_everywhere(client):
    admin = auth(client, "fiete")
    assert client.delete("/documents/hr/gehaltsbaender", headers=admin).json() == {
        "deleted": "hr/gehaltsbaender"
    }
    assert client.get("/documents/hr/gehaltsbaender", headers=admin).status_code == 404
    assert client.delete("/documents/hr/gehaltsbaender", headers=admin).status_code == 404
    ids = [d["doc_id"] for d in client.get("/documents", headers=admin).json()]
    assert "hr/gehaltsbaender" not in ids and len(ids) == 34


def test_upload_then_ask_finds_new_document(client):
    question = "Wie reserviere ich die Dachterrasse am Elbkai mit Formular DT-77?"
    mira = auth(client, "mira")
    before = client.post(
        "/ask", json={"question": question, "mode": "classic"}, headers=mira
    ).json()
    assert before["answer"] == NOT_FOUND

    admin = auth(client, "fiete")
    files = {"file": ("Dachterrasse Reservierung.md", NEW_DOC.encode(), "text/markdown")}
    res = client.post("/documents", files=files, data={"department": "hr"}, headers=admin)
    assert res.status_code == 201, res.text
    assert res.json()["doc_id"] == "hr/dachterrasse-reservierung"
    assert res.json()["chunks_embedded"] >= 1

    for mode in ("classic", "agentic"):
        found = client.post("/ask", json={"question": question, "mode": mode}, headers=mira).json()
        assert found["found"], mode
        assert any(c["doc_id"] == "hr/dachterrasse-reservierung" for c in found["citations"])
    # hr-only document stays invisible for other departments
    katja = auth(client, "katja")
    hidden = client.post("/ask", json={"question": question}, headers=katja).json()
    # (FakeLLM quotes whatever it is given, so the contract is: the hr document never shows up)
    assert "Dachterrasse" not in hidden["answer"]
    assert all(c["doc_id"] != "hr/dachterrasse-reservierung" for c in hidden["citations"])
    assert client.get("/documents/hr/dachterrasse-reservierung", headers=katja).status_code == 404


def test_upload_validation(client):
    admin = auth(client, "fiete")
    bad_type = {"file": ("evil.exe", b"x", "application/octet-stream")}
    assert client.post("/documents", files=bad_type, headers=admin).status_code == 422
    bad_pdf = {"file": ("kaputt.pdf", b"not a pdf", "application/pdf")}
    assert client.post("/documents", files=bad_pdf, headers=admin).status_code == 422
    ok = {"file": ("x.md", b"# X\n\ntext", "text/markdown")}
    assert (
        client.post("/documents", files=ok, data={"department": "nope"}, headers=admin).status_code
        == 422
    )
    # path traversal in the file name is reduced to a plain, safe name
    sneaky = {"file": ("../../etc/passwort.md", b"# P\n\ntext text", "text/markdown")}
    res = client.post("/documents", files=sneaky, data={"department": "it"}, headers=admin)
    assert res.json()["doc_id"] == "it/passwort"


def test_metrics_aggregation(client):
    katja, admin = auth(client, "katja"), auth(client, "fiete")
    assert client.get("/metrics", headers=katja).status_code == 403
    empty = client.get("/metrics", headers=admin).json()
    assert empty["total_requests"] == 0 and empty["modes"] == {}

    client.post("/ask", json={"question": HOTEL, "mode": "classic"}, headers=katja)
    client.post("/ask", json={"question": HOTEL, "mode": "agentic"}, headers=katja)
    for _ in range(2):
        client.post("/ask", json={"question": NOTHING, "mode": "classic"}, headers=katja)
    client.post("/ask", json={"question": NOTHING.upper(), "mode": "agentic"}, headers=katja)

    body = client.get("/metrics", headers=admin).json()
    assert body["total_requests"] == 5
    classic, agentic = body["modes"]["classic"], body["modes"]["agentic"]
    assert classic["count"] == 3 and agentic["count"] == 2
    assert classic["not_found_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert agentic["not_found_rate"] == 0.5
    assert agentic["input_tokens"] > 0 and agentic["cost"] > 0
    assert classic["avg_latency_ms"] > 0 and classic["p95_latency_ms"] > 0
    top = body["top_not_found"][0]
    assert top["question"] == NOTHING.lower() and top["count"] == 3
    assert "eval" in body and "scale" in body  # latest result files (or null)
