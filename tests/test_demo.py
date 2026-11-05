"""Demo export: offline (HashEmbedder + FakeLLM); checks shape and that permissions hold."""

import json

from ks.cli import main
from ks.demo import build_demo
from ks.eval import load_questions
from ks.llm import FakeLLM


def test_demo_data_respects_permissions(searcher):
    demo = build_demo(searcher, FakeLLM(), load_questions())
    assert demo["generated_by"] == "fake" and demo["generated_at"]
    assert demo["default_mode"] == "auto"
    assert {u["login"] for u in demo["users"]} == set(demo["questions"])
    for login, questions in demo["questions"].items():
        assert 6 <= len(questions) <= 8
        kinds = {q["kind"] for q in questions}
        assert "not_found" in kinds
        assert ("permission" in kinds) == (login != "fiete")
        assert all(set(q["results"]) == {"auto", "classic", "agentic"} for q in questions)
    # katja (vertrieb) never gets hr content, neither in the list nor in the document texts
    assert all(d["department"] in ("alle", "vertrieb") for d in demo["documents"]["katja"])
    assert all(not doc_id.startswith("hr/") for doc_id in demo["contents"]["katja"])
    assert "hr/gehaltsbaender" not in json.dumps(demo["questions"]["katja"])


def test_cli_export_demo(tmp_path, monkeypatch):
    out = tmp_path / "demo.json"
    monkeypatch.setenv("KS_EMBEDDER", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    main(["--index-dir", str(tmp_path / "none"), "export-demo", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["generated_by"] == "fake"
