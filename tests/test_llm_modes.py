"""Classic and agentic mode with FakeLLM, tool permissions, limits, eval metrics. No network."""

import pytest

import ks.llm as llm_module
from ks.agent import DOC_NOT_FOUND, MAX_TOOL_CALLS, ToolBox, answer_agentic
from ks.auth import allowed_departments, get_user
from ks.eval import compute_metrics
from ks.llm import NOT_FOUND, FakeLLM, LLMResponse, ToolCall, Usage, get_llm
from ks.rag import AskResult, Citation, answer_classic

HOTEL = "Wie viel darf ein Hotel im Vertrieb pro Nacht kosten, und wie schnell muss ich abrechnen?"
NOTHING = "Wie funktioniert die Parkplatzvergabe in der Tiefgarage?"


def test_classic_answers_with_valid_citations(searcher):
    res = answer_classic(HOTEL, get_user("katja"), searcher, FakeLLM())
    assert res.mode == "classic" and res.found
    assert res.citations and all(c.chunk_id.startswith(c.doc_id) for c in res.citations)
    assert {c.doc_id for c in res.citations} & {"vertrieb/vertrieb-reisekosten"}
    assert res.input_tokens > 0 and res.output_tokens > 0 and res.cost > 0
    assert res.steps[0]["tool"] == "search"


def test_classic_not_found_sentence(searcher):
    res = answer_classic(NOTHING, get_user("bent"), searcher, FakeLLM())
    assert res.answer == NOT_FOUND and not res.found and res.citations == []


def test_agentic_runs_search_then_read_then_answers(searcher):
    res = answer_agentic(HOTEL, get_user("katja"), searcher, FakeLLM())
    assert [s["tool"] for s in res.steps] == ["search", "read_document"]
    assert res.steps[0]["args"]["query"] == HOTEL
    assert res.found and res.citations
    assert res.input_tokens > 0


def test_agentic_not_found(searcher):
    res = answer_agentic(NOTHING, get_user("bent"), searcher, FakeLLM())
    assert res.answer == NOT_FOUND and res.citations == []
    assert [s["tool"] for s in res.steps] == ["search"]


def test_tools_hide_hr_documents_from_katja(searcher):
    box = ToolBox(searcher, allowed_departments(get_user("katja")))
    assert box.read_document("hr/gehaltsbaender") == DOC_NOT_FOUND
    # a forbidden document looks exactly like one that does not exist
    assert box.read_document("hr/gehaltsbaender") == box.read_document("hr/gibt-es-nicht")
    assert "hr/" not in box.list_documents()
    assert "hr/" not in box.search("Gehaltsbänder Entgeltgruppe E5 Jahresbrutto")
    assert not any(d.startswith("hr/") for d in box.visited)


def test_hr_user_can_read_the_same_document(searcher):
    box = ToolBox(searcher, allowed_departments(get_user("mira")))
    assert "Entgeltgruppe" in box.read_document("hr/gehaltsbaender")


class AsksForHrDocument:
    """A misbehaving model that requests a forbidden document by id and then cites it."""

    model = "scripted"
    price_in = price_out = 0.0

    def complete(self, system, user, **kwargs):
        raise NotImplementedError

    def chat(self, system, messages, tools, allow_tools=True, **kwargs):
        if len(messages) == 1:
            call = ToolCall("t1", "read_document", {"doc_id": "hr/gehaltsbaender"})
            return LLMResponse("", [call], Usage(1, 1), [])
        return LLMResponse("Band E5 [hr/gehaltsbaender#Bänder]", [], Usage(1, 1), [])


def test_agent_cannot_read_hr_doc_as_katja(searcher):
    res = answer_agentic("Gehaltsband E5?", get_user("katja"), searcher, AsksForHrDocument())
    assert res.steps[0]["summary"] == DOC_NOT_FOUND
    assert res.citations == []  # the invented citation does not match anything the tool returned
    assert res.retrieved_doc_ids == []


class NeverStops:
    model = "scripted"
    price_in = price_out = 0.0

    def __init__(self):
        self.calls = 0
        self.last_allow_tools = None

    def complete(self, system, user, **kwargs):
        raise NotImplementedError

    def chat(self, system, messages, tools, allow_tools=True, **kwargs):
        self.calls += 1
        self.last_allow_tools = allow_tools
        if not allow_tools:
            return LLMResponse("Fertig [alle/onboarding#Der erste Tag]", [], Usage(1, 1), [])
        call = ToolCall(f"t{self.calls}", "list_documents", {})
        return LLMResponse("", [call], Usage(1, 1), [])


def test_max_tool_calls_then_forced_answer(searcher):
    llm = NeverStops()
    res = answer_agentic("Was gibt es?", get_user("fiete"), searcher, llm)
    assert len(res.steps) == MAX_TOOL_CALLS == 6
    assert llm.calls == MAX_TOOL_CALLS + 1 and llm.last_allow_tools is False
    assert res.answer.startswith("Fertig")


def test_anthropic_llm_not_built_unless_provider_is_anthropic(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("AnthropicLLM must not be constructed")

    monkeypatch.setattr(llm_module, "AnthropicLLM", boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    for provider in (None, "fake"):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        if provider:
            monkeypatch.setenv("LLM_PROVIDER", provider)
        assert isinstance(get_llm(), FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError):  # provider set but no key: still no client
        get_llm()


def _result(answer, cites, retrieved, latency=10.0):
    citations = [Citation(d, "T", "S", f"{d}#0") for d in cites]
    return AskResult(answer, citations, "classic", [], 100, 10, 0.5, latency, answer != NOT_FOUND,
                     retrieved)  # fmt: skip


def test_eval_metrics_on_tiny_fixture():
    rows = [
        {"type": "single", "user": "katja", "expected_doc_ids": ["vertrieb/crm-pflege"],
         "result": _result("ok", ["vertrieb/crm-pflege"], ["vertrieb/crm-pflege"], 10)},
        {"type": "multi", "user": "katja",
         "expected_doc_ids": ["alle/reisekostenrichtlinie", "vertrieb/vertrieb-reisekosten"],
         "result": _result("x", ["alle/reisekostenrichtlinie"], ["alle/reisekostenrichtlinie"],
                          30)},
        {"type": "not_found", "user": "katja", "expected_doc_ids": [],
         "result": _result(NOT_FOUND, [], [], 20)},
        {"type": "permission", "user": "katja", "expected_doc_ids": [],
         "result": _result("Band [hr/gehaltsbaender#Bänder]", ["hr/gehaltsbaender"],
                           ["hr/gehaltsbaender"], 40)},
    ]  # fmt: skip
    m = compute_metrics(rows)
    assert m["hit_rate_at_5"] == 1.0  # both answerable questions retrieved an expected doc
    assert m["full_coverage"] == 0.5
    assert m["citation_correctness"] == 0.75  # only the permission row is wrong
    assert m["permission_leaks"] == 1
    assert m["avg_latency_ms"] == 25.0
    assert m["input_tokens"] == 400 and m["output_tokens"] == 40
    assert m["total_cost_usd"] == pytest.approx(2.0)
    assert m["by_type"]["permission"] == {"n": 1, "correct": 0}


def test_not_found_with_extra_text_is_normalised():
    from ks.llm import NOT_FOUND, FakeLLM
    from ks.rag import finish

    res = finish(
        f"{NOT_FOUND} Aber vielleicht [a/b#C].", {}, "classic", [], (1, 1), 0.0, FakeLLM(), []
    )
    assert res.answer == NOT_FOUND and not res.found and res.citations == []
