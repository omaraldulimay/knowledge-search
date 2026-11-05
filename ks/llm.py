"""LLM layer: one small protocol, a deterministic FakeLLM and the real AnthropicLLM.

Messages use the Anthropic shape (role + str or list of content blocks), so the agent loop
works the same for both implementations. Every call returns token usage.
The real API is only used when LLM_PROVIDER=anthropic AND ANTHROPIC_API_KEY is set.
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

NOT_FOUND = "Dazu habe ich in den Dokumenten nichts gefunden."
# A citation looks like [hr/kuendigungsfristen#Probezeit] = [doc_id#section heading]
LABEL_RE = re.compile(r"\[([a-z0-9_-]+/[a-z0-9_-]+#[^\]\n]+)\]")
# Short citation keys of the compressed prompts: [1], [2] or [1, 3]; mapped back to chunks in code.
KEY_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
DOC_ID_RE = re.compile(r"\b([a-z0-9_-]+/[a-z0-9_-]+)\b")
TextCallback = Callable[[str], None]  # called with every piece of streamed answer text

DEFAULT_MODEL = "claude-haiku-5-5"
# USD per million tokens, from the claude-api skill price table (Haiku 5.5, prompts <= 100K tokens)
DEFAULT_PRICE_IN = 0.10
DEFAULT_PRICE_OUT = 0.50


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    content: list[dict] = field(default_factory=list)  # assistant blocks, to append to messages


class LLM(Protocol):
    model: str
    price_in: float  # USD per million input tokens
    price_out: float

    def complete(
        self, system: str, user: str, max_tokens: int | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        """Plain completion: one user prompt, no tools. `on_text` receives streamed text."""
        ...

    def chat(
        self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True,
        max_tokens: int | None = None, on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        """One step of a tool-use conversation. allow_tools=False forces a text answer."""
        ...


def _prices() -> tuple[float, float]:
    return (
        float(os.environ.get("LLM_PRICE_IN", DEFAULT_PRICE_IN)),
        float(os.environ.get("LLM_PRICE_OUT", DEFAULT_PRICE_OUT)),
    )


def cost_usd(llm: LLM, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * llm.price_in + output_tokens * llm.price_out) / 1_000_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb, only used by the fake


def _sentences(text: str, limit: int = 2) -> str:
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:limit])[:350]


def _labelled_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into (label, body) pairs; a label alone on a line starts a block.
    Search result lines ("[label] snippet") are handled as one-line blocks."""
    blocks: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = re.match(r"^\[([a-z0-9_-]+/[a-z0-9_-]+#[^\]\n]+|\d+)\]\s*(.*)$", line)
        if m:
            blocks.append((m.group(1), m.group(2)))
        elif blocks and line.strip() and not line.startswith("Frage:"):
            label, body = blocks[-1]
            blocks[-1] = (label, f"{body} {line.strip()}".strip())
    return blocks


class FakeLLM:
    """Deterministic stand-in: no network, no randomness. It does not understand the question,
    it just quotes the best chunks and cites them, so metrics measure retrieval and plumbing."""

    model = "fake"

    def __init__(self) -> None:
        self.price_in, self.price_out = _prices()

    def _compose(self, blocks: list[tuple[str, str]]) -> str:
        seen: set[str] = set()
        lines = []
        for label, body in blocks:
            if label in seen or not body:
                continue
            seen.add(label)
            lines.append(f"- {_sentences(body)} [{label}]")
            if len(lines) == 3:
                break
        return "Laut den Dokumenten:\n" + "\n".join(lines) if lines else NOT_FOUND

    def _response(
        self, text: str, calls: list[ToolCall], prompt: str, on_text: TextCallback | None = None
    ) -> LLMResponse:
        if on_text:
            for word in text.split(" "):
                on_text(word + " ")
        content: list[dict] = [{"type": "text", "text": text}] if text else []
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in calls
        ]
        out = text + "".join(json.dumps(c.args) for c in calls)
        usage = Usage(_estimate_tokens(prompt), _estimate_tokens(out))
        return LLMResponse(text, calls, usage, content)

    def complete(
        self, system: str, user: str, max_tokens: int | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        text = self._compose(_labelled_blocks(user))
        return self._response(text, [], system + user, on_text)

    def chat(
        self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True,
        max_tokens: int | None = None, on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        prompt = system + json.dumps(messages, ensure_ascii=False)
        question = messages[0]["content"]
        results = [
            block["content"]
            for m in messages
            if isinstance(m["content"], list)
            for block in m["content"]
            if block.get("type") == "tool_result"
        ]
        # script: search(question) -> read_document(top doc) -> answer
        if not results and allow_tools:
            call = ToolCall("fake_1", "search", {"query": question, "k": 5})
            return self._response("", [call], prompt)
        search_result = results[0] if results else ""
        blocks = _labelled_blocks(search_result)
        if len(results) == 1 and allow_tools and blocks:
            found = DOC_ID_RE.search(blocks[0][0] + " " + blocks[0][1])
            doc_id = found.group(1) if found else blocks[0][0].split("#")[0]
            call = ToolCall("fake_2", "read_document", {"doc_id": doc_id})
            return self._response("", [call], prompt)
        return self._response(self._compose(blocks), [], prompt, on_text)


class AnthropicLLM:
    """Claude via the official `anthropic` SDK (optional extra: pip install -e ".[anthropic]")."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError('anthropic missing: pip install -e ".[anthropic]"') from exc
        self.model = model
        self.price_in, self.price_out = _prices()
        self._client = anthropic.Anthropic(api_key=api_key)  # key from ANTHROPIC_API_KEY

    def complete(
        self, system: str, user: str, max_tokens: int | None = None,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        messages = [{"role": "user", "content": user}]
        return self.chat(system, messages, [], False, max_tokens, on_text)

    def chat(
        self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True,
        max_tokens: int | None = None, on_text: TextCallback | None = None,
    ) -> LLMResponse:  # fmt: skip
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            if not allow_tools:
                kwargs["tool_choice"] = {"type": "none"}  # tools stay declared for the history
        params: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens or 2048,
            system=system,
            messages=messages,
            output_config={"effort": "low"},  # short factual answers; no sampling params on 5.5
            **kwargs,
        )
        if on_text:  # streaming: text arrives piece by piece, the final message has the usage
            with self._client.messages.stream(**params) as stream:
                for piece in stream.text_stream:
                    on_text(piece)
                response = stream.get_final_message()
        else:
            response = self._client.messages.create(**params)
        # keep the raw blocks (incl. thinking blocks) so they can be sent back unchanged
        content = [b.model_dump(exclude_none=True) for b in response.content]
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"
        ]
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        return LLMResponse(text, calls, usage, content)


def get_llm(provider: str | None = None) -> LLM:
    """Default is the fake. The real API needs LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "fake")).lower()
    if provider == "fake":
        return FakeLLM()
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        return AnthropicLLM(os.environ.get("LLM_MODEL", DEFAULT_MODEL), api_key=key)
    raise ValueError(f"unknown LLM provider: {provider}")
