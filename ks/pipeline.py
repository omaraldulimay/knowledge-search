"""One entry point for answering: picks the mode, uses the semantic cache if the config has it."""

import time

from ks.agent import answer_agentic
from ks.auth import User, allowed_departments
from ks.cache import SemanticCache
from ks.config import BASELINE, Config
from ks.llm import LLM, TextCallback
from ks.rag import AskResult, answer_classic
from ks.router import answer_auto
from ks.search import HybridSearcher

MODES = ("classic", "agentic", "auto")


def answer(
    question: str, user: User, searcher: HybridSearcher, llm: LLM, mode: str = "classic",
    cfg: Config = BASELINE, cache: SemanticCache | None = None,
    on_text: TextCallback | None = None,
) -> AskResult:  # fmt: skip
    started = time.perf_counter()
    allowed = allowed_departments(user)
    use_cache = cfg.cache and cache is not None
    if use_cache:
        qvec = searcher.embed_query(question)
        cached = cache.lookup(qvec, allowed, mode)  # same question vector AND same department set
        if cached:
            hit = cached[0]
            hit.latency_ms = hit.ttft_ms = (time.perf_counter() - started) * 1000  # incl. embedding
            if on_text:
                on_text(hit.answer)
            return hit
    if mode == "auto":
        result = answer_auto(question, user, searcher, llm, cfg, on_text)
    elif mode == "agentic":
        result = answer_agentic(question, user, searcher, llm, cfg=cfg, on_text=on_text)
    else:
        result = answer_classic(question, user, searcher, llm, cfg=cfg, on_text=on_text)
    if use_cache:
        cache.put(qvec, allowed, result, mode)
    return result
