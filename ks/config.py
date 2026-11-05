"""Optimisation switches. Every lever can be turned on or off to measure its own effect.

BASELINE is exactly the system before the optimisation work. The ABLATION list turns the levers
on one after the other (each row keeps everything the rows above it switched on)."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    compress: bool = False  # lever 3: best sentences only, short citation keys, short prompts
    glossary: bool = False  # lever 2: synonyms / abbreviations widen the keyword query
    graph: bool = False  # lever 1: pull in the best chunk of strongly linked documents
    router: bool = False  # lever 4: mode "auto" + a leaner agent
    cache: bool = False  # lever 5: semantic answer cache (never across permission scopes)
    rewrite: bool = (
        False  # lever 6: weak search result -> the model turns the question into keywords
    )
    mode: str = "classic"  # which answer mode this row uses: classic | agentic | auto


BASELINE = Config()
COMPRESSION = replace(BASELINE, name="+compression", compress=True)
GLOSSARY = replace(COMPRESSION, name="+glossary", glossary=True)
GRAPH = replace(GLOSSARY, name="+graph", graph=True)
AUTO = replace(GRAPH, name="+router (auto)", router=True, mode="auto")
CACHED = replace(AUTO, name="+cache", cache=True)

ABLATION = [BASELINE, COMPRESSION, GLOSSARY, GRAPH, AUTO, CACHED]
BASELINE_AGENTIC = replace(BASELINE, name="baseline (agentic)", mode="agentic")
LEAN_AGENTIC = replace(AUTO, name="agentic (all levers)", mode="agentic")  # lean agent on its own
# Round 2: classic by default (agent only for clearly multi-document questions), facts kept in the
# compressed context, a more complete prompt, a lean agent that compresses less.
FINAL = replace(AUTO, name="final (round 2)")
# Round 3: when the first search is weak, a tiny model call turns the colloquial question into
# search keywords and the search runs a second time (see ks/rewrite.py).
FINAL_REWRITE = replace(FINAL, name="final + rewrite", rewrite=True)
# What the API and the demo run by default: everything on, including the semantic cache.
SERVING = replace(FINAL_REWRITE, name="serving", cache=True)
