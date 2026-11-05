"""Optimisation switches. Every lever can be turned on or off to measure its own effect.

BASELINE is exactly the system before the optimisation work. The ABLATION list turns the levers
on one after the other (each row keeps everything the rows above it switched on)."""

import os
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
    # --- colloquial-wording levers (round 4)
    embedder: str = "fastembed"  # lever 1: embedding model, "fastembed" = the small MiniLM
    rerank: bool = False  # lever 2: local cross-encoder reorders the best 20 hybrid hits
    headers: bool = False  # lever 3: department + title + heading are part of the embedded text
    doc2query: bool = False  # lever 4: generated colloquial questions are searched next to chunks
    glossary_plus: bool = False  # lever 5: glossary groups generated from the documents

    def search_options(self) -> dict:
        """Keyword arguments for HybridSearcher.search."""
        return {
            "expand": self.glossary, "graph": self.graph, "d2q": self.doc2query,
            "rerank": self.rerank,
        }  # fmt: skip


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

# Round 4: colloquial wording. Every lever alone on top of FINAL (round 2, without the rewrite,
# which needs a real model), then everything that was kept.
GEMMA = "fastembed:google/embeddinggemma-300m"
LEVERS: list[Config] = []
# Index-level choices of the served system (a different value needs a re-index):
# the embedding model (see ks/embed.py: "fastembed" = MiniLM) and contextual chunk headers.
SERVING_EMBEDDER = os.environ.get("KS_EMBEDDER", "fastembed")
SERVING_HEADERS = os.environ.get("KS_HEADERS", "0") == "1"
