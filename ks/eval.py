"""Evaluation: run eval/questions.yaml in classic and/or agentic mode and compute metrics."""

import json
from pathlib import Path

import numpy as np
import yaml

from ks.auth import allowed_departments, get_user
from ks.cache import SemanticCache
from ks.config import BASELINE, Config
from ks.embed import HashEmbedder, get_embedder
from ks.ingest import chunk_documents, embedding_text, load_documents
from ks.llm import LABEL_RE, LLM, NOT_FOUND
from ks.pipeline import answer
from ks.rag import AskResult
from ks.search import HybridSearcher, build_searcher
from ks.store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS = ROOT / "eval" / "results.json"
HOLDOUT = ROOT / "eval" / "holdout.yaml"
HOLDOUT2 = ROOT / "eval" / "holdout2.yaml"
QUESTION_SETS = {"tuning": QUESTIONS, "holdout": HOLDOUT, "holdout2": HOLDOUT2}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def build_variant(
    embedder_name: str, headers: bool, docs_dir: Path | None = None
) -> HybridSearcher:
    """In-memory searcher for one index variant (embedding model x contextual headers). Vectors
    come from the disk cache of the embedder, so a variant is only slow the first time."""
    embedder = get_embedder(embedder_name)
    chunks = chunk_documents(load_documents(docs_dir or ROOT / "data" / "docs"))
    store = MemoryStore()
    store.add(chunks, embedder.embed([embedding_text(c, headers) for c in chunks]))
    searcher = build_searcher(store, embedder, warm=True, headers=headers)
    getattr(embedder, "flush", lambda: None)()
    return searcher


class VariantPool:
    """Gives every configuration the searcher it needs. The opened (saved) index serves the rows
    that use its embedder and header setting; other variants are built in memory once. The
    cross-encoder is loaded once and only used by rows with rerank=True."""

    def __init__(self, base: HybridSearcher, embedder_name: str, headers: bool) -> None:
        self.base = base
        self.base_key = (embedder_name, headers)
        self._built: dict[tuple[str, bool], HybridSearcher] = {}
        self._reranker = None

    def get(self, cfg: Config) -> HybridSearcher:
        key = (cfg.embedder, cfg.headers)
        if key == self.base_key:
            searcher = self.base
        else:
            if key not in self._built:
                self._built[key] = build_variant(*key)
            searcher = self._built[key]
        if cfg.rerank and searcher.reranker is None:
            if self._reranker is None:
                from ks.rerank import CrossEncoderReranker

                self._reranker = CrossEncoderReranker()
            searcher.reranker = self._reranker
        return searcher


def open_searcher(index_dir: Path, docs_dir: Path) -> tuple[HybridSearcher, str]:
    """Use the saved index (FastEmbed) if there is one; otherwise build a HashEmbedder index."""
    if (index_dir / "meta.json").exists():
        try:
            name = MemoryStore.saved_embedder_name(index_dir)
            store = MemoryStore.load(index_dir)
            return build_searcher(store, get_embedder(name), index_dir, warm=True), name
        except Exception:  # model not downloadable offline etc.: fall through to hash
            pass
    embedder = HashEmbedder()
    chunks = chunk_documents(load_documents(docs_dir))
    store = MemoryStore()
    store.add(chunks, embedder.embed([embedding_text(c) for c in chunks]))
    return build_searcher(store, embedder, warm=True), embedder.name


def open_pool(index_dir: Path, docs_dir: Path) -> VariantPool:
    """The saved index as the base searcher plus on-demand in-memory variants (see VariantPool)."""
    searcher, name = open_searcher(index_dir, docs_dir)
    return VariantPool(searcher, name, searcher.headers)


def _leaks(row: dict) -> bool:
    """True if the user got a citation / retrieved a document outside their departments."""
    allowed = allowed_departments(get_user(row["user"]))
    res: AskResult = row["result"]
    doc_ids = {c.doc_id for c in res.citations} | set(res.retrieved_doc_ids)
    doc_ids |= {label.split("#")[0] for label in LABEL_RE.findall(res.answer)}
    return any(d.split("/")[0] not in allowed for d in doc_ids)


def _correct(row: dict) -> bool:
    res: AskResult = row["result"]
    expected = set(row["expected_doc_ids"])
    if not expected:  # not-found and permission questions
        return res.answer.strip() == NOT_FOUND and not res.citations
    return bool(expected & {c.doc_id for c in res.citations})


def compute_metrics(rows: list[dict]) -> dict:
    """rows: question dict + 'result' (AskResult). Returns the metrics of one mode."""
    answerable = [r for r in rows if r["expected_doc_ids"]]
    latencies = [r["result"].latency_ms for r in rows]
    hits = [
        bool(set(r["expected_doc_ids"]) & set(r["result"].retrieved_doc_ids)) for r in answerable
    ]
    full = [set(r["expected_doc_ids"]) <= set(r["result"].retrieved_doc_ids) for r in answerable]
    per_type: dict[str, dict] = {}
    for r in rows:
        t = per_type.setdefault(r["type"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += _correct(r)
    return {
        "questions": len(rows),
        "hit_rate_at_5": sum(hits) / len(hits) if hits else 0.0,
        "full_coverage": sum(full) / len(full) if full else 0.0,
        "citation_correctness": sum(_correct(r) for r in rows) / len(rows) if rows else 0.0,
        "permission_leaks": sum(_leaks(r) for r in rows),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "input_tokens": sum(r["result"].input_tokens for r in rows),
        "output_tokens": sum(r["result"].output_tokens for r in rows),
        "total_cost_usd": sum(r["result"].cost for r in rows),
        "by_type": per_type,
    }


def run_mode(
    mode: str, questions: list[dict], searcher: HybridSearcher, llm: LLM,
    cfg: Config = BASELINE, cache: SemanticCache | None = None,
) -> list[dict]:  # fmt: skip
    rows = []
    for q in questions:
        result = answer(q["question"], get_user(q["user"]), searcher, llm, mode, cfg, cache)
        rows.append({**q, "result": result})
    return rows


def run_eval(
    modes: list[str], questions: list[dict], searcher: HybridSearcher, llm: LLM, embedder: str,
    out: Path = RESULTS, cfg: Config = BASELINE,
) -> dict:  # fmt: skip
    report: dict = {"llm": llm.model, "embedder": embedder, "config": cfg.name, "modes": {}}
    for mode in modes:
        rows = run_mode(mode, questions, searcher, llm, cfg)
        report["modes"][mode] = {
            "metrics": compute_metrics(rows),
            "questions": [
                {k: v for k, v in r.items() if k != "result"} | {"result": r["result"].to_dict(),
                 "correct": _correct(r), "leak": _leaks(r)}
                for r in rows
            ],
        }  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_table(report: dict) -> str:
    rows = [
        ("hit rate@5", lambda m: f"{m['hit_rate_at_5']:.0%}"),
        ("both docs retrieved", lambda m: f"{m['full_coverage']:.0%}"),
        ("citation correct", lambda m: f"{m['citation_correctness']:.0%}"),
        ("permission leaks", lambda m: str(m["permission_leaks"])),
        ("latency avg ms", lambda m: f"{m['avg_latency_ms']:.0f}"),
        ("latency p95 ms", lambda m: f"{m['p95_latency_ms']:.0f}"),
        ("tokens in/out", lambda m: f"{m['input_tokens']}/{m['output_tokens']}"),
        ("total cost USD", lambda m: f"{m['total_cost_usd']:.4f}"),
    ]
    modes = list(report["modes"])
    lines = [f"LLM: {report['llm']} | embedder: {report['embedder']}"]
    lines.append(f"{'metric':<22}" + "".join(f"{m:>14}" for m in modes))
    for name, fmt in rows:
        cells = "".join(f"{fmt(report['modes'][m]['metrics']):>14}" for m in modes)
        lines.append(f"{name:<22}{cells}")
    return "\n".join(lines)
