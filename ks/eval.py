"""Evaluation: run eval/questions.yaml in classic and/or agentic mode and compute metrics."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from ks.agent import answer_agentic
from ks.auth import allowed_departments, get_user
from ks.embed import HashEmbedder, get_embedder
from ks.ingest import chunk_documents, embedding_text, load_documents
from ks.llm import LABEL_RE, LLM, NOT_FOUND
from ks.rag import AskResult, answer_classic
from ks.search import HybridSearcher
from ks.store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS = ROOT / "eval" / "results.json"
MODES: dict[str, Callable] = {"classic": answer_classic, "agentic": answer_agentic}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def open_searcher(index_dir: Path, docs_dir: Path) -> tuple[HybridSearcher, str]:
    """Use the saved index (FastEmbed) if there is one; otherwise build a HashEmbedder index."""
    if (index_dir / "meta.json").exists():
        try:
            name = MemoryStore.saved_embedder_name(index_dir)
            return HybridSearcher(MemoryStore.load(index_dir), get_embedder(name)), name
        except Exception:  # model not downloadable offline etc.: fall through to hash
            pass
    embedder = HashEmbedder()
    chunks = chunk_documents(load_documents(docs_dir))
    store = MemoryStore()
    store.add(chunks, embedder.embed([embedding_text(c) for c in chunks]))
    return HybridSearcher(store, embedder), embedder.name


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


def run_mode(mode: str, questions: list[dict], searcher: HybridSearcher, llm: LLM) -> list[dict]:
    rows = []
    for q in questions:
        result = MODES[mode](q["question"], get_user(q["user"]), searcher, llm)
        rows.append({**q, "result": result})
    return rows


def run_eval(
    modes: list[str], questions: list[dict], searcher: HybridSearcher, llm: LLM, embedder: str,
    out: Path = RESULTS,
) -> dict:  # fmt: skip
    report: dict = {"llm": llm.model, "embedder": embedder, "modes": {}}
    for mode in modes:
        rows = run_mode(mode, questions, searcher, llm)
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
