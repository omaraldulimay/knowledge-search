"""Ablation study: run every configuration on the tuning questions and on the held-out questions.

With FakeLLM no API call is made, so hit rate, coverage, leaks, routing and cache numbers are free.
The same code runs with the real model for the latency / token / cost columns (`--real`)."""

import json
import random
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from ks.auth import get_user
from ks.cache import SemanticCache
from ks.config import ABLATION, BASELINE_AGENTIC, FINAL, LEAN_AGENTIC, Config
from ks.eval import QUESTION_SETS, ROOT, _correct, _leaks, load_questions
from ks.llm import LLM
from ks.pipeline import answer
from ks.search import HybridSearcher

OPTIMIZATION_JSON = ROOT / "eval" / "optimization.json"
FACTS = ROOT / "eval" / "facts.yaml"
PARAPHRASE_SHARE = 0.30  # share of the workload that repeats an earlier question
# Templates, not an LLM: the way people re-ask the same thing.
TEMPLATES = [
    "{q}",  # same question again
    "Kurze Frage: {q}",
    "{q} Danke!",
    "Hallo, {q}",
    "Hey, {q} Wäre nett, wenn du helfen kannst.",
    "{q_plain}",  # lower case, no question mark
]


def _norm(text: str) -> str:
    text = text.lower().replace("%", " prozent")
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)  # 1.200 -> 1200
    return re.sub(r"\s+", " ", text)


def facts_ok(answer_text: str, facts: list[str]) -> bool:
    """Every fact (a list of alternatives joined by '|') must appear in the answer."""
    text = _norm(answer_text)
    return all(any(alt.strip() in text for alt in fact.split("|")) for fact in facts)


def paraphrase(question: str, index: int) -> str:
    plain = question.rstrip("?! .").lower()
    return TEMPLATES[index % len(TEMPLATES)].format(q=question, q_plain=plain)


def build_workload(questions: list[dict], seed: int = 7) -> list[dict]:
    """The questions in order, with ~30 % repeats (paraphrases of earlier ones, same user)
    mixed in after the original has been asked. Each item has `repeat_of` (id or None)."""
    rng = random.Random(seed)
    n_repeat = round(len(questions) * PARAPHRASE_SHARE / (1 - PARAPHRASE_SHARE))
    positions = sorted(rng.sample(range(2, len(questions) + 1), n_repeat))
    work, asked, k = [], [], 0
    for i, q in enumerate(questions, start=1):
        work.append({**q, "repeat_of": None})
        asked.append(q)
        while k < len(positions) and positions[k] == i:
            src = rng.choice(asked[:-1] or asked)
            work.append({**src, "question": paraphrase(src["question"], k), "repeat_of": src["id"]})
            k += 1
    return work


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def metrics(rows: list[dict], facts: dict[str, list[str]]) -> dict:
    answerable = [r for r in rows if r["expected_doc_ids"]]
    multi = [r for r in answerable if r["type"] == "multi"]
    lat = [r["result"].latency_ms for r in rows]
    ttft = [r["result"].ttft_ms for r in rows if r["result"].ttft_ms is not None]
    with_facts = [r for r in answerable if r["id"] in facts]
    n = max(len(rows), 1)
    cost = sum(r["result"].cost for r in rows)
    routes: dict[str, int] = {}
    for r in rows:
        key = r["result"].route or r["result"].mode
        routes[key] = routes.get(key, 0) + 1
    return {
        "questions": len(rows),
        "hit_at_5": _ratio(sum(_hit(r) for r in answerable), len(answerable)),
        "coverage": _ratio(sum(_covered(r) for r in answerable), len(answerable)),
        "multi_coverage": _ratio(sum(_covered(r) for r in multi), len(multi)),
        "citation_correct": _ratio(sum(_correct(r) for r in rows), len(rows)),
        "fact_correct": _ratio(
            sum(facts_ok(r["result"].answer, facts[r["id"]]) for r in with_facts), len(with_facts)
        ),
        "leaks": sum(_leaks(r) for r in rows),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "ttft_p50_ms": _pct(ttft, 50),
        "tokens_in": sum(r["result"].input_tokens for r in rows) / n,
        "tokens_out": sum(r["result"].output_tokens for r in rows) / n,
        "cost_per_q": cost / n,
        "cost_per_1000": cost / n * 1000,
        "model_calls_free": sum(1 for r in rows if r["result"].input_tokens == 0),
        "routes": routes,
    }  # fmt: skip


def _hit(row: dict) -> bool:
    """At least one expected document was retrieved."""
    return bool(set(row["expected_doc_ids"]) & set(row["result"].retrieved_doc_ids))


def _covered(row: dict) -> bool:
    """All expected documents were retrieved."""
    return set(row["expected_doc_ids"]) <= set(row["result"].retrieved_doc_ids)


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def load_facts() -> dict[str, list[str]]:
    return yaml.safe_load(FACTS.read_text(encoding="utf-8")) if FACTS.exists() else {}


def run_config(
    cfg: Config, questions: list[dict], searcher: HybridSearcher, llm: LLM,
    facts: dict[str, list[str]], budget: dict | None = None,
) -> dict:  # fmt: skip
    """Run one configuration over one question set; with cache=True over a workload with repeats."""
    cache = SemanticCache() if cfg.cache else None
    work = build_workload(questions) if cfg.cache else questions
    rows = []
    for q in work:
        result = answer(q["question"], get_user(q["user"]), searcher, llm, cfg.mode, cfg, cache)
        rows.append({**q, "result": result})
        if budget is not None:
            budget["spent"] += result.cost
            if budget["spent"] > budget["limit"]:
                raise RuntimeError(f"budget exceeded: ${budget['spent']:.4f} > ${budget['limit']}")
    out = metrics(rows, facts)
    out["config"], out["mode"] = cfg.name, cfg.mode
    out["answers"] = [  # kept in a side file (see run_ablation), not in the summary rows
        {
            "id": r["id"], "question": r["question"], "user": r["user"], "type": r["type"],
            "route": r["result"].route or r["result"].mode, "answer": r["result"].answer,
            "citations": [c.doc_id for c in r["result"].citations],
            "correct": _correct(r), "covered": _covered(r) if r["expected_doc_ids"] else None,
            "facts_ok": facts_ok(r["result"].answer, facts[r["id"]]) if r["id"] in facts else None,
            "cost": r["result"].cost, "latency_ms": r["result"].latency_ms,
            "ttft_ms": r["result"].ttft_ms, "cached": r["result"].cached,
            "result": r["result"].to_dict(),
            "repeat_of": r.get("repeat_of"),
        }
        for r in rows
    ]  # fmt: skip
    if cache:
        repeats = [r for r in rows if r["repeat_of"]]
        originals = [r for r in rows if not r["repeat_of"]]
        out["cache"] = {
            "workload": len(rows),
            "repeats": len(repeats),
            "repeat_share": len(repeats) / len(rows),
            "repeat_hits": sum(r["result"].cached for r in repeats),
            "repeat_hit_rate": _ratio(sum(r["result"].cached for r in repeats), len(repeats)),
            "false_hits_on_new_questions": sum(r["result"].cached for r in originals),
        }
        # a cached paraphrase must give the same answer as the original question
        by_id = {r["id"]: r for r in originals}
        hit_ms = sorted(r["result"].latency_ms for r in repeats if r["result"].cached)
        out["cache"]["hit_latency_ms_median"] = hit_ms[len(hit_ms) // 2] if hit_ms else None
        out["cache"]["hit_latency_ms_max"] = hit_ms[-1] if hit_ms else None
        out["cache"]["repeat_same_answer"] = sum(
            r["result"].answer == by_id[r["repeat_of"]]["result"].answer
            for r in repeats if r["result"].cached
        )  # fmt: skip
    return out


def run_ablation(
    searcher: HybridSearcher, llm: LLM, configs: list[Config] | None = None,
    sets: list[str] | None = None, out: Path = OPTIMIZATION_JSON, budget: dict | None = None,
) -> dict:  # fmt: skip
    """Run configs x sets and merge the rows into eval/optimization.json (keyed by llm)."""
    configs = configs or [*ABLATION, BASELINE_AGENTIC, LEAN_AGENTIC, FINAL]
    facts = load_facts()
    report = (
        json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"rows": [], "ledger": []}
    )
    answers_file = out.with_name("optimization_answers.json")
    answer_store = (
        json.loads(answers_file.read_text(encoding="utf-8")) if answers_file.exists() else {}
    )
    for name in sets or list(QUESTION_SETS):
        questions = load_questions(QUESTION_SETS[name])
        for cfg in configs:
            started = time.perf_counter()
            row = run_config(cfg, questions, searcher, llm, facts, budget)
            answers = row.pop("answers")
            if llm.model != "fake":  # real answers are kept for inspection and for the demo
                answer_store.setdefault(llm.model, {}).setdefault(name, {})[cfg.name] = answers
            row |= {
                "set": name,
                "llm": llm.model,
                "seconds": round(time.perf_counter() - started, 1),
            }
            report["rows"] = [
                r for r in report["rows"]
                if not (r["config"] == cfg.name and r["set"] == name and r["llm"] == llm.model)
            ] + [row]  # fmt: skip
            if llm.model != "fake":
                report["ledger"].append(
                    {"config": cfg.name, "set": name, "cost": row["cost_per_q"] * row["questions"]}
                )
    report["generated"] = date.today().isoformat()
    report["embedder"] = searcher.embedder.name
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    answers_file.write_text(
        json.dumps(answer_store, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


def rescore(out: Path = OPTIMIZATION_JSON) -> int:
    """Recompute the answer-dependent columns of every stored real row with the CURRENT labels
    and facts (after a label correction), from the kept answers. No model call, no cost.
    Returns the number of rows updated."""
    from ks.rag import AskResult, Citation

    report = json.loads(out.read_text(encoding="utf-8"))
    store = json.loads(out.with_name("optimization_answers.json").read_text(encoding="utf-8"))
    facts = load_facts()
    n = 0
    for row in report["rows"]:
        saved = store.get(row["llm"], {}).get(row["set"], {}).get(row["config"])
        if not saved:
            continue
        labels = {q["id"]: q for q in load_questions(QUESTION_SETS[row["set"]])}
        rows = []
        for a in saved:
            if a["id"] not in labels:
                continue
            res = a["result"] | {"citations": [Citation(**c) for c in a["result"]["citations"]]}
            rows.append({**labels[a["id"]], "result": AskResult(**res)})
        new = metrics(rows, facts)
        for key in ("hit_at_5", "coverage", "multi_coverage", "citation_correct", "fact_correct"):
            row[key] = new[key]
        n += 1
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def format_markdown(report: dict, llm: str) -> str:
    """One markdown table per question set (rows = configurations) for the README."""
    names = [c.name for c in [*ABLATION, BASELINE_AGENTIC, LEAN_AGENTIC, FINAL]]
    head = (
        "| Configuration | hit@5 | both docs | citation | fact | leaks | p50 / p95 | TTFT p50 "
        "| tokens in / out | $ / question | $ / 1,000 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    out = []
    for name in ("tuning", "holdout"):
        rows = {r["config"]: r for r in report["rows"] if r["set"] == name and r["llm"] == llm}
        if not rows:
            continue
        title = "Tuning set (30 questions)" if name == "tuning" else "Held-out set (15 questions)"
        out.append(f"**{title}**\n\n{head}")
        for cfg in names:
            if cfg not in rows:
                continue
            r = rows[cfg]
            cache_note = f" ({r['questions']} q. with repeats)" if "cache" in r else ""
            out.append(
                f"| {cfg}{cache_note} | {r['hit_at_5']:.0%} | {r['coverage']:.0%} "
                f"| {r['citation_correct']:.0%} | {r['fact_correct']:.0%} | {r['leaks']} "
                f"| {r['p50_ms'] / 1000:.2f} / {r['p95_ms'] / 1000:.2f} s "
                f"| {r['ttft_p50_ms'] / 1000:.2f} s "
                f"| {r['tokens_in']:,.0f} / {r['tokens_out']:,.0f} "
                f"| ${r['cost_per_q']:.5f} | ${r['cost_per_1000']:.2f} |"
            )
        out.append("")
    return "\n".join(out)
