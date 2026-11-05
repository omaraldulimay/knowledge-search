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
from ks.config import ABLATION, BASELINE_AGENTIC, FINAL, FINAL_REWRITE, LEAN_AGENTIC, LEVERS, Config
from ks.eval import QUESTION_SETS, ROOT, VariantPool, _correct, _leaks, load_questions
from ks.llm import LLM, cost_usd
from ks.pipeline import answer
from ks.search import HybridSearcher

DEFAULT_SETS = ["tuning", "holdout"]  # holdout2 is only run on request (see eval/holdout2.yaml)
ALL_CONFIGS = [*ABLATION, BASELINE_AGENTIC, LEAN_AGENTIC, FINAL, FINAL_REWRITE, *LEVERS]
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


class Recorder:
    """Wraps a model and remembers every text it was given (prompt, tool results). Used to check
    whether the facts of an answer survived retrieval and compression, whatever the model says."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.seen: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._llm, name)

    def complete(self, system: str, user: str, *args, **kwargs):
        self.seen.append(user)
        return self._llm.complete(system, user, *args, **kwargs)

    def chat(self, system: str, messages: list[dict], *args, **kwargs):
        self.seen.append(json.dumps(messages[1:], ensure_ascii=False))
        return self._llm.chat(system, messages, *args, **kwargs)

    def context(self) -> str:
        text = " ".join(self.seen)
        self.seen = []
        return text


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
    in_context = [r for r in with_facts if r.get("context") and not r["result"].cached]
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
        "fact_in_context": _ratio(  # free check: did the facts reach the model at all?
            sum(facts_ok(r["context"], facts[r["id"]]) for r in in_context),
            len(in_context),
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
    cache = SemanticCache(calibrate=searcher.calibrate) if cfg.cache else None
    work = build_workload(questions) if cfg.cache else questions
    rows = []
    rec = Recorder(llm)
    for q in work:
        result = answer(q["question"], get_user(q["user"]), searcher, rec, cfg.mode, cfg, cache)
        rows.append({**q, "result": result, "context": rec.context()})
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
            "fact_in_context": facts_ok(r["context"], facts[r["id"]]) if r["id"] in facts else None,
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


NODOCS_NAME = "no documents"
NODOCS_SYSTEM = (
    "Du bist ein hilfreicher Assistent. Beantworte die Frage eines Mitarbeitenden der Elbkontor "
    "Logistik GmbH in Hamburg kurz auf Deutsch (höchstens drei Sätze)."
)
NODOCS_MAX_TOKENS = 150


def run_nodocs(
    questions: list[dict], llm: LLM, facts: dict[str, list[str]], budget: dict | None = None,
) -> dict:  # fmt: skip
    """The comparison that shows what the search is worth: ask the model every answerable question
    WITHOUT any document (short prompt, 150 tokens) and score the facts with the same checker.
    Not-found and permission questions are skipped: there is nothing to look up."""
    asked = [q for q in questions if q["expected_doc_ids"] and q["id"] in facts]
    answers, lat = [], []
    tokens_in = tokens_out = 0
    cost = 0.0
    for q in asked:
        started = time.perf_counter()
        res = llm.complete(NODOCS_SYSTEM, q["question"], max_tokens=NODOCS_MAX_TOKENS, think=False)
        lat.append((time.perf_counter() - started) * 1000)
        q_cost = cost_usd(llm, res.usage.input_tokens, res.usage.output_tokens)
        tokens_in += res.usage.input_tokens
        tokens_out += res.usage.output_tokens
        cost += q_cost
        ok = facts_ok(res.text, facts[q["id"]])
        answers.append({"id": q["id"], "question": q["question"], "answer": res.text,
                        "facts_ok": ok, "cost": q_cost})  # fmt: skip
        if budget is not None:
            budget["spent"] += q_cost
            if budget["spent"] > budget["limit"]:
                raise RuntimeError(f"budget exceeded: ${budget['spent']:.4f} > ${budget['limit']}")
    n = max(len(asked), 1)
    row = {
        "config": NODOCS_NAME, "mode": "none", "questions": len(asked),
        "hit_at_5": None, "coverage": None, "multi_coverage": None, "citation_correct": None,
        "fact_correct": _ratio(sum(a["facts_ok"] for a in answers), len(answers)),
        "fact_in_context": None, "leaks": 0, "p50_ms": _pct(lat, 50), "p95_ms": _pct(lat, 95),
        "ttft_p50_ms": 0.0, "tokens_in": tokens_in / n, "tokens_out": tokens_out / n,
        "cost_per_q": cost / n, "cost_per_1000": cost / n * 1000, "routes": {"none": len(asked)},
    }  # fmt: skip
    return row | {"answers": answers}


def run_ablation(
    searcher: "HybridSearcher | VariantPool", llm: LLM, configs: list[Config] | None = None,
    sets: list[str] | None = None, out: Path = OPTIMIZATION_JSON, budget: dict | None = None,
    nodocs: bool = False,
) -> dict:  # fmt: skip
    """Run configs x sets and merge the rows into eval/optimization.json (keyed by llm).
    With `nodocs` the "no documents" comparison row is added for every set."""
    configs = configs if configs is not None else ALL_CONFIGS
    pool = searcher if isinstance(searcher, VariantPool) else None
    base = pool.base if pool else searcher
    facts = load_facts()
    report = (
        json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"rows": [], "ledger": []}
    )
    answers_file = out.with_name("optimization_answers.json")
    answer_store = (
        json.loads(answers_file.read_text(encoding="utf-8")) if answers_file.exists() else {}
    )

    def keep(row: dict, answers: list[dict], name: str, started: float) -> None:
        if llm.model != "fake":  # real answers are kept for inspection and for the demo
            answer_store.setdefault(llm.model, {}).setdefault(name, {})[row["config"]] = answers
        row |= {"set": name, "llm": llm.model, "seconds": round(time.perf_counter() - started, 1)}
        report["rows"] = [
            r for r in report["rows"]
            if not (r["config"] == row["config"] and r["set"] == name and r["llm"] == llm.model)
        ] + [row]  # fmt: skip
        if llm.model != "fake":
            report["ledger"].append(
                {"config": row["config"], "set": name, "cost": row["cost_per_q"] * row["questions"]}
            )

    for name in sets or DEFAULT_SETS:
        questions = load_questions(QUESTION_SETS[name])
        if nodocs:
            started = time.perf_counter()
            row = run_nodocs(questions, llm, facts, budget)
            keep(row, row.pop("answers"), name, started)
        for cfg in configs:
            started = time.perf_counter()
            row = run_config(cfg, questions, pool.get(cfg) if pool else base, llm, facts, budget)
            keep(row, row.pop("answers"), name, started)
    report["generated"] = date.today().isoformat()
    report["embedder"] = base.embedder.name
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


def _fmt_pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def agent_share(row: dict) -> float:
    """Share of the questions the agent answered (route "agentic" or mode agentic)."""
    routes = row.get("routes", {})
    return routes.get("agentic", 0) / max(row["questions"], 1)


def format_markdown(report: dict, llm: str) -> str:
    """One markdown table per question set for the README. Rows measured with the real model show
    everything; rows that only ran offline (no model call, free) show retrieval and facts in the
    prompt, and a dash for what only a real model can tell (answer facts, citations, time, cost)."""
    names = [
        NODOCS_NAME,
        *(c.name for c in ALL_CONFIGS),
    ]
    head = (
        "| Configuration | measured | hit@5 | both docs | facts in prompt | fact correct "
        "| citation | leaks | agent | p50 | $ / 1,000 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    out = []
    for name in ("tuning", "holdout"):
        real = {r["config"]: r for r in report["rows"] if r["set"] == name and r["llm"] == llm}
        fake = {r["config"]: r for r in report["rows"] if r["set"] == name and r["llm"] == "fake"}
        if not real and not fake:
            continue
        title = "Tuning set (30 questions)" if name == "tuning" else "Held-out set (15 questions)"
        out.append(f"**{title}**\n\n{head}")
        for cfg in names:
            r = real.get(cfg) or fake.get(cfg)
            if r is None:
                continue
            is_real = cfg in real
            note = f" ({r['questions']} q. with repeats)" if "cache" in r else ""
            cells = [
                f"{cfg}{note}", "real" if is_real else "offline",
                _fmt_pct(r["hit_at_5"]), _fmt_pct(r["coverage"]),
                _fmt_pct(r.get("fact_in_context")),
                _fmt_pct(r["fact_correct"]) if is_real else "–",
                _fmt_pct(r["citation_correct"]) if is_real else "–", str(r["leaks"]),
                _fmt_pct(agent_share(r)) if cfg != NODOCS_NAME else "–",
                f"{r['p50_ms'] / 1000:.2f} s" if is_real else "–",
                f"${r['cost_per_1000']:.2f}" if is_real else "–",
            ]  # fmt: skip
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
