"""Static demo export: precomputed answers for a few questions per demo user (no server needed).

`python -m ks.cli export-demo` runs the prepared questions in BOTH modes with whatever
`get_llm()` returns and writes web/public/demo/demo.json. With the default FakeLLM that file is
free and deterministic; re-run it with LLM_PROVIDER=anthropic to refresh it with real Claude output.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

from ks.auth import USERS, User, allowed_departments
from ks.config import AUTO, Config
from ks.eval import RESULTS, load_questions
from ks.llm import LLM
from ks.optimize import OPTIMIZATION_JSON
from ks.pipeline import answer
from ks.scale import SCALE_JSON
from ks.search import HybridSearcher

DEMO_JSON = Path(__file__).resolve().parent.parent / "web" / "public" / "demo" / "demo.json"
ADMIN_LOGIN = "fiete"
DEMO_MODES = ("auto", "classic", "agentic")  # "auto" is the default in the UI
MAX_QUESTIONS = 8
MIN_QUESTIONS = 6

# Extra questions per department, used when a user has too few eval questions.
EXAMPLES = {
    "hr": ["Wie beantrage ich Elternzeit?"],
    "it": ["Wie richte ich den VPN-Zugang ein?"],
    "vertrieb": ["Wie wird die Provision ausgezahlt?"],
    "logistik": ["Welche Schutzausrüstung ist im Lager Pflicht?"],
    "alle": ["Wie läuft die Evakuierung im Brandfall ab?"],
}


def pick_questions(login: str, eval_questions: list[dict]) -> list[dict]:
    """Up to 6 answerable questions for one user: their own eval questions, topped up with
    department examples. The not-found and permission cases are added in build_demo."""
    base = [q for q in eval_questions if q["user"] == login and q["type"] in ("single", "multi")]
    base = base[: MAX_QUESTIONS - 2]
    texts = {q["question"] for q in base}
    for dept in (*USERS[login].departments, "alle"):
        for text in EXAMPLES.get(dept, []):
            if len(base) < MIN_QUESTIONS and text not in texts:
                base.append({"question": text, "type": "example"})
                texts.add(text)
    return base


def candidates(login: str, kind: str, eval_questions: list[dict]) -> list[dict]:
    """Eval questions of one kind, the user's own first, then those of the other users."""
    same = [q for q in eval_questions if q["type"] == kind]
    return sorted(same, key=lambda q: q["user"] != login)


# Which row of the ablation (eval/optimization.json) holds the answers of each demo mode.
MODE_CONFIG = {"auto": "+router (auto)", "classic": "+graph", "agentic": "agentic (all levers)"}
ANSWERS_JSON = OPTIMIZATION_JSON.with_name("optimization_answers.json")


def eval_cache(llm: LLM) -> dict:
    """Answers of the real ablation run, keyed by (user, question). Reused for the demo so the
    same question is not paid for twice. Only valid for the model that produced them."""
    stored = _read(ANSWERS_JSON) or {}
    if llm.model == "fake":
        return {}
    cache: dict = {}
    for question_set in stored.get(llm.model, {}).values():
        for mode, config in MODE_CONFIG.items():
            for a in question_set.get(config, []):
                cache.setdefault((a["user"], a["question"]), {})[mode] = a["result"]
    return cache


def ask_all(
    q: dict, user: User, searcher: HybridSearcher, llm: LLM, cache: dict | None = None,
    cfg: Config = AUTO,
) -> dict:  # fmt: skip
    cached = (cache or {}).get((user.login, q["question"]), {})
    results = {
        mode: cached.get(mode) or answer(q["question"], user, searcher, llm, mode, cfg).to_dict()
        for mode in DEMO_MODES
    }
    return {"question": q["question"], "kind": q["type"], "results": results}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _live_metrics(rows: list[tuple[str, dict]]) -> dict:
    """Same shape as GET /metrics, computed from the demo answers themselves."""
    modes: dict[str, dict] = {}
    for mode in DEMO_MODES:
        sel = [r for m, r in rows if m == mode]
        lat = np.array([r["latency_ms"] for r in sel])
        modes[mode] = {
            "count": len(sel),
            "avg_latency_ms": round(float(lat.mean()), 2),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
            "input_tokens": sum(r["input_tokens"] for r in sel),
            "output_tokens": sum(r["output_tokens"] for r in sel),
            "cost": round(sum(r["cost"] for r in sel), 6),
            "not_found_rate": round(sum(1 for r in sel if not r["found"]) / len(sel), 4),
        }
    missing: dict[str, int] = {}
    for m, r in rows:
        if m == "classic" and not r["found"]:
            missing[r["question"].lower()] = missing.get(r["question"].lower(), 0) + 1
    return {
        "total_requests": len(rows),
        "modes": modes,
        "top_not_found": [{"question": q, "count": n} for q, n in missing.items()],
    }


def build_demo(
    searcher: HybridSearcher, llm: LLM, eval_questions: list[dict], cfg: Config = AUTO
) -> dict:
    users, questions, documents, contents = [], {}, {}, {}
    rows: list[tuple[str, dict]] = []
    cache = eval_cache(llm)
    for login, user in USERS.items():
        allowed = allowed_departments(user)
        chunks = searcher.store.get_chunks(allowed)
        titles = {c.doc_id: (c.title, c.department) for c in chunks}
        users.append({
            "login": login, "name": user.name, "departments": ["alle", *user.departments],
            "is_admin": login == ADMIN_LOGIN,
        })  # fmt: skip
        documents[login] = [
            {"doc_id": d, "title": t, "department": dep} for d, (t, dep) in sorted(titles.items())
        ]
        questions[login], cited = [], set()

        entries = [
            ask_all(q, user, searcher, llm, cache, cfg)
            for q in pick_questions(login, eval_questions)
        ]
        for kind in ("not_found", "permission"):
            if kind == "permission" and login == ADMIN_LOGIN:
                continue  # the admin may read everything, so there is no permission case
            # show a case the system handles correctly: the "not found" sentence in both modes
            first = None
            for q in candidates(login, kind, eval_questions):  # stop at the first good case
                entry = ask_all(q, user, searcher, llm, cache, cfg)
                first = first or entry
                if not any(r["found"] for r in entry["results"].values()):
                    first = entry
                    break
            entries.append(first)
        for entry in entries:
            for mode, res in entry["results"].items():
                rows.append((mode, {**res, "question": entry["question"]}))
                cited |= {c["doc_id"] for c in res["citations"]}
        questions[login] = entries
        # content of the documents this user may open: only ones they may see, cited ones always
        contents[login] = {}
        for doc_id in sorted(titles):
            doc_chunks = searcher.store.get_document_chunks(doc_id, allowed)
            contents[login][doc_id] = {
                "doc_id": doc_id, "title": doc_chunks[0].title,
                "department": doc_chunks[0].department,
                "chunks": [
                    {"chunk_id": c.chunk_id, "heading": c.heading, "text": c.text}
                    for c in doc_chunks
                ],
            }  # fmt: skip
        assert cited <= set(contents[login]), "a citation points to a document the user may not see"
    return {
        "generated_by": llm.model,
        "generated_at": date.today().isoformat(),
        "embedder": searcher.embedder.name,
        "default_mode": "auto",
        "config": cfg.name,
        "users": users,
        "questions": questions,
        "documents": documents,
        "contents": contents,
        "metrics": {
            **_live_metrics(rows),
            "eval": _read(RESULTS),
            "optimization": _read(OPTIMIZATION_JSON),
            "scale": _read(SCALE_JSON),
        },
    }


def export_demo(searcher: HybridSearcher, llm: LLM, out: Path = DEMO_JSON) -> dict:
    demo = build_demo(searcher, llm, load_questions())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(demo, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return demo
