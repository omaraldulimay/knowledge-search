"""Scale benchmark: the same 30 eval questions on the 35 real documents vs. real + N generated ones.

Everything runs locally with MemoryStore and FakeLLM (classic mode). For pgvector see README."""

import importlib.util
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ks.auth import allowed_departments, get_user
from ks.embed import Embedder, get_embedder
from ks.eval import ROOT, load_questions, run_eval
from ks.indexer import load_manifest, save_manifest, sync_index
from ks.ingest import load_documents
from ks.llm import FakeLLM
from ks.search import HybridSearcher
from ks.store import MemoryStore

GENERATED = ROOT / "data" / "generated"
REAL_DOCS = ROOT / "data" / "docs"
SCALE_JSON = ROOT / "eval" / "scale.json"
SCALE_PNG = ROOT / "eval" / "scale.png"


def clean_generated(extra_dirs: list[Path] | None = None) -> list[Path]:
    """Delete data/generated (documents and indexes). Returns what was removed."""
    removed = []
    for path in [GENERATED, *(extra_dirs or [])]:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_corpus", ROOT / "scripts" / "gen_corpus.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_corpus"] = module
    spec.loader.exec_module(module)
    return module


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KB on Linux


def _build(docs, embedder: Embedder, index_dir: Path, label: str) -> tuple[MemoryStore, dict]:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    store, manifest = MemoryStore(), {}
    print(f"[{label}] indexing {len(docs)} documents ...")
    stats = sync_index(docs, store, embedder, manifest, progress=print)
    store.save(index_dir, embedder.name)
    save_manifest(index_dir, manifest)
    return store, {
        "documents": len(docs),
        "chunks": store.count(),
        "index_seconds": round(stats.seconds, 2),
        "index_size_mb": round(_dir_size(index_dir) / 1e6, 2),
    }


def _measure(store: MemoryStore, embedder: Embedder, questions: list[dict], label: str) -> dict:
    searcher = HybridSearcher(store, embedder)
    # warm-up: builds the BM25 index per department set (one-time cost, reported separately)
    start = time.perf_counter()
    for q in questions:
        searcher.search(q["question"], allowed_departments(get_user(q["user"])), k=5)
    warmup = time.perf_counter() - start
    times = []
    for _ in range(3):
        for q in questions:
            allowed = allowed_departments(get_user(q["user"]))
            t0 = time.perf_counter()
            searcher.search(q["question"], allowed, k=5)
            times.append((time.perf_counter() - t0) * 1000)
    report = run_eval(
        ["classic"], questions, searcher, FakeLLM(), embedder.name, GENERATED / f"eval_{label}.json"
    )
    m = report["modes"]["classic"]["metrics"]
    return {
        "hit_rate_at_5": round(m["hit_rate_at_5"], 3),
        "citation_correctness": round(m["citation_correctness"], 3),
        "permission_leaks": m["permission_leaks"],
        "search_p50_ms": round(float(np.percentile(times, 50)), 1),
        "search_p95_ms": round(float(np.percentile(times, 95)), 1),
        "bm25_warmup_seconds": round(warmup, 2),
        "peak_rss_mb": round(_peak_rss_mb()),
    }


def make_chart(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    small, large = result["small"], result["large"]
    labels = [f"{small['documents']} docs", f"{large['documents']:,} docs"]
    colors = ["#8aa4c4", "#1f4e8c"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=160)
    panels = [
        ("Hit rate @5 (higher is better)", "hit_rate_at_5", 100, "{:.0f}%"),
        ("Search latency p95 (lower is better)", "search_p95_ms", 1, "{:.0f} ms"),
    ]
    for ax, (title, key, factor, fmt) in zip(axes, panels, strict=True):
        values = [small[key] * factor, large[key] * factor]
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, max(values) * 1.2)
        for bar, v in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")  # fmt: skip
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
    fig.suptitle(
        f"30 eval questions, embedder {result['embedder']}, MemoryStore, classic mode",
        fontsize=8, color="#555",
    )  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_scale(
    n: int, embedder_name: str = "fastembed", seed: int = 42,
    out_json: Path = SCALE_JSON, out_png: Path | None = SCALE_PNG,
    real_docs: Path = REAL_DOCS, work_dir: Path = GENERATED,
) -> dict:  # fmt: skip
    try:
        embedder = get_embedder(embedder_name)
    except Exception as exc:  # fastembed missing or model not downloadable
        print(f"Embedder '{embedder_name}' unavailable ({exc}); falling back to hash")
        embedder = get_embedder("hash")
    questions = load_questions()
    gen = _load_generator()
    t0 = time.perf_counter()
    written = gen.generate(n, seed, work_dir / "docs")
    print(f"Generated {written} documents in {time.perf_counter() - t0:.1f}s")

    real = load_documents(real_docs)
    t0 = time.perf_counter()
    generated = load_documents(work_dir / "docs")
    print(f"Loaded {len(generated)} generated documents in {time.perf_counter() - t0:.1f}s")

    small_store, small = _build(real, embedder, work_dir / "index_small", "small")
    small |= _measure(small_store, embedder, questions, "small")
    del small_store

    large_dir = work_dir / "index_large"
    large_store, large = _build(real + generated, embedder, large_dir, "large")
    large |= _measure(large_store, embedder, questions, "large")

    # incremental indexing: nothing changed -> nothing is embedded
    manifest = load_manifest(large_dir)
    stats = sync_index(real + generated, large_store, embedder, manifest)
    result = {
        "generated_documents": len(generated),
        "n_requested": n,
        "seed": seed,
        "embedder": embedder.name,
        "store": "MemoryStore",
        "llm": "fake",
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "small": small,
        "large": large,
        "reindex_unchanged": {
            "seconds": round(stats.seconds, 2),
            "documents_embedded": stats.added + stats.changed,
        },
        "note": "peak_rss_mb is the process peak so far (small runs first, so large includes it)",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if out_png is not None:
        make_chart(result, out_png)
    return result


def format_scale(result: dict) -> str:
    rows = [
        ("documents", "documents", "{}"), ("chunks", "chunks", "{}"),
        ("hit rate@5", "hit_rate_at_5", "{:.0%}"),
        ("citation correct", "citation_correctness", "{:.0%}"),
        ("search p50 ms", "search_p50_ms", "{}"), ("search p95 ms", "search_p95_ms", "{}"),
        ("index time s", "index_seconds", "{}"), ("index size MB", "index_size_mb", "{}"),
        ("peak RSS MB", "peak_rss_mb", "{}"),
    ]  # fmt: skip
    lines = [f"{'metric':<20}{'small':>12}{'large':>12}"]
    for name, key, fmt in rows:
        a, b = fmt.format(result["small"][key]), fmt.format(result["large"][key])
        lines.append(f"{name:<20}{a:>12}{b:>12}")
    return "\n".join(lines)
