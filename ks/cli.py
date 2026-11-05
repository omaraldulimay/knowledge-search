"""Command line: `python -m ks.cli index` and `python -m ks.cli search "frage" --user mira`."""

import argparse
import os
import sys
from pathlib import Path

from ks.auth import USERS, allowed_departments, get_user
from ks.config import SERVING_EMBEDDER
from ks.embed import Embedder, get_embedder
from ks.eval import QUESTIONS, RESULTS
from ks.indexer import load_manifest, save_manifest, sync_index
from ks.ingest import load_documents
from ks.search import build_searcher
from ks.store import MemoryStore, PgVectorStore, VectorStore

DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"
DEFAULT_INDEX = Path(os.environ.get("KS_INDEX_DIR", ".ks_index"))


def _pg_store(dim: int) -> PgVectorStore:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        sys.exit("PG_DSN is not set")
    return PgVectorStore(dsn, dim)


def cmd_index(args: argparse.Namespace) -> None:
    embedder = get_embedder(args.embedder)
    docs = load_documents(Path(args.docs))
    index_dir = Path(args.index_dir)
    if args.store == "pgvector":
        store = _pg_store(embedder.dim)
        store.init_schema(reset=True)
        manifest: dict[str, str] = {}
    elif (
        not args.full
        and (index_dir / "meta.json").exists()
        and MemoryStore.saved_embedder_name(index_dir) == embedder.spec
    ):
        store, manifest = MemoryStore.load(index_dir), load_manifest(index_dir)
    else:  # first run, --full, or another embedder: start from scratch
        store, manifest = MemoryStore(), {}
    stats = sync_index(docs, store, embedder, manifest, progress=print)
    if args.store == "memory":
        store.save(index_dir, embedder.spec)
        save_manifest(index_dir, manifest)
        build_searcher(
            store, embedder, index_dir, save=True, warm=True
        )  # links.json, sentences.npz
    print(
        f"Index '{embedder.name}' ({args.store}): {stats.added} new, {stats.changed} changed, "
        f"{stats.removed} removed, {stats.unchanged} unchanged documents; "
        f"{stats.chunks_embedded} chunks embedded in {stats.seconds:.1f}s; "
        f"{store.count()} chunks in total."
    )


def cmd_bench_scale(args: argparse.Namespace) -> None:
    from ks.scale import format_scale, run_scale

    result = run_scale(args.n, args.embedder, args.seed)
    print(format_scale(result))
    print("Written: eval/scale.json, eval/scale.png")


def cmd_clean_generated(args: argparse.Namespace) -> None:
    from ks.scale import clean_generated

    removed = clean_generated()
    print("Removed: " + ", ".join(map(str, removed)) if removed else "Nothing to remove.")


def _open(args: argparse.Namespace) -> tuple[VectorStore, Embedder]:
    if args.store == "pgvector":
        embedder = get_embedder(os.environ.get("KS_EMBEDDER", "fastembed"))
        return _pg_store(embedder.dim), embedder
    index_dir = Path(args.index_dir)
    if not (index_dir / "meta.json").exists():
        sys.exit("No index found. Run: python -m ks.cli index")
    name = MemoryStore.saved_embedder_name(index_dir)  # must match the indexing embedder
    return MemoryStore.load(index_dir), get_embedder(name)


def cmd_search(args: argparse.Namespace) -> None:
    try:
        user = get_user(args.user)
    except KeyError:
        sys.exit(f"Unknown user '{args.user}'. Known: {', '.join(USERS)}")
    store, embedder = _open(args)
    searcher = build_searcher(store, embedder, Path(args.index_dir))
    hits = searcher.search(
        args.query, allowed_departments(user), k=args.k, expand=args.optimized, graph=args.optimized
    )
    print(f"User: {user.name} (departments: alle, {', '.join(user.departments)})")
    if not hits:
        print("No results.")
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        snippet = " ".join(c.text.split())[:160]
        print(f"{i}. {c.title} > {c.heading}  [{c.department}]")
        print(f"   {c.chunk_id}, score {hit.score:.4f}")
        print(f"   {snippet}...")


def cmd_eval(args: argparse.Namespace) -> None:
    from ks.eval import format_table, load_questions, open_searcher, run_eval
    from ks.llm import get_llm

    searcher, embedder = open_searcher(Path(args.index_dir), Path(args.docs))
    from ks.config import ABLATION

    cfg = next((c for c in ABLATION if c.name == args.config), None)
    if cfg is None:
        sys.exit(f"Unknown config. Known: {', '.join(c.name for c in ABLATION)}")
    all_modes = {"both": ["classic", "agentic"], "all": ["auto", "classic", "agentic"]}
    modes = all_modes.get(args.mode, [args.mode])
    questions = load_questions(Path(args.questions))
    report = run_eval(modes, questions, searcher, get_llm(), embedder, Path(args.out), cfg)
    print(format_table(report))
    print(f"Written: {args.out}")


def cmd_doc2query(args: argparse.Namespace) -> None:
    """Generate colloquial questions for every chunk without an entry (one model call each)."""
    from ks.doc2query import CACHE_FILE, chunk_key, generate, load_cache, save_cache
    from ks.ingest import chunk_documents
    from ks.llm import get_llm

    chunks = chunk_documents(load_documents(Path(args.docs)))
    cache = load_cache()
    missing = [c for c in chunks if chunk_key(c) not in cache]
    words = sum(len(c.text.split()) for c in missing)
    est = (len(missing) * 220 + words * 1.3) * 0.1 / 1e6 + len(missing) * 80 * 0.5 / 1e6
    print(f"{len(chunks)} chunks, {len(missing)} without questions; estimated cost ${est:.4f}")
    if args.estimate or not missing:
        return
    llm = get_llm("anthropic")
    try:
        spent = generate(missing[: args.limit or None], llm, cache, args.budget)
    finally:
        save_cache(cache)  # keep what was generated, even after an error
    print(f"Generated for {len(cache)} chunks in total, cost ${spent:.4f}. Written: {CACHE_FILE}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Print the straight line that maps this model's cosine onto the MiniLM scale."""
    from ks.embed import FASTEMBED_MODEL
    from ks.eval import calibration_anchors

    ref_yes, ref_no = calibration_anchors("fastembed")
    yes, no = calibration_anchors(args.embedder)
    slope = (ref_yes - ref_no) / (yes - no)
    offset = ref_yes - slope * yes
    model = args.embedder.split(":", 1)[1] if ":" in args.embedder else FASTEMBED_MODEL
    print(f"MiniLM medians: answerable {ref_yes:.3f}, unanswerable {ref_no:.3f}")
    print(f"{model}: answerable {yes:.3f}, unanswerable {no:.3f}")
    print(f'CALIBRATION["{model}"] = ({slope:.3f}, {offset:.3f})')


def cmd_seed_glossary(args: argparse.Namespace) -> None:
    import yaml

    from ks.auth import DEPARTMENTS
    from ks.glossary import GLOSSARY_FILE, seed_from_chunks
    from ks.links import build_links

    store, _ = _open(args)
    chunks = store.get_chunks(DEPARTMENTS)
    groups = seed_from_chunks(chunks, build_links(chunks).key_terms)
    text = yaml.safe_dump(groups, allow_unicode=True, default_flow_style=True, width=100)
    print(f"{len(groups)} automatic groups:\n{text}")
    if args.write:  # replace everything after the "auto:" line, keep the curated part as is
        old = GLOSSARY_FILE.read_text(encoding="utf-8").split("\nauto:")[0]
        dump = lambda g: yaml.safe_dump(g, allow_unicode=True, default_flow_style=True, width=200)  # noqa: E731
        lines = "".join(f"  - {dump(g).strip()}\n" for g in groups)
        GLOSSARY_FILE.write_text(f"{old}\nauto:\n{lines}", encoding="utf-8")
        print(f"Written: {GLOSSARY_FILE}")


def cmd_optimize(args: argparse.Namespace) -> None:
    from ks.eval import open_pool
    from ks.llm import get_llm
    from ks.optimize import ALL_CONFIGS, run_ablation

    searcher = open_pool(Path(args.index_dir), Path(args.docs))
    llm = get_llm("anthropic" if args.real else "fake")
    wanted = args.configs.split(",") if args.configs else None
    configs = [c for c in ALL_CONFIGS if wanted is None or c.name in wanted]
    if args.nodocs_only:
        configs = []
    sets = args.sets.split(",")
    budget = {"spent": 0.0, "limit": args.budget} if args.real else None
    report = run_ablation(
        searcher, llm, configs, sets, Path(args.out), budget, args.nodocs or args.nodocs_only
    )
    if budget:
        total = sum(entry["cost"] for entry in report["ledger"])
        print(f"API cost of this run: ${budget['spent']:.4f}; ledger total: ${total:.4f}")
    print(f"Written: {args.out} ({len(report['rows'])} rows)")


def cmd_optimize_table(args: argparse.Namespace) -> None:
    import json

    from ks.optimize import OPTIMIZATION_JSON, format_markdown

    report = json.loads(OPTIMIZATION_JSON.read_text(encoding="utf-8"))
    print(format_markdown(report, args.llm))


def cmd_optimize_rescore(args: argparse.Namespace) -> None:
    from ks.optimize import rescore

    print(f"Rescored {rescore()} rows with the current labels and facts (no model call).")


def cmd_export_demo(args: argparse.Namespace) -> None:
    from ks.demo import export_demo
    from ks.eval import open_searcher
    from ks.llm import get_llm

    searcher, _ = open_searcher(Path(args.index_dir), Path(args.docs))
    demo = export_demo(searcher, get_llm(), Path(args.out))
    n = sum(len(q) for q in demo["questions"].values())
    by = demo["generated_by"]
    print(f"Demo written to {args.out}: {n} questions x 3 modes (auto, classic, agentic), by {by}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ks")
    parser.add_argument("--store", choices=["memory", "pgvector"], default="memory")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="load documents, embed chunks, store them")
    p_index.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_index.add_argument("--embedder", default=SERVING_EMBEDDER,
                         help="hash | fastembed (MiniLM) | fastembed:<model name>")  # fmt: skip
    p_index.set_defaults(func=cmd_index)

    p_index.add_argument("--full", action="store_true", help="ignore the old index, rebuild all")

    p_bench = sub.add_parser("bench-scale", help="generate N filler docs, benchmark small vs large")
    p_bench.add_argument("--n", type=int, default=10000)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--embedder", choices=["fastembed", "hash"], default="fastembed")
    p_bench.set_defaults(func=cmd_bench_scale)

    p_clean = sub.add_parser("clean-generated", help="delete data/generated (docs and indexes)")
    p_clean.set_defaults(func=cmd_clean_generated)

    p_search = sub.add_parser("search", help="hybrid search as a given user")
    p_search.add_argument("query")
    p_search.add_argument("--user", required=True)
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--optimized", action="store_true", help="glossary + graph expansion")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="run eval/questions.yaml and write eval/results.json")
    p_eval.add_argument(
        "--mode", choices=["classic", "agentic", "auto", "both", "all"], default="both"
    )
    p_eval.add_argument("--config", default="baseline", help="which levers are on (see optimize)")
    p_eval.add_argument("--questions", default=str(QUESTIONS))
    p_eval.add_argument("--out", default=str(RESULTS))
    p_eval.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_eval.set_defaults(func=cmd_eval)

    p_d2q = sub.add_parser("doc2query", help="generate colloquial questions per chunk (real model)")
    p_d2q.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_d2q.add_argument("--estimate", action="store_true", help="only print the estimated cost")
    p_d2q.add_argument("--limit", type=int, default=0, help="only the first N missing chunks")
    p_d2q.add_argument("--budget", type=float, default=0.03, help="stop above this USD amount")
    p_d2q.set_defaults(func=cmd_doc2query)

    p_cal = sub.add_parser("calibrate", help="map a model's cosine scores onto the MiniLM scale")
    p_cal.add_argument("--embedder", required=True)
    p_cal.set_defaults(func=cmd_calibrate)

    p_seed = sub.add_parser("seed-glossary", help="propose glossary groups from the documents")
    p_seed.add_argument("--write", action="store_true", help="write them into data/glossary.yaml")
    p_seed.set_defaults(func=cmd_seed_glossary)

    from ks.optimize import OPTIMIZATION_JSON

    p_opt = sub.add_parser("optimize", help="ablation: every lever on/off, tuning + held-out set")
    p_opt.add_argument("--real", action="store_true", help="use the real model (costs money)")
    p_opt.add_argument("--configs", help="comma-separated config names, default: all")
    p_opt.add_argument("--sets", default="tuning,holdout")
    p_opt.add_argument(
        "--nodocs", action="store_true", help="add the 'no documents' comparison row"
    )
    p_opt.add_argument("--nodocs-only", action="store_true", help="only the 'no documents' row")
    p_opt.add_argument("--budget", type=float, default=0.05, help="abort above this USD amount")
    p_opt.add_argument("--out", default=str(OPTIMIZATION_JSON))
    p_opt.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_opt.set_defaults(func=cmd_optimize)

    p_table = sub.add_parser("optimize-table", help="print the ablation tables as markdown")
    p_table.add_argument("--llm", default="claude-haiku-5-5")
    p_table.set_defaults(func=cmd_optimize_table)

    p_rescore = sub.add_parser("optimize-rescore", help="re-score stored answers after a label fix")
    p_rescore.set_defaults(func=cmd_optimize_rescore)

    from ks.demo import DEMO_JSON

    p_demo = sub.add_parser("export-demo", help="write the static GitHub Pages demo data")
    p_demo.add_argument("--out", default=str(DEMO_JSON))
    p_demo.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_demo.set_defaults(func=cmd_export_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
