"""Command line: `python -m ks.cli index` and `python -m ks.cli search "frage" --user mira`."""

import argparse
import os
import sys
from pathlib import Path

from ks.auth import USERS, allowed_departments, get_user
from ks.embed import Embedder, get_embedder
from ks.eval import QUESTIONS, RESULTS
from ks.ingest import chunk_documents, embedding_text, load_documents
from ks.search import HybridSearcher
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
    chunks = chunk_documents(load_documents(Path(args.docs)))
    vectors = embedder.embed([embedding_text(c) for c in chunks])
    if args.store == "pgvector":
        store = _pg_store(embedder.dim)
        store.init_schema(reset=True)
        store.add(chunks, vectors)
    else:
        store = MemoryStore()
        store.add(chunks, vectors)
        store.save(Path(args.index_dir), embedder.name)
    docs = len({c.doc_id for c in chunks})
    print(f"Indexed {docs} documents / {len(chunks)} chunks with '{embedder.name}' ({args.store}).")


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
    hits = HybridSearcher(store, embedder).search(args.query, allowed_departments(user), k=args.k)
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
    modes = ["classic", "agentic"] if args.mode == "both" else [args.mode]
    questions = load_questions(Path(args.questions))
    report = run_eval(modes, questions, searcher, get_llm(), embedder, Path(args.out))
    print(format_table(report))
    print(f"Written: {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ks")
    parser.add_argument("--store", choices=["memory", "pgvector"], default="memory")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="load documents, embed chunks, store them")
    p_index.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_index.add_argument("--embedder", choices=["fastembed", "hash"],
                         default=os.environ.get("KS_EMBEDDER", "fastembed"))  # fmt: skip
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="hybrid search as a given user")
    p_search.add_argument("query")
    p_search.add_argument("--user", required=True)
    p_search.add_argument("-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="run eval/questions.yaml and write eval/results.json")
    p_eval.add_argument("--mode", choices=["classic", "agentic", "both"], default="both")
    p_eval.add_argument("--questions", default=str(QUESTIONS))
    p_eval.add_argument("--out", default=str(RESULTS))
    p_eval.add_argument("--docs", default=str(DEFAULT_DOCS))
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
