# knowledge-search

Permission-aware search over the internal documents of a fictional company
("Elbkontor Logistik GmbH"). Employees ask questions in German, the search returns the matching
document sections, and every user only sees documents of their own department.
Later phases add an LLM answer layer (classic RAG and agentic search), an API and a web UI.

**Status: phase 1** - sample data, ingestion, embeddings, vector stores, hybrid search, permissions.

## Run phase 1

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"      # "local" installs fastembed (downloads a ~220 MB model once)

python -m ks.cli index                                   # build the index in .ks_index/
python -m ks.cli search "Wie viel Urlaub habe ich?" --user mira
python -m ks.cli search "Wie hoch ist das Gehaltsband E5?" --user katja   # vertrieb: no hr hits

pytest && ruff check .                                   # offline, no model download
```

Without the `local` extra index with `--embedder hash` (or `KS_EMBEDDER=hash`): a deterministic toy
embedder for tests. Demo users: `mira` (hr), `ole` (it), `katja` (vertrieb), `bent` (logistik),
`fiete` (admin, all departments). Everybody additionally sees the department `alle`.

Optional Postgres + pgvector store: set `PG_DSN` and put `--store pgvector` before the command
(`python -m ks.cli --store pgvector index`).
The sample documents and the "combine two documents" questions are described in `data/README.md`.
