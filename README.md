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

## Scale benchmark (phase 2b)

How does search behave with 35 documents vs. 35 + N generated ones? `scripts/gen_corpus.py`
writes seeded, template-based filler (delivery notes, shift logs, project notes, minutes,
customer letters) plus distractors: outdated policy versions ("Reisekostenrichtlinie (Stand 2019)",
other numbers) and near-duplicate FAQ pages. No LLM involved, everything is invented.

```bash
python -m ks.cli bench-scale --n 10000     # writes eval/scale.json and eval/scale.png
python -m ks.cli clean-generated           # deletes data/generated (docs and indexes, gitignored)
python -m ks.cli index                     # incremental: only new/changed docs are embedded
```

- **Versions:** documents may have front matter (`policy`, `valid_from`, `superseded`). If search
  candidates contain several versions of one policy, only the newest valid one stays
  (`apply_version_rule` in `ks/search.py`); an outdated chunk without a newer sibling is demoted.
- **Incremental index:** a content hash per document (`manifest.json` in the index dir) decides what
  is re-embedded; deleted files are removed. Embeddings run in batches with progress output.
- **Against pgvector** (HNSW index is in `ks/schema.sql`): start Postgres with the pgvector
  extension, `export PG_DSN=postgresql://...`, then
  `python scripts/gen_corpus.py --n 10000 --out data/generated/all && cp -r data/docs/* data/generated/all/`,
  `python -m ks.cli --store pgvector index --docs data/generated/all`, then
  `python -m ks.cli --store pgvector search "..." --user mira`
  to compare latency with the numbers in `eval/scale.json`. The benchmark command itself uses
  `MemoryStore` only.
