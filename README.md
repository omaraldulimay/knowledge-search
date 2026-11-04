# knowledge-search

Permission-aware search over the internal documents of a fictional company
("Elbkontor Logistik GmbH"). Employees ask questions in German, the search returns the matching
document sections, and every user only sees documents of their own department.
Later phases add an LLM answer layer (classic RAG and agentic search), an API and a web UI.

**Status: phase 3** - search, LLM answers (classic and agentic), scale benchmark, HTTP API, Docker, CI.

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

## Run the API (phase 3)

```bash
pip install -e ".[dev,local]"
uvicorn ks.api:app --reload          # http://localhost:8000/docs has the interactive OpenAPI page
```

On startup the API indexes `data/docs` (incrementally; `MemoryStore`, or pgvector if `PG_DSN` is set).
Configuration by environment: `APP_SECRET` (token signing; a development default and a warning are
used if unset), `KS_EMBEDDER` (`fastembed`, `hash` or `auto` = fastembed if installed), `PG_DSN`,
`KS_DOCS_DIR`, `LLM_PROVIDER` (default `fake`; `anthropic` needs `ANTHROPIC_API_KEY`).
CORS allows the Vite dev server (`http://localhost:5173`).

```bash
TOKEN=$(curl -s -X POST localhost:8000/login -H 'content-type: application/json' \
  -d '{"username": "katja"}' | python -c "import sys,json; print(json.load(sys.stdin)['token'])")
AUTH="Authorization: Bearer $TOKEN"

curl -s -X POST localhost:8000/ask -H "$AUTH" -H 'content-type: application/json' \
  -d '{"question": "Wie viel darf ein Hotel pro Nacht kosten?", "mode": "agentic"}'
curl -s localhost:8000/documents -H "$AUTH"                    # only documents katja may see
curl -s localhost:8000/documents/hr/gehaltsbaender -H "$AUTH"  # 404, same as a missing document
# admin (fiete) only: upload, delete, metrics
curl -s -X POST localhost:8000/documents -H "$ADMIN" -F department=it -F file=@neu.md
curl -s -X DELETE localhost:8000/documents/it/neu -H "$ADMIN"
curl -s localhost:8000/metrics -H "$ADMIN"
```

| Endpoint | Who | What |
|---|---|---|
| `GET /health` | public | status, chunk count |
| `POST /login` `{username}` | public | signed token (8 h). **Demo simplification:** no passwords; a real setup would use SSO / OIDC |
| `POST /ask` `{question, mode: classic\|agentic}` | user | answer, citations, steps, tokens, cost, latency |
| `GET /documents`, `GET /documents/{doc_id}` | user | only allowed documents; forbidden and missing give the identical 404 |
| `POST /documents` (multipart `file`, `department`), `DELETE /documents/{doc_id}` | admin (`fiete`) | md/pdf/docx upload with incremental indexing; delete |
| `GET /metrics` | admin | per mode: count, avg/p95 latency, tokens, cost, not-found rate; top unanswered questions (= missing documentation); latest `eval/results.json` and `eval/scale.json` |

`/metrics` is admin-only because the list of unanswered questions reveals what other departments ask.
The request log is in memory (SQLite); set `KS_METRICS_DB=/path/file.db` to keep it.

## Docker

```bash
docker compose up --build            # api on http://localhost:8000, Postgres + pgvector behind it
```

`docker-compose.yml` starts `postgres` (`pgvector/pgvector:pg16`, with a healthcheck) and `api`
(`PG_DSN` points to it; the API waits until Postgres is healthy). The image is built from
`python:3.12-slim`, runs as a non-root user and downloads the embedding model at build time.
Set `APP_SECRET` (and optionally `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`) in your shell
or an `.env` file. Uploaded documents live inside the container, so they are gone when it is recreated.

CI (`.github/workflows/ci.yml`) runs ruff and pytest against a pgvector service container (so the
pgvector tests run too, with the fake LLM only, no secrets) and builds the Docker image without pushing it.
