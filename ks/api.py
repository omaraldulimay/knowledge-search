"""FastAPI app: login, ask (classic / agentic), documents, admin upload/delete, metrics.

Demo simplification: `/login` only takes a user name, there are no passwords. A real deployment
would put SSO / OIDC in front and derive the user from the verified identity token. The rest
(signed token -> user -> allowed departments -> permission-filtered store) stays the same.
"""

import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from ks.auth import DEPARTMENTS, USERS, User, allowed_departments, get_user
from ks.cache import SemanticCache
from ks.config import ABLATION, CACHED, Config
from ks.embed import Embedder, HashEmbedder, get_embedder
from ks.eval import RESULTS
from ks.indexer import load_manifest, save_manifest, sync_index, upsert_documents
from ks.ingest import LOADERS, load_document, load_documents
from ks.llm import LLM, get_llm
from ks.pipeline import answer
from ks.scale import SCALE_JSON
from ks.search import HybridSearcher, build_searcher
from ks.store import MemoryStore, PgVectorStore, VectorStore

log = logging.getLogger("ks.api")

ROOT = Path(__file__).resolve().parent.parent
DEV_SECRET = "dev-secret-change-me"
TOKEN_MAX_AGE = 8 * 3600  # seconds
ADMIN_LOGIN = "fiete"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
NOT_FOUND_DETAIL = "Dokument nicht gefunden"  # same text for missing AND forbidden documents
SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]  # Vite dev server


# ---------------------------------------------------------------- request log (SQLite)


class RequestLog:
    """One row per /ask call. In memory by default; set KS_METRICS_DB to keep it in a file."""

    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                ts REAL, user TEXT, mode TEXT, question TEXT, latency_ms REAL,
                input_tokens INTEGER, output_tokens INTEGER, cost REAL, found INTEGER)"""
        )

    def add(self, user: str, question: str, result: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), user, result["mode"], question, result["latency_ms"],
                 result["input_tokens"], result["output_tokens"], result["cost"],
                 int(result["found"])),
            )  # fmt: skip
            self._db.commit()

    def summary(self, top_n: int = 10) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT mode, latency_ms, input_tokens, output_tokens, cost, found FROM requests"
            ).fetchall()
            missing = self._db.execute(
                """SELECT lower(trim(question)) AS q, count(*) AS n, max(ts) AS last
                   FROM requests WHERE found = 0 GROUP BY q ORDER BY n DESC, last DESC LIMIT ?""",
                (top_n,),
            ).fetchall()
        modes: dict[str, dict] = {}
        for mode in sorted({r[0] for r in rows}):
            sel = [r for r in rows if r[0] == mode]
            lat = np.array([r[1] for r in sel])
            modes[mode] = {
                "count": len(sel),
                "avg_latency_ms": round(float(lat.mean()), 2),
                "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
                "input_tokens": sum(r[2] for r in sel),
                "output_tokens": sum(r[3] for r in sel),
                "cost": round(sum(r[4] for r in sel), 6),
                "not_found_rate": round(sum(1 for r in sel if not r[5]) / len(sel), 4),
            }
        return {
            "total_requests": len(rows),
            "modes": modes,
            "top_not_found": [{"question": q, "count": n} for q, n, _ in missing],
        }


# ---------------------------------------------------------------- application state


@dataclass
class AppState:
    store: VectorStore
    embedder: Embedder
    searcher: HybridSearcher
    llm: LLM
    docs_dir: Path
    index_dir: Path
    manifest: dict[str, str]
    log: RequestLog
    secret: str
    lock: threading.Lock  # serialises upload / delete (they change store, files and manifest)
    cfg: Config  # which optimisation levers are on (KS_CONFIG, default: all)
    cache: SemanticCache


def make_embedder() -> Embedder:
    """KS_EMBEDDER=fastembed|hash|auto (default: fastembed if installed, else hash)."""
    name = os.environ.get("KS_EMBEDDER", "auto")
    if name != "auto":
        return get_embedder(name)
    try:
        return get_embedder("fastembed")
    except Exception as exc:  # fastembed not installed or model not available
        log.warning("fastembed unavailable (%s) - falling back to the hash embedder", exc)
        return HashEmbedder()


def build_state(
    embedder: Embedder | None = None, llm: LLM | None = None, docs_dir: Path | None = None,
    store: VectorStore | None = None, index_dir: Path | None = None,
) -> AppState:  # fmt: skip
    embedder = embedder or make_embedder()
    docs_dir = docs_dir or Path(os.environ.get("KS_DOCS_DIR", ROOT / "data" / "docs"))
    index_dir = index_dir or Path(os.environ.get("KS_INDEX_DIR", ".ks_index_api"))
    dsn = os.environ.get("PG_DSN")
    manifest: dict[str, str] = {}
    if store is None and dsn:
        pg = PgVectorStore(dsn, embedder.dim)
        pg.init_schema()
        # reuse the stored index only if we still know which file versions it was built from
        if pg.count() and (index_dir / "manifest.json").exists():
            manifest = load_manifest(index_dir)
        else:
            pg.init_schema(reset=True)
        store = pg
    store = store if store is not None else MemoryStore()
    stats = sync_index(load_documents(docs_dir), store, embedder, manifest)
    log.info("index ready: %s chunks (%s)", store.count(), stats)
    if isinstance(store, PgVectorStore):
        save_manifest(index_dir, manifest)
    cfg = next((c for c in ABLATION if c.name == os.environ.get("KS_CONFIG")), CACHED)
    secret = os.environ.get("APP_SECRET")
    if not secret:
        log.warning("APP_SECRET is not set - using an insecure development secret")
    return AppState(
        store=store, embedder=embedder, searcher=build_searcher(store, embedder, warm=True),
        llm=llm or get_llm(), docs_dir=docs_dir, index_dir=index_dir, manifest=manifest,
        log=RequestLog(os.environ.get("KS_METRICS_DB", ":memory:")),
        secret=secret or DEV_SECRET, lock=threading.Lock(), cfg=cfg, cache=SemanticCache(),
    )  # fmt: skip


# ---------------------------------------------------------------- request / response models


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["classic", "agentic", "auto"] = "auto"


# Technical details of an answer: only the admin gets them (employees see answer + sources).
ADMIN_ONLY_FIELDS = (
    "steps", "input_tokens", "output_tokens", "cost", "latency_ms", "ttft_ms", "cached", "route",
    "retrieved_doc_ids",
)  # fmt: skip


def visible_result(result: dict, user: User) -> dict:
    """The /ask response for this user. The request log keeps the full result."""
    if user.login == ADMIN_LOGIN:
        return result
    return {k: v for k, v in result.items() if k not in ADMIN_ONLY_FIELDS}


def refresh_after_change(st: AppState) -> None:
    """Documents changed: rebuild the link index and forget cached answers."""
    st.searcher.links = build_searcher(st.store, st.embedder).links
    st.cache.clear()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_app(
    embedder: Embedder | None = None, llm: LLM | None = None, docs_dir: Path | None = None,
    store: VectorStore | None = None, index_dir: Path | None = None,
) -> FastAPI:  # fmt: skip
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ks = build_state(embedder, llm, docs_dir, store, index_dir)
        yield

    app = FastAPI(title="Elbkontor Wissenssuche", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"]
    )
    bearer = HTTPBearer(auto_error=False)

    def state() -> AppState:
        return app.state.ks

    def current_user(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
        st: Annotated[AppState, Depends(state)],
    ) -> User:
        unauthorized = HTTPException(
            401, "Nicht angemeldet", headers={"WWW-Authenticate": "Bearer"}
        )
        if creds is None:
            raise unauthorized
        try:
            data = URLSafeTimedSerializer(st.secret).loads(creds.credentials, max_age=TOKEN_MAX_AGE)
            return get_user(data["u"])
        except (BadSignature, SignatureExpired, KeyError, TypeError):
            raise unauthorized from None

    def admin_user(user: Annotated[User, Depends(current_user)]) -> User:
        if user.login != ADMIN_LOGIN:
            raise HTTPException(403, "Nur für Administratoren")
        return user

    St = Annotated[AppState, Depends(state)]
    Current = Annotated[User, Depends(current_user)]
    Admin = Annotated[User, Depends(admin_user)]

    @app.get("/health")
    def health(st: St) -> dict:
        return {"status": "ok", "chunks": st.store.count(), "embedder": st.embedder.name}

    @app.post("/login")
    def login(body: LoginRequest, st: St) -> dict:
        try:
            user = get_user(body.username)
        except KeyError:
            raise HTTPException(401, "Unbekannter Benutzer") from None
        token = URLSafeTimedSerializer(st.secret).dumps({"u": user.login})
        return {
            "token": token, "token_type": "bearer", "login": user.login, "name": user.name,
            "departments": ["alle", *user.departments], "is_admin": user.login == ADMIN_LOGIN,
        }  # fmt: skip

    @app.post("/ask")
    def ask(body: AskRequest, user: Current, st: St) -> dict:
        result = answer(
            body.question, user, st.searcher, st.llm, body.mode, st.cfg, st.cache
        ).to_dict()
        st.log.add(user.login, body.question, result)
        return visible_result(result, user)

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest, user: Current, st: St) -> StreamingResponse:
        """Server-sent events: `delta` (answer text as it is written), then `done` with the same
        JSON as /ask (citations resolved). Tool turns of the agent are not streamed."""
        events: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                result = answer(
                    body.question, user, st.searcher, st.llm, body.mode, st.cfg, st.cache,
                    on_text=lambda text: events.put(("delta", {"text": text})),
                ).to_dict()  # fmt: skip
                st.log.add(user.login, body.question, result)
                events.put(("done", visible_result(result, user)))
            except Exception:  # the client gets an error event instead of a hanging stream
                log.exception("streaming answer failed")
                events.put(("error", {"detail": "Antwort fehlgeschlagen"}))

        def generate():
            threading.Thread(target=work, daemon=True).start()
            while True:
                kind, payload = events.get()
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if kind != "delta":
                    break

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    def titles_for(user: User, st: AppState) -> dict[str, tuple[str, str]]:
        """doc_id -> (title, department) of every document this user may see."""
        return {
            c.doc_id: (c.title, c.department)
            for c in st.store.get_chunks(allowed_departments(user))
        }

    @app.get("/documents")
    def list_documents(user: Current, st: St) -> list[dict]:
        docs = titles_for(user, st)
        return [
            {"doc_id": d, "title": t, "department": dep} for d, (t, dep) in sorted(docs.items())
        ]

    @app.get("/documents/{doc_id:path}")
    def get_document(doc_id: str, user: Current, st: St) -> dict:
        # the store filters by department: forbidden and missing look exactly the same
        chunks = st.store.get_document_chunks(doc_id, allowed_departments(user))
        if not chunks:
            raise HTTPException(404, NOT_FOUND_DETAIL)
        return {
            "doc_id": doc_id, "title": chunks[0].title, "department": chunks[0].department,
            "chunks": [
                {"chunk_id": c.chunk_id, "heading": c.heading, "text": c.text} for c in chunks
            ],
        }  # fmt: skip

    @app.post("/documents", status_code=201)
    def upload_document(
        _admin: Admin, st: St, file: Annotated[UploadFile, File()],
        department: Annotated[str, Form()] = "alle",
    ) -> dict:  # fmt: skip
        if department not in DEPARTMENTS:
            raise HTTPException(422, f"Unbekannte Abteilung (erlaubt: {', '.join(DEPARTMENTS)})")
        original = Path(file.filename or "").name
        suffix = Path(original).suffix.lower()
        stem = SAFE_STEM.sub("-", Path(original).stem).strip("-").lower()
        if suffix not in LOADERS or not stem:
            raise HTTPException(422, "Nur .md, .pdf und .docx sind erlaubt")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Datei zu groß (max. 5 MB)")
        with st.lock:
            target = st.docs_dir / department / f"{stem}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:  # parse first: a broken file changes nothing
                candidate = Path(tmp) / target.name
                candidate.write_bytes(data)
                try:
                    doc = load_document(candidate, department)
                except Exception:
                    raise HTTPException(422, "Datei konnte nicht gelesen werden") from None
                shutil.copyfile(candidate, target)
            stats = upsert_documents([doc], st.store, st.embedder, st.manifest)
            refresh_after_change(st)
            if isinstance(st.store, PgVectorStore):
                save_manifest(st.index_dir, st.manifest)
        return {"doc_id": doc.doc_id, "title": doc.title, "department": department,
                "chunks_embedded": stats.chunks_embedded}  # fmt: skip

    @app.delete("/documents/{doc_id:path}")
    def delete_document(doc_id: str, _admin: Admin, st: St) -> dict:
        everything = frozenset(DEPARTMENTS)
        with st.lock:
            if not st.store.get_document_chunks(doc_id, everything):
                raise HTTPException(404, NOT_FOUND_DETAIL)
            department = doc_id.split("/")[0]
            for path in (st.docs_dir / department).glob(f"{doc_id.split('/', 1)[1]}.*"):
                if path.suffix.lower() in LOADERS:
                    path.unlink()
            st.store.remove_documents([doc_id])
            st.manifest.pop(doc_id, None)
            refresh_after_change(st)
            if isinstance(st.store, PgVectorStore):
                save_manifest(st.index_dir, st.manifest)
        return {"deleted": doc_id}

    @app.get("/metrics")
    def metrics(_admin: Admin, st: St) -> dict:
        # admin only: the list of unanswered questions can reveal what other departments ask
        return {
            **st.log.summary(),
            "eval": _read_json(RESULTS),
            "scale": _read_json(SCALE_JSON),
        }

    @app.get("/users")
    def users() -> list[dict]:
        """Demo users for the login screen (no secrets involved)."""
        return [{"login": u.login, "name": u.name} for u in USERS.values()]

    return app


app = create_app()
