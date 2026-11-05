"""Vector stores. Every read method takes `allowed` (departments) and applies it while
selecting candidates - never only after ranking. An empty `allowed` set returns nothing."""

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol

import numpy as np

from ks.ingest import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def clear(self) -> None: ...

    def remove_documents(self, doc_ids: Collection[str]) -> None:
        """Delete every chunk of these documents (used for changed and deleted files)."""
        ...

    def count(self) -> int: ...

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]: ...

    def search_documents(
        self, query: np.ndarray, doc_ids: Collection[str], allowed: Collection[str]
    ) -> list[ScoredChunk]:
        """The best-matching chunk of each of these documents (only if the user may see it)."""
        ...

    def get_chunks(self, allowed: Collection[str]) -> list[Chunk]:
        """All chunks the user may see (used to build the keyword index)."""
        ...

    def get_document_chunks(self, doc_id: str, allowed: Collection[str]) -> list[Chunk]: ...


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class MemoryStore:
    """numpy-based store; cosine similarity = dot product of unit vectors."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._masks: dict[frozenset[str], np.ndarray] = {}  # allowed departments -> bool per row

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if not len(chunks):
            return
        vectors = _unit(vectors)
        self._matrix = vectors if not len(self._chunks) else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)
        self._masks.clear()

    def remove_documents(self, doc_ids: Collection[str]) -> None:
        drop = set(doc_ids)
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id not in drop]
        if len(keep) == len(self._chunks):
            return
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self._masks.clear()

    def clear(self) -> None:
        self.__init__()

    def count(self) -> int:
        return len(self._chunks)

    def _mask(self, allowed: Collection[str]) -> np.ndarray:
        """Boolean row mask of the chunks the user may see (cached per department set)."""
        key = frozenset(allowed)
        if key not in self._masks:
            self._masks[key] = np.array([c.department in key for c in self._chunks], dtype=bool)
        return self._masks[key]

    def _allowed_rows(self, allowed: Collection[str]) -> list[int]:
        return np.flatnonzero(self._mask(allowed)).tolist()

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        mask = self._mask(allowed)  # forbidden rows get -inf BEFORE ranking, so they never win
        if not mask.any():
            return []
        sims = np.where(mask, self._matrix @ _unit(query.reshape(1, -1))[0], -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-sims, k - 1)[:k]
        order = top[np.argsort(-sims[top])]
        return [ScoredChunk(self._chunks[i], float(sims[i])) for i in order]

    def search_documents(
        self, query: np.ndarray, doc_ids: Collection[str], allowed: Collection[str]
    ) -> list[ScoredChunk]:
        wanted = set(doc_ids)
        rows = [i for i in self._allowed_rows(allowed) if self._chunks[i].doc_id in wanted]
        if not rows:
            return []
        sims = self._matrix[rows] @ _unit(query.reshape(1, -1))[0]
        best: dict[str, ScoredChunk] = {}
        for i, sim in zip(rows, sims, strict=True):
            doc = self._chunks[i].doc_id
            if doc not in best or sim > best[doc].score:
                best[doc] = ScoredChunk(self._chunks[i], float(sim))
        return sorted(best.values(), key=lambda sc: -sc.score)

    def get_chunks(self, allowed: Collection[str]) -> list[Chunk]:
        return [self._chunks[i] for i in self._allowed_rows(allowed)]

    def get_document_chunks(self, doc_id: str, allowed: Collection[str]) -> list[Chunk]:
        return [c for c in self.get_chunks(allowed) if c.doc_id == doc_id]

    # --- persistence so that `index` and `search` can be separate CLI calls
    def save(self, directory: Path, embedder_name: str, **options) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(json.dumps({"embedder": embedder_name, **options}))

    @classmethod
    def load(cls, directory: Path) -> "MemoryStore":
        store = cls()
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store._chunks = [Chunk(**item) for item in raw]
        store._matrix = np.load(directory / "vectors.npy")
        if not store._chunks:
            store._matrix = np.zeros((0, 0), dtype=np.float32)
        return store

    @staticmethod
    def saved_embedder_name(directory: Path) -> str:
        return json.loads((directory / "meta.json").read_text())["embedder"]


def index_options(directory: Path) -> dict:
    """Everything saved in meta.json: the embedder name and how the text was prepared (headers)."""
    path = directory / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


class PgVectorStore:
    """Postgres + pgvector. The department filter is part of the SQL WHERE clause."""

    def __init__(self, dsn: str, dim: int) -> None:
        self.dsn = dsn
        self.dim = dim

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def init_schema(self, reset: bool = False) -> None:
        import psycopg

        sql = resources.files("ks").joinpath("schema.sql").read_text(encoding="utf-8")
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            if reset:
                conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute(sql.format(dim=int(self.dim)))

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = _unit(vectors)
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.title,
                c.department,
                c.heading,
                c.text,
                c.position,
                c.valid_from,
                c.superseded,
                c.policy,
                v,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks
                   (chunk_id, doc_id, title, department, heading, text, position,
                    valid_from, superseded, policy, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                     title = EXCLUDED.title, department = EXCLUDED.department,
                     heading = EXCLUDED.heading, text = EXCLUDED.text,
                     position = EXCLUDED.position, valid_from = EXCLUDED.valid_from,
                     superseded = EXCLUDED.superseded, policy = EXCLUDED.policy,
                     embedding = EXCLUDED.embedding""",
                rows,
            )

    def remove_documents(self, doc_ids: Collection[str]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (list(doc_ids),))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    @staticmethod
    def _chunk(row) -> Chunk:
        return Chunk(
            chunk_id=row[0], doc_id=row[1], title=row[2], department=row[3],
            heading=row[4], text=row[5], position=row[6],
            valid_from=row[7], superseded=row[8], policy=row[9],
        )  # fmt: skip

    @staticmethod
    def _iterative_scan(conn) -> None:
        """An HNSW index returns ~40 candidates and the department filter is applied afterwards,
        so a strict filter could leave fewer than k rows. pgvector >= 0.8 can keep scanning."""
        try:
            conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        except Exception:  # older pgvector: the setting does not exist
            conn.rollback()

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]:
        q = _unit(query.reshape(1, -1))[0]
        with self._connect() as conn:
            self._iterative_scan(conn)
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position,
                          valid_from, superseded, policy, 1 - (embedding <=> %s) AS score
                   FROM chunks WHERE department = ANY(%s)
                   ORDER BY embedding <=> %s LIMIT %s""",
                (q, list(allowed), q, k),
            ).fetchall()
        return [ScoredChunk(self._chunk(r), float(r[10])) for r in rows]

    def search_documents(
        self, query: np.ndarray, doc_ids: Collection[str], allowed: Collection[str]
    ) -> list[ScoredChunk]:
        q = _unit(query.reshape(1, -1))[0]
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (doc_id) chunk_id, doc_id, title, department, heading, text,
                          position, valid_from, superseded, policy, 1 - (embedding <=> %s) AS score
                   FROM chunks WHERE doc_id = ANY(%s) AND department = ANY(%s)
                   ORDER BY doc_id, embedding <=> %s""",
                (q, list(doc_ids), list(allowed), q),
            ).fetchall()
        return sorted(
            (ScoredChunk(self._chunk(r), float(r[10])) for r in rows), key=lambda x: -x.score
        )

    def get_chunks(self, allowed: Collection[str]) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position,
                          valid_from, superseded, policy
                   FROM chunks WHERE department = ANY(%s) ORDER BY chunk_id""",
                (list(allowed),),
            ).fetchall()
        return [self._chunk(r) for r in rows]

    def get_document_chunks(self, doc_id: str, allowed: Collection[str]) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position,
                          valid_from, superseded, policy
                   FROM chunks WHERE doc_id = %s AND department = ANY(%s) ORDER BY position""",
                (doc_id, list(allowed)),
            ).fetchall()
        return [self._chunk(r) for r in rows]
