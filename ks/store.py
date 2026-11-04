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

    def count(self) -> int: ...

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]: ...

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

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if not len(chunks):
            return
        vectors = _unit(vectors)
        self._matrix = vectors if not len(self._chunks) else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def clear(self) -> None:
        self.__init__()

    def count(self) -> int:
        return len(self._chunks)

    def _allowed_rows(self, allowed: Collection[str]) -> list[int]:
        return [i for i, c in enumerate(self._chunks) if c.department in allowed]

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]:
        rows = self._allowed_rows(allowed)  # candidate selection happens BEFORE scoring
        if not rows:
            return []
        sims = self._matrix[rows] @ _unit(query.reshape(1, -1))[0]
        order = np.argsort(-sims)[:k]
        return [ScoredChunk(self._chunks[rows[i]], float(sims[i])) for i in order]

    def get_chunks(self, allowed: Collection[str]) -> list[Chunk]:
        return [self._chunks[i] for i in self._allowed_rows(allowed)]

    def get_document_chunks(self, doc_id: str, allowed: Collection[str]) -> list[Chunk]:
        return [c for c in self.get_chunks(allowed) if c.doc_id == doc_id]

    # --- persistence so that `index` and `search` can be separate CLI calls
    def save(self, directory: Path, embedder_name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        (directory / "meta.json").write_text(json.dumps({"embedder": embedder_name}))

    @classmethod
    def load(cls, directory: Path) -> "MemoryStore":
        store = cls()
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store._chunks = [Chunk(**item) for item in raw]
        store._matrix = np.load(directory / "vectors.npy")
        return store

    @staticmethod
    def saved_embedder_name(directory: Path) -> str:
        return json.loads((directory / "meta.json").read_text())["embedder"]


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
            (c.chunk_id, c.doc_id, c.title, c.department, c.heading, c.text, c.position, v)
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks
                   (chunk_id, doc_id, title, department, heading, text, position, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                     title = EXCLUDED.title, department = EXCLUDED.department,
                     heading = EXCLUDED.heading, text = EXCLUDED.text,
                     position = EXCLUDED.position, embedding = EXCLUDED.embedding""",
                rows,
            )

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
        )  # fmt: skip

    def search_vector(
        self, query: np.ndarray, k: int, allowed: Collection[str]
    ) -> list[ScoredChunk]:
        q = _unit(query.reshape(1, -1))[0]
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position,
                          1 - (embedding <=> %s) AS score
                   FROM chunks WHERE department = ANY(%s)
                   ORDER BY embedding <=> %s LIMIT %s""",
                (q, list(allowed), q, k),
            ).fetchall()
        return [ScoredChunk(self._chunk(r), float(r[7])) for r in rows]

    def get_chunks(self, allowed: Collection[str]) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position
                   FROM chunks WHERE department = ANY(%s) ORDER BY chunk_id""",
                (list(allowed),),
            ).fetchall()
        return [self._chunk(r) for r in rows]

    def get_document_chunks(self, doc_id: str, allowed: Collection[str]) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT chunk_id, doc_id, title, department, heading, text, position
                   FROM chunks WHERE doc_id = %s AND department = ANY(%s) ORDER BY position""",
                (doc_id, list(allowed)),
            ).fetchall()
        return [self._chunk(r) for r in rows]
