-- Schema for PgVectorStore. {dim} is filled in by Python (must match the embedder).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   text PRIMARY KEY,
    doc_id     text NOT NULL,
    title      text NOT NULL,
    department text NOT NULL,
    heading    text NOT NULL,
    text       text NOT NULL,
    position   integer NOT NULL,
    valid_from text NOT NULL DEFAULT '',
    superseded boolean NOT NULL DEFAULT false,
    policy     text NOT NULL DEFAULT '',
    embedding  vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_department_idx ON chunks (department);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);

-- Approximate nearest-neighbour index for large corpora (cosine distance, like the queries use).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
