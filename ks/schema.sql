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
    embedding  vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_department_idx ON chunks (department);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
