from pathlib import Path

from ks.ingest import (
    Document,
    approx_tokens,
    chunk_document,
    load_document,
    split_sections,
)

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_corpus_size_and_formats(documents):
    assert len(documents) == 35
    suffixes = [d.source.rsplit(".", 1)[1] for d in documents]
    assert suffixes.count("pdf") == 3
    assert suffixes.count("docx") == 2
    assert suffixes.count("md") == 30
    assert {d.department for d in documents} == {"alle", "hr", "it", "vertrieb", "logistik"}


def test_document_length(documents):
    for d in documents:
        assert 300 <= len(d.text.split()) <= 1200, d.doc_id


def test_load_markdown():
    doc = load_document(DOCS / "hr" / "gehaltsbaender.md", "hr")
    assert doc.doc_id == "hr/gehaltsbaender"
    assert doc.title.startswith("Gehaltsbänder")
    assert "E5" in doc.text


def test_load_pdf_detects_headings():
    doc = load_document(DOCS / "logistik" / "zollabwicklung.pdf", "logistik")
    assert doc.title == "Zollabwicklung"
    headings = [h for h, _ in split_sections(doc.text)]
    assert "Gebühren" in headings
    assert "45 Euro" in doc.text


def test_load_docx_detects_headings():
    doc = load_document(DOCS / "hr" / "probezeit-feedback.docx", "hr")
    assert doc.title.startswith("Leitfaden Probezeit")
    assert "Gesprächstermine" in [h for h, _ in split_sections(doc.text)]
    assert "sechs Monate" in doc.text


def test_chunks_carry_metadata(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.chunk_id.startswith(c.doc_id + "#")
        assert c.doc_id.startswith(c.department + "/")
        assert c.title and c.heading and c.text


def test_chunks_follow_headings():
    doc = Document(
        "x/doc", "Titel", "x", "# Titel\n\n## A\n\nAlpha text.\n\n## B\n\nBeta text.", "doc.md"
    )
    chunks = chunk_document(doc)
    assert [(c.heading, c.text) for c in chunks] == [("A", "Alpha text."), ("B", "Beta text.")]


def test_long_section_is_split_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    paragraphs = [" ".join(words[i : i + 50]) for i in range(0, 1000, 50)]
    doc = Document(
        "x/long", "Lang", "x", "# Lang\n\n## Abschnitt\n\n" + "\n\n".join(paragraphs), "l.md"
    )
    chunks = chunk_document(doc, max_tokens=500, overlap_tokens=60)
    assert len(chunks) >= 3
    assert all(approx_tokens(c.text) <= 500 + 60 for c in chunks)
    assert {c.heading for c in chunks} == {"Abschnitt"}
    # overlap: the start of chunk n+1 repeats the end of chunk n
    tail = chunks[0].text.split()[-5:]
    assert all(w in chunks[1].text.split()[:60] for w in tail)
    # nothing lost
    assert {w for c in chunks for w in c.text.split()} == set(words)
