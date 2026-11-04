"""Load documents (md, pdf, docx) and split them into heading-aware chunks."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TOKENS_PER_WORD = 1.3  # rough German average, good enough for sizing chunks


@dataclass(frozen=True)
class Document:
    doc_id: str  # "<department>/<file stem>", e.g. "hr/gehaltsbaender"
    title: str
    department: str
    text: str  # markdown-like: headings start with "#"
    source: str  # file name, e.g. "gehaltsbaender.md"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<position>"
    doc_id: str
    title: str
    department: str
    heading: str
    text: str
    position: int


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


# ---------------------------------------------------------------- loading


def _load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_docx(path: Path) -> str:
    """Convert Word heading styles to markdown headings."""
    lines: list[str] = []
    for para in DocxDocument(str(path)).paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style == "Title":
            lines.append(f"# {text}")
        elif style.startswith("Heading "):
            level = int(style.split()[-1])
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _load_pdf(path: Path) -> str:
    """Extract text; text set in a bold font is treated as a heading (our own PDFs do this)."""
    parts: list[str] = []

    def visitor(text: str, cm, tm, font_dict, font_size) -> None:
        if not text.strip():
            return
        base_font = str((font_dict or {}).get("/BaseFont", ""))
        if "Bold" in base_font:
            parts.append(f"\n\n## {text.strip()}\n\n")
        else:
            parts.append(text)

    for page in PdfReader(str(path)).pages:
        page.extract_text(visitor_text=visitor)
        parts.append("\n")
    raw = "".join(parts)
    # body lines are wrapped by the PDF; join them back into paragraphs
    paragraphs = []
    for block in re.split(r"\n\s*\n", raw):
        block = " ".join(block.split())
        if block:
            paragraphs.append(block)
    return "\n\n".join(paragraphs)


LOADERS = {".md": _load_md, ".docx": _load_docx, ".pdf": _load_pdf}


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            return m.group(2)
    return fallback


def load_document(path: Path, department: str) -> Document:
    loader = LOADERS[path.suffix.lower()]
    text = loader(path)
    # PDF: the first bold text is the title; promote it to a level-1 heading
    if path.suffix.lower() == ".pdf":
        text = re.sub(r"^## ", "# ", text, count=1)
    return Document(
        doc_id=f"{department}/{path.stem}",
        title=_title_of(text, path.stem),
        department=department,
        text=text,
        source=path.name,
    )


def load_documents(root: Path) -> list[Document]:
    """Load every supported file under root/<department>/."""
    docs = []
    for dept_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(dept_dir.iterdir()):
            if path.suffix.lower() in LOADERS:
                docs.append(load_document(path, dept_dir.name))
    return docs


# ---------------------------------------------------------------- chunking


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs. Sections without body are dropped."""
    sections: list[tuple[str, str]] = []
    heading, body = "", []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if "\n".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading, body = m.group(2), []
        else:
            body.append(line)
    if "\n".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections


def _split_long(paragraph: str, max_words: int, overlap_words: int) -> list[str]:
    """Cut a paragraph that is longer than one chunk into overlapping word windows."""
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]
    step = max_words - overlap_words
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap_words, step)]


def _pack(paragraphs: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Greedily pack paragraphs into chunks; each new chunk starts with the previous tail."""
    pieces = [p for para in paragraphs for p in _split_long(para, max_words, overlap_words)]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        n = len(piece.split())
        if current and current_words + n > max_words:
            joined = "\n\n".join(current)
            chunks.append(joined)
            tail = " ".join(joined.split()[-overlap_words:])
            current, current_words = [tail], len(tail.split())
        current.append(piece)
        current_words += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, max_tokens: int = 500, overlap_tokens: int = 60) -> list[Chunk]:
    max_words = int(max_tokens / TOKENS_PER_WORD)
    overlap_words = int(overlap_tokens / TOKENS_PER_WORD)
    chunks: list[Chunk] = []
    for heading, body in split_sections(doc.text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for text in _pack(paragraphs, max_words, overlap_words):
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{position}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    department=doc.department,
                    heading=heading or doc.title,
                    text=text,
                    position=position,
                )
            )
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, **kwargs)]


def embedding_text(chunk: Chunk) -> str:
    """Text that is embedded / keyword-indexed: title and heading give the chunk context."""
    return f"{chunk.title} - {chunk.heading}\n{chunk.text}"
