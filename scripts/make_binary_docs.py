"""Generate the sample .pdf and .docx documents from scripts/binary_sources/*.md.

Reproducible: run `python scripts/make_binary_docs.py` and commit the result.
Source file name: <department>__<name>.md  ->  data/docs/<department>/<name>.<pdf|docx>
Needs the dev extras (reportlab) and python-docx.
"""

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "scripts" / "binary_sources"
OUT = ROOT / "data" / "docs"

# which source becomes which format
FORMATS = {
    "alle__datenschutz-grundlagen": "pdf",
    "it__datensicherung": "pdf",
    "logistik__zollabwicklung": "pdf",
    "hr__probezeit-feedback": "docx",
    "vertrieb__vertragsvorlagen": "docx",
}


def parse(md: str) -> list[tuple[str, str]]:
    """Return [(kind, text)] with kind in h1, h2, p."""
    items = []
    for block in md.split("\n\n"):
        block = block.strip()
        if block.startswith("## "):
            items.append(("h2", block[3:]))
        elif block.startswith("# "):
            items.append(("h1", block[2:]))
        elif block:
            items.append(("p", " ".join(block.split())))
    return items


def write_pdf(items: list[tuple[str, str]], path: Path) -> None:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold")
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold")
    flow = []
    for kind, text in items:
        style = {"h1": h1, "h2": h2, "p": body}[kind]
        flow.append(Paragraph(escape(text), style))
        flow.append(Spacer(1, 0.2 * cm))
    # invariant=1 -> no timestamps/ids, same bytes on every run
    SimpleDocTemplate(str(path), pagesize=A4, invariant=1, title=items[0][1],
                      author="Elbkontor Logistik GmbH").build(flow)  # fmt: skip


def write_docx(items: list[tuple[str, str]], path: Path) -> None:
    doc = Document()
    for kind, text in items:
        if kind == "h1":
            doc.add_heading(text, level=1)
        elif kind == "h2":
            doc.add_heading(text, level=2)
        else:
            doc.add_paragraph(text)
    doc.core_properties.author = "Elbkontor Logistik GmbH"
    doc.core_properties.created = datetime(2026, 1, 1)
    doc.core_properties.modified = datetime(2026, 1, 1)
    doc.save(str(path))


def main() -> None:
    for source in sorted(SOURCES.glob("*.md")):
        key = source.stem
        department, name = key.split("__", 1)
        fmt = FORMATS[key]
        target = OUT / department / f"{name}.{fmt}"
        target.parent.mkdir(parents=True, exist_ok=True)
        items = parse(source.read_text(encoding="utf-8"))
        (write_pdf if fmt == "pdf" else write_docx)(items, target)
        print("wrote", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
