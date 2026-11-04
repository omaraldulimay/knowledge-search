"""Generate a large synthetic filler corpus for the scale benchmark. No LLM, no network.

    python scripts/gen_corpus.py --n 10000 [--seed 42] [--out data/generated/docs]

Document i depends only on (seed, i), so a smaller corpus is a prefix of a bigger one and a
re-run with the same arguments writes byte-identical files (the incremental index then sees
nothing to do). All names and companies are invented.

Two kinds of documents:
- filler: delivery notes, shift logs, project notes, meeting minutes, customer letters
- distractors: outdated versions of real policies (same wording, other numbers, front matter
  `superseded: true`) and near-duplicate FAQ pages built from real policy sections
"""

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ks.ingest import parse_front_matter  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "generated" / "docs"
REAL_DOCS = ROOT / "data" / "docs"

# real policies that get outdated versions: (department, file stem)
POLICIES = [
    ("alle", "reisekostenrichtlinie"),
    ("vertrieb", "vertrieb-reisekosten"),
    ("it", "passwortrichtlinie"),
    ("alle", "urlaubsantrag"),
    ("alle", "arbeitszeit-homeoffice"),
    ("vertrieb", "rabattstaffel-grosskunden"),
]
OLD_YEARS = (2019, 2022)  # two outdated versions per policy

FIRST = ["Jonas", "Lena", "Malte", "Svenja", "Torben", "Imke", "Henrik", "Greta", "Niklas",
         "Frauke", "Jörn", "Marit", "Kai", "Silke", "Lasse", "Wiebke", "Dirk", "Anneke",
         "Rune", "Birte", "Tjark", "Hilke", "Enno", "Maren", "Ole", "Jette"]  # fmt: skip
LAST = ["Hartwig", "Petersen", "Lübbers", "Sandmann", "Wiechmann", "Brodersen", "Thiele",
        "Marquardt", "Ohlsen", "Rickmers", "Seeger", "Tiedemann", "Ahlers", "Boysen",
        "Dethlefs", "Fedder", "Gerken", "Harms", "Ipsen", "Jessen"]  # fmt: skip
CO_A = ["Alster", "Marsch", "Düne", "Fjord", "Möwen", "Kranich", "Tiefwasser", "Sandbank",
        "Leuchtfeuer", "Hafenkante", "Sturmflut", "Gezeiten", "Wattwurm", "Salzwiese"]  # fmt: skip
CO_B = ["Metallbau", "Textil", "Getränke", "Möbel", "Elektro", "Farben", "Papier", "Agrar",
        "Kunststoff", "Werkzeug", "Backwaren", "Fahrrad", "Verpackung", "Baustoffe"]  # fmt: skip
CO_C = ["GmbH", "AG", "KG", "GmbH & Co. KG", "e. K.", "UG"]
CITIES = ["Bremen", "Kiel", "Lübeck", "Rostock", "Hannover", "Oldenburg", "Flensburg",
          "Schwerin", "Stade", "Cuxhaven", "Hamburg", "Osnabrück"]  # fmt: skip
GOODS = ["Europaletten", "Kartons", "Gitterboxen", "Fässer", "Rollcontainer", "Kisten",
         "Big Bags", "Ersatzteilpakete", "Textilrollen", "Kühlware"]  # fmt: skip
HALLS = ["Halle 1", "Halle 2", "Halle 3", "Halle 4", "Rampe Nord", "Rampe Süd", "Außenlager"]
TOPICS = ["Tourenplanung", "Rampenbelegung", "Reklamationsquote", "Fuhrparkwartung",
          "Neukundenanfrage", "Angebotsvorlage", "Passwortwechsel", "Homeoffice-Regelung",
          "Urlaubsplanung Sommer", "Hotelbuchung Messe", "Lagerinventur", "Schulungsplan",
          "Zolldokumente", "Gefahrgut-Unterweisung", "Rabattanfrage Großkunde"]  # fmt: skip
STATUS = ["erledigt", "in Arbeit", "wartet auf Rückmeldung", "verschoben", "offen"]

# department -> (weight, document kinds); the mix of a mid-sized logistics company
DEPT_MIX = {
    "logistik": (0.40, ["lieferschein", "schichtprotokoll"]),
    "vertrieb": (0.22, ["kundenkorrespondenz", "projektnotiz"]),
    "it": (0.14, ["projektnotiz", "meeting"]),
    "hr": (0.09, ["meeting", "projektnotiz"]),
    "alle": (0.15, ["meeting", "projektnotiz"]),
}


def _person(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _company(r: random.Random) -> str:
    return f"{r.choice(CO_A)}{r.choice(CO_B).lower()} {r.choice(CO_C)}"


def _date(r: random.Random, first_year: int = 2023, last_year: int = 2026) -> str:
    return f"{r.randint(first_year, last_year)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"


def _front(valid_from: str, extra: str = "") -> str:
    return f"---\nvalid_from: {valid_from}\nsuperseded: false\n{extra}---\n"


# ------------------------------------------------------------------ filler templates


def lieferschein(r: random.Random, i: int) -> tuple[str, str, str]:
    nr = f"LS-{r.randint(2023, 2026)}-{r.randint(10000, 99999)}"
    date = _date(r)
    rows = "\n".join(
        f"| {r.randint(1, 40)} | {r.choice(GOODS)} | {r.randint(40, 2200)} kg |"
        for _ in range(r.randint(2, 5))
    )
    body = f"""# Lieferschein {nr}

Ausstellungsdatum: {date}

## Empfänger

{_company(r)}, {r.randint(1, 120)} {r.choice(LAST)}straße, {r.choice(CITIES)}. Ansprechperson vor Ort:
{_person(r)}. Anlieferung an {r.choice(HALLS)} im Zeitfenster {r.randint(6, 15):02d}:00 bis
{r.randint(16, 19):02d}:00 Uhr.

## Positionen

| Menge | Ware | Gewicht |
|---|---|---|
{rows}

## Transport

Fahrer: {_person(r)}, Kennzeichen HH-{r.choice("ABCDEFGHKLMNPRST")}{r.choice("ABCDEFGHKLMNPRST")} {r.randint(100, 9999)}.
Die Ware wurde {r.choice(["vollständig", "mit einer Teilmenge", "ohne Beanstandung"])} übergeben.
Empfangsbestätigung durch {_person(r)}. Bemerkung: {r.choice(["keine", "Palette 3 leicht beschädigt, fotografiert", "Rampe war belegt, 20 Minuten Wartezeit", "Ware nach Absprache an Nebeneingang gestellt"])}.
"""
    return "logistik", f"lieferschein-{i:05d}", body


def schichtprotokoll(r: random.Random, i: int) -> tuple[str, str, str]:
    shift = r.choice(["Frühschicht", "Spätschicht", "Nachtschicht"])
    body = f"""# Schichtprotokoll {shift} {_date(r)}

## Besetzung

Schichtleitung: {_person(r)}. Im Einsatz waren {r.randint(6, 24)} Mitarbeitende, davon
{r.randint(0, 3)} Aushilfen. Krankmeldungen: {r.randint(0, 4)}.

## Durchsatz

In {r.choice(HALLS)} wurden {r.randint(80, 640)} Paletten eingelagert und {r.randint(80, 640)}
Paletten ausgelagert. Es gab {r.randint(0, 5)} Staplerstörungen und {r.randint(0, 3)} Abweichungen
bei der Inventur.

## Besondere Vorkommnisse

{r.choice(["Keine besonderen Vorkommnisse.", "Ein Gabelstapler wurde wegen defekter Hydraulik gesperrt und der Werkstatt gemeldet.", "Ladungssicherung bei einem Fahrzeug nachgebessert, der Fahrer wurde unterwiesen.", "Kühlcontainer meldete kurzzeitig zu hohe Temperatur, Ware wurde geprüft und freigegeben."])}

## Übergabe

Offene Punkte für die nächste Schicht: {r.choice(TOPICS)}, {r.choice(TOPICS)}.
"""
    return "logistik", f"schichtprotokoll-{i:05d}", body


def projektnotiz(r: random.Random, i: int, dept: str) -> tuple[str, str, str]:
    topic = r.choice(TOPICS)
    people = ", ".join(_person(r) for _ in range(3))
    body = f"""# Projektnotiz: {topic}

Stand: {_date(r)}. Verantwortlich: {_person(r)}.

## Ausgangslage

Im Zusammenhang mit dem Thema {topic} wurde festgestellt, dass der Aufwand höher ist als
geplant. Beteiligt sind {people}. Das Budget liegt bei {r.randint(2, 90)}.000 Euro.

## Nächste Schritte

- {r.choice(TOPICS)} klären ({r.choice(STATUS)})
- Abstimmung mit {_company(r)} ({r.choice(STATUS)})
- Ergebnis bis {_date(r, 2026, 2026)} dokumentieren ({r.choice(STATUS)})

## Risiken

{r.choice(["Termin ist eng, wenn der Lieferant nicht rechtzeitig liefert.", "Abhängigkeit von einer einzelnen Person im Team.", "Die Kosten können steigen, falls die Anforderungen wachsen.", "Keine kritischen Risiken bekannt."])}
"""
    return dept, f"projektnotiz-{i:05d}", body


def meeting(r: random.Random, i: int, dept: str) -> tuple[str, str, str]:
    body = f"""# Meeting-Protokoll {dept.upper()} {_date(r)}

Teilnehmende: {_person(r)}, {_person(r)}, {_person(r)}, {_person(r)}. Protokoll: {_person(r)}.

## Themen

1. {r.choice(TOPICS)}: {r.choice(["Status besprochen, keine Änderung.", "Entscheidung vertagt.", "Maßnahme beschlossen.", "Rückfrage an die Geschäftsführung."])}
2. {r.choice(TOPICS)}: {r.choice(["Status besprochen, keine Änderung.", "Entscheidung vertagt.", "Maßnahme beschlossen.", "Rückfrage an die Geschäftsführung."])}

## Beschlüsse

Das Team beschließt, {r.choice(TOPICS)} bis {_date(r, 2026, 2026)} abzuschließen. Zuständig ist
{_person(r)}. Das nächste Treffen findet in {r.randint(1, 6)} Wochen statt.
"""
    return dept, f"meeting-{i:05d}", body


def kundenkorrespondenz(r: random.Random, i: int) -> tuple[str, str, str]:
    company = _company(r)
    body = f"""# Kundenkorrespondenz {company}

Datum: {_date(r)}. Bearbeitung: {_person(r)}.

## Anfrage des Kunden

{_person(r)} von {company} fragt an, ob eine Lieferung von {r.randint(2, 60)} {r.choice(GOODS)}
nach {r.choice(CITIES)} bis zum {_date(r, 2026, 2026)} möglich ist.

## Antwort

Vielen Dank für Ihre Anfrage. Wir können die Lieferung {r.choice(["wie gewünscht", "mit einem Tag Verzögerung", "nur als Teillieferung"])} durchführen.
Das Angebot Nr. {r.randint(1000, 9999)} sendet Ihnen {_person(r)} gesondert zu.

## Ergebnis

{r.choice(["Auftrag erteilt.", "Kunde prüft das Angebot.", "Kunde hat abgesagt.", "Rückruf vereinbart."])}
"""
    return "vertrieb", f"kundenkorrespondenz-{i:05d}", body


# ------------------------------------------------------------------ distractors


def _perturb_numbers(text: str, r: random.Random) -> str:
    """Change every number of 2-4 digits (not years, not parts of dates) by a random factor."""

    def change(m: re.Match) -> str:
        n = int(m.group(0))
        if n < 10 or 1900 <= n <= 2100:
            return m.group(0)
        return str(max(1, round(n * r.choice([0.6, 0.75, 0.8, 1.25, 1.5]))))

    return re.sub(r"(?<![\d.,])\d{2,4}(?![\d.,])", change, text)


def outdated_policy(dept: str, stem: str, year: int, r: random.Random) -> tuple[str, str, str]:
    """Same text as the real policy, but old title, old date, other numbers, superseded."""
    meta, text = parse_front_matter((REAL_DOCS / dept / f"{stem}.md").read_text(encoding="utf-8"))
    text = re.sub(r"\d{2}\.\d{2}\.\d{4}", f"01.01.{year}", text)  # dates first, then numbers
    text = _perturb_numbers(text, r)
    first_line, _, rest = text.lstrip().partition("\n")
    title = first_line.lstrip("# ").strip()
    text = f"# {title} (Stand {year})\n\nHinweis: Diese Fassung ist {year} in Kraft getreten.{rest}"
    front = (f"---\npolicy: {meta['policy']}\nvalid_from: {year}-01-01\n"
             f"superseded: true\n---\n")  # fmt: skip
    return dept, f"{stem}-stand-{year}", front + text


def faq_near_duplicate(
    dept: str, stem: str, variant: int, r: random.Random
) -> tuple[str, str, str]:
    """FAQ page that re-phrases 2-3 sections of a real policy: similar words, no policy key."""
    _, text = parse_front_matter((REAL_DOCS / dept / f"{stem}.md").read_text(encoding="utf-8"))
    title = text.lstrip().split("\n", 1)[0].lstrip("# ").strip()
    sections = [s for s in re.split(r"\n(?=## )", text) if s.startswith("## ")]
    picked = r.sample(sections, k=min(len(sections), r.randint(2, 3)))
    parts = [
        f"# FAQ: {title} (Variante {variant})\n\nHäufige Fragen aus dem Team, kurz beantwortet."
    ]
    for sec in picked:
        head, _, body = sec.partition("\n")
        sentences = re.split(r"(?<=[a-zäöüß][.!?]) ", " ".join(body.split()))
        parts.append(
            f"## Frage zu: {head.lstrip('# ')}\n\n" + " ".join(sentences[: r.randint(1, 3)])
        )
    return dept, f"faq-{stem}-{variant}", _front(_date(r, 2024, 2026)) + "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------ driver


def make_document(seed: int, i: int) -> tuple[str, str, str]:
    """Filler document number i -> (department, file stem, markdown)."""
    r = random.Random(f"{seed}-{i}")
    dept = r.choices(list(DEPT_MIX), weights=[w for w, _ in DEPT_MIX.values()])[0]
    kind = r.choice(DEPT_MIX[dept][1])
    if kind == "lieferschein":
        d, stem, body = lieferschein(r, i)
    elif kind == "schichtprotokoll":
        d, stem, body = schichtprotokoll(r, i)
    elif kind == "kundenkorrespondenz":
        d, stem, body = kundenkorrespondenz(r, i)
    elif kind == "meeting":
        d, stem, body = meeting(r, i, dept)
    else:
        d, stem, body = projektnotiz(r, i, dept)
    return d, stem, _front(_date(r)) + body


def distractors(seed: int, limit: int) -> list[tuple[str, str, str]]:
    """Outdated policy versions first, then FAQ near-duplicates (at most `limit` documents)."""
    r = random.Random(f"{seed}-distractors")
    docs = [outdated_policy(d, s, y, r) for d, s in POLICIES for y in OLD_YEARS]
    docs += [faq_near_duplicate(d, s, v, r) for d, s in POLICIES for v in (1, 2, 3)]
    return docs[:limit]


def generate(n: int, seed: int = 42, out: Path = DEFAULT_OUT) -> int:
    """Write n documents (distractors included) to out/<department>/. Returns files written."""
    wanted: dict[Path, str] = {}
    specials = distractors(seed, limit=max(0, n // 4))  # distractors are a few dozen at most
    for dept, stem, text in specials:
        wanted[out / dept / f"{stem}.md"] = text
    for i in range(n - len(specials)):
        dept, stem, text = make_document(seed, i)
        wanted[out / dept / f"{stem}.md"] = text
    if out.exists():  # remove files of an earlier, different run
        for old in out.rglob("*.md"):
            if old not in wanted:
                old.unlink()
    for path, text in wanted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    return len(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 1 <= args.n <= 50000:
        parser.error("--n must be between 1 and 50000")
    print(f"Wrote {generate(args.n, args.seed, args.out)} documents to {args.out}")


if __name__ == "__main__":
    main()
