"""HTML -> clean, table-aware plain text.

10-K filings (especially Workiva/XBRL-generated ones like Apple's) bury the
real content in deeply nested <div>/<span> soup plus large financial <table>s.
Naive ``get_text()`` either glues paragraphs together or shatters tables into
meaningless token streams.

We do two things that matter for downstream QA quality:

1. Render every <table> into a readable "Label | col1 | col2" text block so the
   numbers stay associated with their row labels. A huge share of the most
   valuable questions (segment revenue, year-over-year deltas) live in tables.
2. Insert newlines only at *block* boundaries so paragraphs stay intact while
   inline <span>s concatenate naturally.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

try:  # bs4 warns when an XML (XBRL) doc is fed to the HTML parser; harmless here.
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:  # pragma: no cover - older bs4
    pass

from .utils import normalize_ws, squash

BLOCK_TAGS = {
    "p", "div", "li", "tr", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote",
}
DROP_TAGS = {"script", "style", "head", "noscript"}


@dataclass
class ParsedDocument:
    text: str
    n_tables: int
    n_chars: int


def _render_table(table: Tag) -> str:
    """Turn a <table> into 'cell | cell | cell' rows separated by newlines."""
    lines = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        values = []
        for cell in cells:
            txt = squash(cell.get_text(" ", strip=True))
            if txt:
                values.append(txt)
        if values:
            lines.append(" | ".join(values))
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _decode(raw: bytes) -> str:
    """Decode filing bytes robustly.

    Many SEC/Workiva filings declare ``encoding='ASCII'`` in the XML prolog but
    actually contain UTF-8 (curly quotes, em dashes). A few older filings are
    Windows-1252. We therefore ignore the declaration and try real encodings in
    order of likelihood.
    """
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_html(html) -> ParsedDocument:
    if isinstance(html, (bytes, bytearray)):
        html = _decode(bytes(html))
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()

    # Render tables first, then replace them with their text rendering so the
    # later block-boundary pass does not re-fragment them.
    tables = soup.find_all("table")
    n_tables = 0
    for table in tables:
        rendered = _render_table(table)
        if rendered.strip():
            n_tables += 1
        table.replace_with(NavigableString(rendered))

    # Mark block boundaries with newlines (headings get an extra one).
    for tag in soup.find_all(True):
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            tag.insert_after(NavigableString("\n\n"))
        elif tag.name == "br":
            tag.replace_with(NavigableString("\n"))
        elif tag.name in BLOCK_TAGS:
            tag.insert_after(NavigableString("\n"))

    raw = soup.get_text()
    text = normalize_ws(raw)
    return ParsedDocument(text=text, n_tables=n_tables, n_chars=len(text))


def parse_file(path: str) -> ParsedDocument:
    with open(path, "rb") as f:
        return parse_html(f.read())
