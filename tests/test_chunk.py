"""Tests for section detection and chunking (the deterministic core)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_pipeline.config import Config
from qa_pipeline.chunk import split_sections, chunk_document, chunk_section_text
from qa_pipeline.parse import parse_html


SAMPLE = """\
Cover page boilerplate.

Item 1. | Business | 1
Item 1A. | Risk Factors | 5
Item 7. | Management's Discussion | 20

Item 1. Business
The Company designs and sells smartphones and personal computers.
Net sales for the year were $400,000 million.

Item 1A. Risk Factors
The Company's business is subject to global and economic risks.
Adverse macroeconomic conditions could materially affect results.

Item 7. Management's Discussion
Total net sales increased 5% compared to the prior year.
Apple Inc. | 2025 Form 10-K | 20
"""


def test_split_sections_ignores_toc_rows():
    sections = split_sections(SAMPLE)
    items = [item for item, _, _ in sections]
    # TOC rows contain '|' and must not be treated as real headers.
    assert items == ["1", "1A", "7"], items
    # Real Business section body should contain the net sales sentence.
    body = dict((i, b) for i, _, b in sections)["1"]
    assert "smartphones" in body
    assert "Item 1A" not in body  # boundary respected


def test_footer_is_stripped():
    sections = dict((i, b) for i, _, b in split_sections(SAMPLE))
    assert "Form 10-K | 20" not in sections["7"]


def test_chunk_document_labels_sections():
    cfg = Config()
    cfg.min_chunk_chars = 10  # tiny sample
    chunks = chunk_document(SAMPLE, cfg)
    names = {c.section_name for c in chunks}
    assert "Business" in names
    assert "Risk Factors" in names
    # every chunk carries an item id and non-empty text
    assert all(c.section_item and c.text for c in chunks)


def test_chunk_packing_respects_max():
    cfg = Config()
    cfg.max_chunk_chars = 200
    big = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(40))
    pieces = chunk_section_text(big, cfg)
    assert len(pieces) > 1
    assert all(len(p) <= cfg.max_chunk_chars + cfg.chunk_overlap_chars for p in pieces)


def test_parse_table_to_pipes():
    html = "<table><tr><td>Products</td><td>$112,887</td></tr></table>"
    doc = parse_html(html)
    assert "Products | $112,887" in doc.text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all chunk tests passed")
