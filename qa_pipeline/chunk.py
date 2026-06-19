"""Split a parsed 10-K into section-labelled chunks ready for QA generation.

Two-level strategy
------------------
1. **Section split.** A 10-K has a fixed skeleton of "Item N." sections
   (Business, Risk Factors, MD&A, Financial Statements, ...). We detect the
   *real* section headers and use them as hard boundaries. This gives every
   chunk a meaningful ``section`` label that flows straight into the dataset.

2. **Sub-chunking.** Long sections (Risk Factors, MD&A, the financial
   statements) are split further into <= ``max_chunk_chars`` windows on
   paragraph/table boundaries, with a small overlap so a fact split across a
   boundary is still answerable from at least one chunk.

Detecting the *real* header (not the table-of-contents row) is the only subtle
part: in the rendered text, TOC rows look like ``Item 1. | Business | 1`` (they
came from a <table>, so they contain ``|``), whereas the real section header is
just ``Item 1. Business``. Requiring the title to contain no ``|`` cleanly
separates the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

from .config import Config, DEFAULT, TEN_K_SECTIONS, LOW_SIGNAL_SECTIONS
from .utils import estimate_tokens, get_logger, stable_id

log = get_logger()

# Real section header: "Item 7A. Management's Discussion ..." with a title that
# contains no pipe (which would mark it as a table-of-contents row).
SECTION_RE = re.compile(
    r"(?im)^Item\s+(\d{1,2}[A-Z]?)\s*\.\s+([^\n|]{2,120})$"
)

# Recurring page footer, e.g. "Apple Inc. | 2025 Form 10-K | 20".
FOOTER_RE = re.compile(r"(?im)^.*\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+\s*$")

# Content that signals an effectively empty section.
EMPTY_RE = re.compile(r"^(none|not applicable|\[reserved\])\.?$", re.I)


@dataclass
class Chunk:
    chunk_id: str
    section_item: str
    section_name: str
    part_index: int
    text: str
    n_chars: int
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_section_text(text: str) -> str:
    text = FOOTER_RE.sub("", text)
    # Drop empty lines created by footer removal, keep paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(full_text: str) -> list[tuple[str, str, str]]:
    """Return [(item_id, section_name, section_text), ...] in document order."""
    matches = list(SECTION_RE.finditer(full_text))
    if not matches:
        return [("", "Full Document", full_text)]

    sections = []
    for idx, m in enumerate(matches):
        item_id = m.group(1).upper()
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        body = _clean_section_text(full_text[start:end])
        name = TEN_K_SECTIONS.get(item_id, title)
        sections.append((item_id, name, body))
    return sections


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(block: str, limit: int) -> list[str]:
    """Split an over-long single block (usually a big table) by lines."""
    out, cur = [], ""
    for line in block.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur.strip())
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_section_text(text: str, cfg: Config) -> list[str]:
    """Greedy paragraph packing into <= max_chunk_chars windows with overlap."""
    paras = _split_paragraphs(text)
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(para) > cfg.max_chunk_chars:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(_hard_split(para, cfg.max_chunk_chars))
            continue
        if len(cur) + len(para) + 2 > cfg.max_chunk_chars and cur:
            chunks.append(cur.strip())
            # carry a small overlap tail for cross-boundary context
            tail = cur[-cfg.chunk_overlap_chars:]
            cur = (tail + "\n\n" + para).strip()
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) >= 1]


def chunk_document(
    full_text: str, cfg: Config = DEFAULT, drop_low_signal: bool = True
) -> list[Chunk]:
    sections = split_sections(full_text)
    chunks: list[Chunk] = []
    for item_id, name, body in sections:
        if drop_low_signal and item_id in LOW_SIGNAL_SECTIONS:
            continue
        if len(body) < cfg.min_chunk_chars or EMPTY_RE.match(body.strip()):
            continue
        for part_idx, piece in enumerate(chunk_section_text(body, cfg)):
            if len(piece) < cfg.min_chunk_chars and part_idx > 0:
                # tiny trailing fragment - fold into previous chunk
                if chunks:
                    prev = chunks[-1]
                    prev.text = (prev.text + "\n\n" + piece).strip()
                    prev.n_chars = len(prev.text)
                    prev.n_tokens = estimate_tokens(prev.text)
                continue
            cid = stable_id(item_id, str(part_idx), piece[:120])
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    section_item=item_id,
                    section_name=name,
                    part_index=part_idx,
                    text=piece,
                    n_chars=len(piece),
                    n_tokens=estimate_tokens(piece),
                )
            )
    log.info(
        "Chunked into %d chunks across %d sections (%d tokens est.)",
        len(chunks),
        len({c.section_item for c in chunks}),
        sum(c.n_tokens for c in chunks),
    )
    return chunks
