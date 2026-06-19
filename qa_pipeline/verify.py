"""Verification: confirm each answer is actually supported by its source.

Two independent checks, cheapest first:

1. Deterministic grounding (no LLM). Confirms the cited ``source_passage`` is
   present in the chunk, so the model cannot invent a quote, and that numbers in
   a factual answer appear in the source. This catches fabricated passages
   without an API call.

2. Independent LLM verification. A different model re-reads the chunk and judges
   whether the answer is correct and entailed, recomputing the arithmetic on
   numeric questions. A separate model reduces correlated errors.

A pair is accepted only if grounding passes and the LLM verdict is in
``cfg.accept_verdicts``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .config import Config, DEFAULT
from .llm import BaseClient
from .prompts import build_verification_messages
from .schema import (
    GeneratedQA,
    GroundingResult,
    VERIFICATION_JSON_SCHEMA,
    VerificationResult,
)
from .utils import fold, norm_punct

# Matches monetary/percentage/plain numbers: $391,035 | 46.9% | 1,234 | 3.50
_NUM_RE = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?\s?%?")
_LENIENT_TYPES = {"numeric_calculation", "multi_step_reasoning"}


def _norm_number(tok: str) -> str:
    return re.sub(r"[^\d.]", "", tok)


def _numbers(text: str) -> list[str]:
    out = []
    for m in _NUM_RE.findall(text):
        n = _norm_number(m)
        if n and any(ch.isdigit() for ch in n):
            out.append(n.rstrip(".").lstrip("0") or "0")
    return out


def _token_overlap(passage: str, chunk: str) -> float:
    p = set(fold(passage).split())
    c = set(fold(chunk).split())
    if not p:
        return 0.0
    return len(p & c) / len(p)


def deterministic_grounding(
    qa: GeneratedQA, chunk_text: str, cfg: Config = DEFAULT
) -> GroundingResult:
    np = fold(qa.source_passage)
    nc = fold(chunk_text)
    exact = bool(np) and np in nc
    overlap = 1.0 if exact else _token_overlap(qa.source_passage, chunk_text)
    passage_in_chunk = exact or overlap >= cfg.min_passage_overlap

    # Numbers in the *answer* should be traceable to the source for factual
    # questions. Calculations may legitimately introduce a derived value.
    chunk_nums = set(_numbers(chunk_text)) | set(_numbers(qa.source_passage))
    answer_nums = _numbers(qa.answer)
    missing = [n for n in answer_nums if n not in chunk_nums]
    if qa.question_type in _LENIENT_TYPES or not answer_nums:
        numbers_supported = True
    else:
        numbers_supported = (len(answer_nums) - len(missing)) / len(answer_nums) >= 0.5

    return GroundingResult(
        passage_in_chunk=passage_in_chunk,
        token_overlap=round(overlap, 3),
        numbers_supported=numbers_supported,
        missing_numbers=missing,
    )


def snap_passage(passage: str, chunk_text: str) -> tuple[str, bool]:
    """Align a model-cited passage to the exact verbatim span in the chunk.

    The generator sometimes lightly reformats a quote (drops the table ``|``
    separators, swaps a curly apostrophe for an ASCII one). To honour the
    "exact text supporting the answer" requirement, we find the matching region
    of the chunk and return that *verbatim* slice. Matching is done on a
    punctuation-folded copy (same length, so indices map 1:1 back to the
    original chunk).

    Returns ``(snapped_text, snapped)`` where ``snapped`` is False if no good
    alignment was found (caller keeps the original passage).
    """
    if not passage.strip():
        return passage, False
    # Match on a punctuation-folded copy (same length => indices map 1:1 back to
    # the original chunk, so the returned slice is still byte-for-byte verbatim).
    nchunk = norm_punct(chunk_text)
    npass = norm_punct(passage)
    sm = SequenceMatcher(None, nchunk, npass, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return passage, False
    start, end = blocks[0].a, blocks[-1].a + blocks[-1].size
    if end - start > 3 * len(passage) + 80:
        # blocks too scattered: fall back to the single longest match
        lm = max(blocks, key=lambda b: b.size)
        start, end = lm.a, lm.a + lm.size
    span = chunk_text[start:end].strip()
    # keep only if the verbatim span still covers the cited content
    p_tokens = set(fold(passage).split())
    s_tokens = set(fold(span).split())
    covered = bool(p_tokens) and len(p_tokens & s_tokens) / len(p_tokens) >= 0.6
    if covered and fold(span) in fold(chunk_text):
        return span, True
    return passage, False


def llm_verify(
    qa: GeneratedQA,
    chunk_text: str,
    client: BaseClient,
    chunk_id: str,
    index: int,
    cfg: Config = DEFAULT,
) -> VerificationResult:
    system, user = build_verification_messages(qa, chunk_text)
    data = client.complete_json(
        model=cfg.verification_model,
        system=system,
        user=user,
        schema=VERIFICATION_JSON_SCHEMA,
        key=f"verify::{chunk_id}::{index}",
    )
    return VerificationResult(**data)


def is_accepted(
    grounding: GroundingResult, verification: VerificationResult, cfg: Config = DEFAULT
) -> bool:
    return grounding.passed and verification.verdict in cfg.accept_verdicts
