"""Tests for the deterministic grounding gate and acceptance logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_pipeline.config import Config
from qa_pipeline.schema import GeneratedQA, VerificationResult
from qa_pipeline.verify import deterministic_grounding, is_accepted


CHUNK = (
    "Total net sales were $416,161 million in 2025, compared to $391,035 "
    "million in 2024. Services gross margin percentage was 75.4%."
)


def _qa(answer, passage, qtype="fact_extraction"):
    return GeneratedQA(
        question="Q?", answer=answer, source_passage=passage,
        question_type=qtype, difficulty="easy",
    )


def test_verbatim_passage_passes():
    qa = _qa("$416,161 million", "Total net sales were $416,161 million in 2025")
    g = deterministic_grounding(qa, CHUNK)
    assert g.passage_in_chunk
    assert g.token_overlap == 1.0
    assert g.passed


def test_fabricated_passage_fails():
    qa = _qa("$999,999 million", "Net sales tripled to $999,999 million overnight")
    g = deterministic_grounding(qa, CHUNK)
    assert not g.passage_in_chunk
    assert not g.passed


def test_hallucinated_number_in_factual_answer_flagged():
    # Passage is real, but the factual answer states a number not in the chunk.
    qa = _qa("$500,000 million", "Total net sales were $416,161 million in 2025")
    g = deterministic_grounding(qa, CHUNK)
    assert g.passage_in_chunk          # the quote is real ...
    assert not g.numbers_supported     # ... but the asserted figure is not
    assert not g.passed


def test_calculation_answer_allows_derived_number():
    # A computed delta (25,126) need not appear verbatim in the source.
    qa = _qa(
        "Net sales rose $25,126 million year over year.",
        "Total net sales were $416,161 million in 2025, compared to $391,035 million in 2024",
        qtype="numeric_calculation",
    )
    g = deterministic_grounding(qa, CHUNK)
    assert g.passage_in_chunk
    assert g.numbers_supported  # lenient for calculations
    assert g.passed


def test_acceptance_requires_grounding_and_verdict():
    cfg = Config()
    qa = _qa("$416,161 million", "Total net sales were $416,161 million in 2025")
    g = deterministic_grounding(qa, CHUNK)
    supported = VerificationResult(verdict="SUPPORTED", confidence=0.9, reasoning="ok")
    notsup = VerificationResult(verdict="NOT_SUPPORTED", confidence=0.9, reasoning="no")
    assert is_accepted(g, supported, cfg)
    assert not is_accepted(g, notsup, cfg)  # verdict gate
    # grounding gate: even a SUPPORTED verdict cannot save a fabricated passage
    bad = _qa("x", "this text is absolutely not in the chunk 12345")
    assert not is_accepted(deterministic_grounding(bad, CHUNK), supported, cfg)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all verify tests passed")
