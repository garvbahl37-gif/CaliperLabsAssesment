"""Typed schema for every artifact that flows through the pipeline.

Using Pydantic gives us free validation of the LLM's structured output: if the
model returns an out-of-range difficulty or a missing field, generation fails
loudly instead of silently polluting the dataset.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal[
    "fact_extraction",
    "numeric_calculation",
    "comparison",
    "multi_step_reasoning",
]
Difficulty = Literal["easy", "medium", "hard"]
Verdict = Literal["SUPPORTED", "PARTIAL", "NOT_SUPPORTED"]

# JSON schema fragments handed to the model's structured-output tool. Kept here
# so the live API client and the offline replay path agree on the exact shape.
GENERATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "source_passage": {"type": "string"},
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "fact_extraction",
                            "numeric_calculation",
                            "comparison",
                            "multi_step_reasoning",
                        ],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "question",
                    "answer",
                    "source_passage",
                    "question_type",
                    "difficulty",
                ],
            },
        }
    },
    "required": ["qa_pairs"],
}

VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "PARTIAL", "NOT_SUPPORTED"],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "answer_is_correct": {"type": "boolean"},
        "passage_supports_answer": {"type": "boolean"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}


class GeneratedQA(BaseModel):
    """A single Q&A as emitted by the generation model (pre-verification)."""

    question: str
    answer: str
    source_passage: str
    question_type: QuestionType
    difficulty: Difficulty
    rationale: Optional[str] = None


class VerificationResult(BaseModel):
    """Output of the independent LLM verifier for one Q&A pair."""

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    answer_is_correct: Optional[bool] = None
    passage_supports_answer: Optional[bool] = None


class GroundingResult(BaseModel):
    """Output of the cheap, deterministic (no-LLM) grounding check."""

    passage_in_chunk: bool
    token_overlap: float
    numbers_supported: bool
    missing_numbers: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.passage_in_chunk and self.numbers_supported


class QAPair(BaseModel):
    """A fully-verified dataset row."""

    id: str
    # --- the five required columns -------------------------------------
    question: str
    answer: str
    source_passage: str
    question_type: QuestionType
    difficulty: Difficulty
    # --- provenance / traceability -------------------------------------
    company: str
    ticker: Optional[str] = None
    fiscal_year: Optional[str] = None
    accession: Optional[str] = None
    section_item: str = ""
    section_name: str = ""
    chunk_id: str = ""
    # --- verification evidence -----------------------------------------
    verdict: Verdict = "SUPPORTED"
    verifier_confidence: float = 0.0
    grounding_overlap: float = 0.0
    passage_in_chunk: bool = False
    source_verbatim: bool = False
    verifier_reasoning: str = ""

    def dataset_row(self) -> dict:
        """The flat row written to CSV/JSONL (required columns first)."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "source_passage": self.source_passage,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "section": self.section_name,
            "company": self.company,
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "chunk_id": self.chunk_id,
            "verdict": self.verdict,
            "verifier_confidence": round(self.verifier_confidence, 3),
            "grounding_overlap": round(self.grounding_overlap, 3),
            "source_verbatim": self.source_verbatim,
        }
