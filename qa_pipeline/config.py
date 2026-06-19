"""Central configuration for the pipeline.

Everything is overridable from the environment so the same code runs in a
notebook, a CI job, or a 1000-document batch without edits. Defaults are
tuned for a single 10-K producing ~100-200 verified Q&A pairs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- SEC EDGAR ----------------------------------------------------------
    # SEC requires a descriptive User-Agent with a contact email. Be a good
    # citizen: identify yourself and stay under ~10 requests/second.
    user_agent: str = _env(
        "SEC_USER_AGENT", "Caliper Assessment Pipeline vian.genomics@gmail.com"
    )
    edgar_rate_limit_s: float = _env_float("SEC_RATE_LIMIT_S", 0.2)

    # --- Models -------------------------------------------------------------
    # Generation and verification deliberately default to *different* models.
    # An independent verifier reduces correlated errors: a passage the writer
    # misreads is unlikely to be misread the same way by a second model.
    generation_model: str = _env("QA_GEN_MODEL", "claude-sonnet-4-6")
    verification_model: str = _env("QA_VERIFY_MODEL", "claude-opus-4-8")
    max_output_tokens: int = _env_int("QA_MAX_OUTPUT_TOKENS", 4096)
    temperature: float = _env_float("QA_TEMPERATURE", 0.4)

    # --- Chunking -----------------------------------------------------------
    # Chunks are kept comfortably inside the model context while staying large
    # enough to contain self-sufficient context for a question.
    max_chunk_chars: int = _env_int("QA_MAX_CHUNK_CHARS", 6000)
    min_chunk_chars: int = _env_int("QA_MIN_CHUNK_CHARS", 600)
    chunk_overlap_chars: int = _env_int("QA_CHUNK_OVERLAP_CHARS", 250)

    # --- Generation targets -------------------------------------------------
    questions_per_chunk: int = _env_int("QA_QUESTIONS_PER_CHUNK", 5)
    target_pairs: int = _env_int("QA_TARGET_PAIRS", 120)
    # Skip chunks that are mostly boilerplate (cover page, exhibit index, ...).
    max_chunks: Optional[int] = None

    # --- Verification -------------------------------------------------------
    # Minimum fraction of the cited source passage that must be found verbatim
    # in the chunk for the cheap deterministic grounding check to pass.
    min_passage_overlap: float = _env_float("QA_MIN_PASSAGE_OVERLAP", 0.6)
    # Only pairs whose verdict is in this set survive into the final dataset.
    accept_verdicts: tuple = ("SUPPORTED",)
    # Require the (snapped) source_passage to be an exact contiguous substring of
    # the chunk. Guarantees the "exact text" column; drops the rare pair whose
    # evidence spans two non-contiguous passages (e.g. a cross-year table
    # comparison). Set False to keep those, flagged via the source_verbatim col.
    require_verbatim_source: bool = True

    # --- Paths --------------------------------------------------------------
    data_dir: str = _env("QA_DATA_DIR", "data")
    output_dir: str = _env("QA_OUTPUT_DIR", "output")
    cache_dir: str = _env("QA_CACHE_DIR", ".cache")
    run_name: str = _env("QA_RUN_NAME", "apple_10k_fy2025")

    def to_dict(self) -> dict:
        return asdict(self)


# A module-level default instance for convenience; callers may build their own.
DEFAULT = Config()

# Canonical 10-K section map. Keys are the Item identifiers as they appear in
# the filing; values are the human-readable section names used in the dataset.
TEN_K_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Sections that are almost always boilerplate / low signal for QA generation.
LOW_SIGNAL_SECTIONS = {
    "1B",  # usually "None"
    "4",   # mine safety - N/A for most companies
    "9",   # usually "None"
    "9B",
    "15",  # exhibit index
    "16",
}
