"""Small shared helpers: logging, IO, text normalisation, dedup hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Iterable


def get_logger(name: str = "qa_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("QA_LOG_LEVEL", "INFO"))
    return logger


_WS_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def normalize_ws(text: str) -> str:
    """Collapse runs of spaces/tabs but keep paragraph structure (newlines)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    # Trim trailing spaces on each line.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# Same-length normalisation of "smart" punctuation. Filings use curly quotes
# and en/em dashes; LLMs tend to emit ASCII ones. Mapping each to a single
# ASCII char (1->1, never changing length) lets us match a model's quote against
# the filing while still slicing exact spans out of the original text.
_PUNCT_TABLE = str.maketrans({
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
})


def norm_punct(text: str) -> str:
    """ASCII-fold smart punctuation without changing string length."""
    return str(text).translate(_PUNCT_TABLE)


def squash(text: str) -> str:
    """Collapse all whitespace to single spaces (layout-insensitive).

    NOTE: this is intentionally punctuation-preserving because it is also used
    to render filing text during parsing. For fuzzy *matching* use ``fold``.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def fold(text: str) -> str:
    """Aggressive normalisation for fuzzy matching: whitespace-collapsed,
    smart-punctuation ASCII-folded, lower-cased. Used by grounding/snapping so
    that quote style, dashes, and layout never cause a false mismatch."""
    return re.sub(r"\s+", " ", norm_punct(str(text))).strip().lower()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English prose)."""
    return max(1, len(text) // 4)


def stable_id(*parts: str, length: int = 12) -> str:
    h = hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()
    return h[:length]


def normalized_question_key(question: str) -> str:
    """A loose key for near-duplicate question detection."""
    q = question.lower()
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    q = re.sub(r"\b(what|is|the|a|an|of|in|for|to|s|was|were|did|how|much|many)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    ensure_dir(os.path.dirname(path) or ".")
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
