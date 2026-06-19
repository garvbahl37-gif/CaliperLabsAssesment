"""End-to-end orchestration: chunks -> generate -> verify -> assemble -> write.

This is the single code path used by BOTH the live API run and the offline
replay run; only the injected ``client`` differs.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass, field

from .chunk import Chunk
from .config import Config, DEFAULT
from .generate import generate_for_chunk
from .llm import BaseClient
from .schema import QAPair
from .utils import (
    ensure_dir,
    get_logger,
    normalized_question_key,
    stable_id,
    write_json,
    write_jsonl,
)
from .verify import deterministic_grounding, is_accepted, llm_verify, snap_passage
from .utils import fold

log = get_logger()


@dataclass
class RunResult:
    accepted: list[QAPair] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def process_chunk(
    chunk: Chunk, meta: dict, client: BaseClient, cfg: Config = DEFAULT
) -> tuple[list[QAPair], list[dict]]:
    accepted: list[QAPair] = []
    rejected: list[dict] = []
    candidates = generate_for_chunk(chunk, meta, client, cfg)
    for i, qa in enumerate(candidates):
        grounding = deterministic_grounding(qa, chunk.text, cfg)
        verification = llm_verify(qa, chunk.text, client, chunk.chunk_id, i, cfg)
        accept = is_accepted(grounding, verification, cfg)
        if accept:
            # Anti-fabrication grounding ran on the *original* cited passage.
            # Now snap it to the exact verbatim span in the chunk so the stored
            # source_passage truly is "exact text supporting the answer".
            passage, snapped = snap_passage(qa.source_passage, chunk.text)
            verbatim = fold(passage) in fold(chunk.text)
            if cfg.require_verbatim_source and not verbatim:
                rejected.append({
                    "chunk_id": chunk.chunk_id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "question_type": qa.question_type,
                    "verdict": verification.verdict,
                    "passage_in_chunk": grounding.passage_in_chunk,
                    "token_overlap": grounding.token_overlap,
                    "missing_numbers": grounding.missing_numbers,
                    "reason": "source_passage not a single contiguous verbatim span",
                })
                continue
            accepted.append(
                QAPair(
                    id=stable_id(chunk.chunk_id, qa.question),
                    question=qa.question,
                    answer=qa.answer,
                    source_passage=passage,
                    question_type=qa.question_type,
                    difficulty=qa.difficulty,
                    company=meta.get("company", ""),
                    ticker=meta.get("ticker"),
                    fiscal_year=meta.get("fiscal_year"),
                    accession=meta.get("accession"),
                    section_item=chunk.section_item,
                    section_name=chunk.section_name,
                    chunk_id=chunk.chunk_id,
                    verdict=verification.verdict,
                    verifier_confidence=verification.confidence,
                    grounding_overlap=1.0 if verbatim else grounding.token_overlap,
                    passage_in_chunk=verbatim,
                    source_verbatim=verbatim,
                    verifier_reasoning=verification.reasoning,
                )
            )
        else:
            rejected.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "question_type": qa.question_type,
                    "verdict": verification.verdict,
                    "passage_in_chunk": grounding.passage_in_chunk,
                    "token_overlap": grounding.token_overlap,
                    "missing_numbers": grounding.missing_numbers,
                    "reason": verification.reasoning,
                }
            )
    return accepted, rejected


def dedupe(pairs: list[QAPair]) -> list[QAPair]:
    """Drop near-duplicate questions, keeping the best-verified instance."""
    best: dict[str, QAPair] = {}
    for p in pairs:
        key = normalized_question_key(p.question)
        cur = best.get(key)
        if cur is None or (
            (p.verifier_confidence, p.grounding_overlap)
            > (cur.verifier_confidence, cur.grounding_overlap)
        ):
            best[key] = p
    # Preserve input order for determinism.
    seen = set()
    out = []
    for p in pairs:
        key = normalized_question_key(p.question)
        if key in best and best[key] is p and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def run_pipeline(
    meta: dict, chunks: list[Chunk], client: BaseClient, cfg: Config = DEFAULT
) -> RunResult:
    all_accepted: list[QAPair] = []
    all_rejected: list[dict] = []
    n = len(chunks) if cfg.max_chunks is None else min(cfg.max_chunks, len(chunks))
    for idx, chunk in enumerate(chunks[:n]):
        acc, rej = process_chunk(chunk, meta, client, cfg)
        all_accepted.extend(acc)
        all_rejected.extend(rej)
        log.info(
            "[%2d/%2d] Item %-4s %-7s  +%d accepted  -%d rejected  (running: %d)",
            idx + 1, n, chunk.section_item, f"p{chunk.part_index}",
            len(acc), len(rej), len(all_accepted),
        )
    before = len(all_accepted)
    deduped = dedupe(all_accepted)
    log.info("Deduplication: %d -> %d pairs", before, len(deduped))
    return RunResult(accepted=deduped, rejected=all_rejected, meta=meta)


def compute_stats(result: RunResult) -> dict:
    acc = result.accepted
    by_type = Counter(p.question_type for p in acc)
    by_diff = Counter(p.difficulty for p in acc)
    by_section = Counter(p.section_name for p in acc)
    confs = [p.verifier_confidence for p in acc] or [0.0]
    return {
        "source": result.meta,
        "n_accepted": len(acc),
        "n_rejected": len(result.rejected),
        "acceptance_rate": round(
            len(acc) / max(1, len(acc) + len(result.rejected)), 3
        ),
        "by_question_type": dict(by_type),
        "by_difficulty": dict(by_diff),
        "by_section": dict(by_section),
        "mean_verifier_confidence": round(sum(confs) / len(confs), 3),
        "pct_exact_passage_match": round(
            100 * sum(1 for p in acc if p.passage_in_chunk) / max(1, len(acc)), 1
        ),
    }


def write_outputs(
    result: RunResult, out_dir: str, run_name: str
) -> dict[str, str]:
    ensure_dir(out_dir)
    rows = [p.dataset_row() for p in result.accepted]

    csv_path = os.path.join(out_dir, f"{run_name}_dataset.csv")
    jsonl_path = os.path.join(out_dir, f"{run_name}_dataset.jsonl")
    stats_path = os.path.join(out_dir, f"{run_name}_stats.json")
    rejects_path = os.path.join(out_dir, f"{run_name}_rejected.jsonl")

    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_jsonl(jsonl_path, rows)
    write_jsonl(rejects_path, result.rejected)
    stats = compute_stats(result)
    write_json(stats_path, stats)

    log.info("Wrote %d rows -> %s", len(rows), csv_path)
    return {
        "csv": csv_path,
        "jsonl": jsonl_path,
        "stats": stats_path,
        "rejects": rejects_path,
    }
