"""Merge raw per-chunk generation + verification captures into replay files.

The capture step writes two files per chunk into ``data/llm_runs/raw/``::

    gen_<chunk_id>.json     -> {"chunk_id", "generation": {"qa_pairs": [...]}}
    verify_<chunk_id>.json  -> {"chunk_id", "verifications": [...]}

This module joins them (by index, same order) into the flat file the
``ReplayClient`` expects at ``data/llm_runs/<chunk_id>.json``::

    {"chunk_id", "generation": {"qa_pairs": [...]}, "verifications": [...]}

Run with: ``python -m qa_pipeline.merge_runs``
"""

from __future__ import annotations

import glob
import json
import os

from .utils import get_logger, read_json, write_json

log = get_logger()


def merge(runs_dir: str = "data/llm_runs") -> dict:
    raw_dir = os.path.join(runs_dir, "raw")
    gen_files = sorted(glob.glob(os.path.join(raw_dir, "gen_*.json")))
    n_pairs = n_verifs = n_merged = n_missing_verify = 0

    for gf in gen_files:
        chunk_id = os.path.basename(gf)[len("gen_"):-len(".json")]
        try:
            gen = read_json(gf)
        except json.JSONDecodeError:
            log.warning("Skipping malformed generation file: %s", gf)
            continue
        qa_pairs = gen.get("generation", {}).get("qa_pairs", [])

        vf = os.path.join(raw_dir, f"verify_{chunk_id}.json")
        verifs = []
        if os.path.exists(vf):
            try:
                verifs = read_json(vf).get("verifications", [])
            except json.JSONDecodeError:
                log.warning("Malformed verification file: %s", vf)
        else:
            n_missing_verify += 1
            log.warning("No verification file for chunk %s", chunk_id)

        # Align lengths defensively: a missing verdict becomes NOT_SUPPORTED so
        # an unverified pair can never slip into the final dataset.
        while len(verifs) < len(qa_pairs):
            verifs.append({
                "verdict": "NOT_SUPPORTED",
                "confidence": 0.0,
                "reasoning": "missing verification",
            })

        write_json(
            os.path.join(runs_dir, f"{chunk_id}.json"),
            {
                "chunk_id": chunk_id,
                "generation": {"qa_pairs": qa_pairs},
                "verifications": verifs[: len(qa_pairs)],
            },
        )
        n_pairs += len(qa_pairs)
        n_verifs += min(len(verifs), len(qa_pairs))
        n_merged += 1

    summary = {
        "chunks_merged": n_merged,
        "total_pairs": n_pairs,
        "total_verifications": n_verifs,
        "chunks_missing_verification": n_missing_verify,
    }
    log.info("Merged %d chunks, %d pairs, %d verifications (%d missing verify)",
             n_merged, n_pairs, n_verifs, n_missing_verify)
    return summary


if __name__ == "__main__":
    print(json.dumps(merge(), indent=2))
