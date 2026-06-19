#!/usr/bin/env python
"""Command-line entrypoint for the 10-K -> verified Q&A dataset pipeline.

Examples
--------
Live run against the Anthropic API (needs ANTHROPIC_API_KEY)::

    python run.py --ticker AAPL

Re-generate the shipped sample dataset offline from captured model outputs
(no API key, no network)::

    python run.py --from-cache --html data/raw/aapl-20250927.htm \
                  --company "Apple Inc." --ticker AAPL --fiscal-year 2025 \
                  --accession 0000320193-25-000079 --run-name apple_10k_fy2025
"""

from __future__ import annotations

import argparse
import os

from qa_pipeline.chunk import chunk_document
from qa_pipeline.config import Config
from qa_pipeline.edgar import fetch_10k
from qa_pipeline.llm import build_client
from qa_pipeline.parse import parse_file
from qa_pipeline.pipeline import run_pipeline, write_outputs, compute_stats
from qa_pipeline.utils import get_logger, write_json

log = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source")
    src.add_argument("--ticker", help="Stock ticker, e.g. AAPL")
    src.add_argument("--cik", help="SEC CIK (alternative to --ticker)")
    src.add_argument("--skip", type=int, default=0,
                     help="0=latest 10-K, 1=previous, ...")
    src.add_argument("--html", help="Use a local 10-K html file instead of downloading")

    meta = p.add_argument_group("metadata (used when --html is given)")
    meta.add_argument("--company", default="")
    meta.add_argument("--fiscal-year", default="")
    meta.add_argument("--accession", default="")

    run = p.add_argument_group("run")
    run.add_argument("--from-cache", action="store_true",
                     help="Replay captured model outputs; no API key needed")
    run.add_argument("--target", type=int, help="Target number of pairs")
    run.add_argument("--questions-per-chunk", type=int)
    run.add_argument("--max-chunks", type=int)
    run.add_argument("--run-name", default=None)
    run.add_argument("--out", default=None, help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.target:
        cfg.target_pairs = args.target
    if args.questions_per_chunk:
        cfg.questions_per_chunk = args.questions_per_chunk
    if args.max_chunks:
        cfg.max_chunks = args.max_chunks
    if args.run_name:
        cfg.run_name = args.run_name
    out_dir = args.out or cfg.output_dir

    # --- obtain the filing -------------------------------------------------
    if args.html:
        html_path = args.html
        meta = {
            "company": args.company or "Unknown",
            "ticker": args.ticker,
            "fiscal_year": args.fiscal_year,
            "accession": args.accession,
            "form": "10-K",
            "period": "",
        }
    else:
        if not (args.ticker or args.cik):
            raise SystemExit("Provide --ticker/--cik, or --html for a local file.")
        filing, html_path = fetch_10k(
            ticker=args.ticker, cik=args.cik,
            out_dir=os.path.join(cfg.data_dir, "raw"), cfg=cfg, skip=args.skip,
        )
        meta = {
            "company": filing.company,
            "ticker": args.ticker,
            "fiscal_year": filing.period_of_report[:4],
            "accession": filing.accession,
            "form": filing.form,
            "period": filing.period_of_report,
        }
        if not args.run_name:
            cfg.run_name = f"{(args.ticker or args.cik)}_{filing.period_of_report[:4]}".lower()

    # --- parse + chunk -----------------------------------------------------
    doc = parse_file(html_path)
    log.info("Parsed %d chars, %d tables", doc.n_chars, doc.n_tables)
    chunks = chunk_document(doc.text, cfg)

    # --- generate + verify + assemble -------------------------------------
    client = build_client(cfg, from_cache=args.from_cache)
    result = run_pipeline(meta, chunks, client, cfg)

    # --- write -------------------------------------------------------------
    paths = write_outputs(result, out_dir, cfg.run_name)
    stats = compute_stats(result)
    print("\n=== SUMMARY ===")
    print(f"Accepted pairs : {stats['n_accepted']}")
    print(f"Rejected pairs : {stats['n_rejected']}")
    print(f"Acceptance rate: {stats['acceptance_rate']}")
    print(f"By type        : {stats['by_question_type']}")
    print(f"By difficulty  : {stats['by_difficulty']}")
    print(f"Outputs        : {paths['csv']}")


if __name__ == "__main__":
    main()
