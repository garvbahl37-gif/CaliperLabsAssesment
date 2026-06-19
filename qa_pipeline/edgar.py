"""Resolve and download 10-K filings from SEC EDGAR.

EDGAR is free and requires no API key, but it does require a descriptive
User-Agent header and polite rate limiting. We use the official JSON APIs:

  * https://www.sec.gov/files/company_tickers.json   (ticker -> CIK)
  * https://data.sec.gov/submissions/CIK##########.json  (filing history)
  * https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>

This module is fully deterministic and needs no LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config, DEFAULT
from .utils import ensure_dir, get_logger

log = get_logger()

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


@dataclass
class Filing:
    company: str
    cik: str
    form: str
    filing_date: str
    period_of_report: str
    accession: str
    primary_document: str
    url: str


def _headers(cfg: Config) -> dict:
    return {"User-Agent": cfg.user_agent, "Accept-Encoding": "gzip, deflate"}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def _get(url: str, cfg: Config) -> requests.Response:
    time.sleep(cfg.edgar_rate_limit_s)  # be polite to SEC
    resp = requests.get(url, headers=_headers(cfg), timeout=60)
    resp.raise_for_status()
    return resp


def resolve_cik(ticker: str, cfg: Config = DEFAULT) -> str:
    """Map a stock ticker (e.g. 'AAPL') to a zero-padded 10-digit CIK."""
    data = _get(TICKERS_URL, cfg).json()
    ticker = ticker.upper().strip()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker!r} not found in SEC ticker index")


def latest_filing(
    cik: str, form: str = "10-K", cfg: Config = DEFAULT, skip: int = 0
) -> Filing:
    """Return the most recent filing of `form` for a CIK (skip>0 for older)."""
    cik10 = str(cik).zfill(10)
    data = _get(SUBMISSIONS_URL.format(cik10=cik10), cfg).json()
    company = data.get("name", "Unknown")
    recent = data["filings"]["recent"]
    matches = [i for i, f in enumerate(recent["form"]) if f == form]
    if not matches:
        raise ValueError(f"No {form} filings found for CIK {cik10}")
    if skip >= len(matches):
        raise ValueError(f"Only {len(matches)} {form} filings; cannot skip {skip}")
    i = matches[skip]
    acc = recent["accessionNumber"][i]
    doc = recent["primaryDocument"][i]
    cik_int = str(int(cik10))
    url = ARCHIVE_URL.format(cik=cik_int, acc_nodash=acc.replace("-", ""), doc=doc)
    return Filing(
        company=company,
        cik=cik10,
        form=form,
        filing_date=recent["filingDate"][i],
        period_of_report=recent["reportDate"][i],
        accession=acc,
        primary_document=doc,
        url=url,
    )


def download_filing(
    filing: Filing, out_dir: str, cfg: Config = DEFAULT
) -> str:
    """Download the primary document HTML to disk and return its path."""
    ensure_dir(out_dir)
    path = f"{out_dir}/{filing.primary_document}"
    log.info("Downloading %s %s -> %s", filing.company, filing.form, path)
    resp = _get(filing.url, cfg)
    with open(path, "wb") as f:
        f.write(resp.content)
    log.info("Saved %d bytes", len(resp.content))
    return path


def fetch_10k(
    ticker: Optional[str] = None,
    cik: Optional[str] = None,
    out_dir: str = "data/raw",
    cfg: Config = DEFAULT,
    skip: int = 0,
) -> tuple[Filing, str]:
    """High-level helper: ticker/CIK -> (Filing metadata, local html path)."""
    if not ticker and not cik:
        raise ValueError("Provide either ticker or cik")
    if not cik:
        cik = resolve_cik(ticker, cfg)
    filing = latest_filing(cik, "10-K", cfg, skip=skip)
    path = download_filing(filing, out_dir, cfg)
    return filing, path
