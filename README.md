# 10-K → Verified Q&A Dataset Pipeline

An automated pipeline that turns any **SEC 10-K filing** into a large,
**verified** dataset of question–answer pairs for benchmarking LLMs on real
financial documents — with no human writing questions by hand.

The headline feature is the **verification stage**: every generated answer must
survive *two independent checks* — a free deterministic grounding check and an
independent LLM verifier (a different model) — before it enters the dataset.
That is what keeps hallucinations out.

> **Sample output:** [`output/apple_10k_fy2025_dataset.csv`](output/apple_10k_fy2025_dataset.csv)
> — **247 verified Q&A pairs** generated from Apple Inc.'s **FY2025 10-K**
> (filed 2025-10-31, fiscal year ended 2025-09-27).

---

## 1. What it produces

Each row has the five required fields plus full provenance and verification
evidence:

| column | meaning |
|---|---|
| `question` | self-contained question (names the company/period) |
| `answer` | ground-truth answer |
| `source_passage` | **exact verbatim text** from the filing that supports the answer |
| `question_type` | `fact_extraction` / `numeric_calculation` / `comparison` / `multi_step_reasoning` |
| `difficulty` | `easy` / `medium` / `hard` |
| `section` | 10-K section the pair came from (Risk Factors, MD&A, …) |
| `company`, `ticker`, `fiscal_year`, `accession`, `chunk_id` | provenance — every pair traces back to the exact filing & passage |
| `verdict`, `verifier_confidence`, `grounding_overlap`, `source_verbatim` | verification evidence |

### Sample rows (real, from the shipped dataset)

```
[fact_extraction / easy / Business]
Q: In fiscal 2025, what percentage of Apple Inc.'s total net sales came through
   indirect distribution channels?
A: 60%
SRC: "During 2025, the Company's net sales through its direct and indirect
      distribution channels accounted for 40% and 60%, respectively, of total
      net sales."

[numeric_calculation / medium / MD&A]
Q: In fiscal year 2025, what was the dollar increase in Apple's Services net
   sales compared to fiscal year 2024?
A: $12,989 million (from $96,169M in 2024 to $109,158M in 2025)
SRC: "Services (1) | 109,158 | 14 | % | 96,169 | 13 | % | 85,200"

[comparison / medium / Market for Registrant's Common Equity]
Q: Based on the five-year cumulative total shareholder return ($100 invested in
   Sept 2020), which delivered the highest return by Sept 2025: Apple, the
   S&P 500, or the Dow Jones U.S. Technology index?
A: The Dow Jones U.S. Technology index ($287), vs Apple ($234) and S&P 500 ($217).
SRC: "Apple Inc. | $ | 100 | ... | 234   S&P 500 Index | $ | 100 | ... | 217 ..."
```

### Dataset statistics

| metric | value |
|---|---|
| Verified pairs | **247** |
| Acceptance rate (passed verification) | 98.4% |
| Mean verifier confidence | 0.98 |
| Source passages that are exact verbatim quotes | **100%** |
| Sections covered | 9 |

**By question type:** fact_extraction 144 · multi_step_reasoning 57 ·
numeric_calculation 28 · comparison 18
**By difficulty:** easy 123 · medium 97 · hard 27
**Top sections:** Risk Factors 98 · Financial Statements 74 · MD&A 24 · Business 22

Full machine-readable stats: [`output/apple_10k_fy2025_stats.json`](output/apple_10k_fy2025_stats.json).
Pairs that *failed* verification are not discarded silently — they are written to
[`output/apple_10k_fy2025_rejected.jsonl`](output/apple_10k_fy2025_rejected.jsonl)
for transparency.

---

## 2. How it works

```
SEC EDGAR 10-K (HTML)
        │  edgar.py        download by ticker/CIK (polite, retrying)
        ▼
   parse.py               table-aware HTML → clean text (numbers stay with labels)
        ▼
   chunk.py               split on real "Item N." section headers, then pack into
                          ~6k-char windows  →  43 section-labelled chunks
        ▼
   generate.py  (LLM #1)  per chunk: write grounded Q&A pairs + verbatim source
        ▼
   verify.py    (LLM #2)  TWO independent gates per pair:
                          (a) deterministic grounding  — no LLM, free
                          (b) independent LLM verifier  — a different model
        ▼
   pipeline.py            snap passages to verbatim · dedupe · assemble
        ▼
   output/*.csv | *.jsonl | *_stats.json | *_rejected.jsonl
```

### The verification stage (the part that matters)

A generated answer is accepted **only if it passes both** of these:

1. **Deterministic grounding** (`verify.deterministic_grounding`) — *no LLM, runs
   anywhere, costs nothing.* It confirms the cited `source_passage` is actually
   present in the chunk (so the model cannot invent a quote) and that numbers
   asserted in a *factual* answer are traceable to the source. This alone kills
   the most damaging failure mode — fabricated passages — for free.

2. **Independent LLM verification** (`verify.llm_verify`) — a **different model**
   (`claude-opus-4-8`) than the generator (`claude-sonnet-4-6`) re-reads the
   chunk and judges whether the answer is correct and fully entailed,
   **re-deriving the arithmetic** for numeric questions. Using a separate model
   reduces correlated errors: a passage the writer misreads is unlikely to be
   misread the same way by an independent checker.

A pair is kept only if grounding passes **and** the verifier returns
`SUPPORTED`.

**This actually catches real errors.** In this run the verifier rejected a pair
where the generator claimed Apple's Services net sales grew **$13,989M** — the
Opus verifier re-derived `109,158 − 96,169 = 12,989` and rejected the answer for
overstating growth by $1,000M. (See it in `..._rejected.jsonl`.) Of 251
candidates: 249 `SUPPORTED`, 2 `PARTIAL`, 1 `NOT_SUPPORTED`.

### Exact source passages (`snap_passage`)

LLMs lightly reformat quotes (drop a table's `|` separators, swap a curly
apostrophe for an ASCII one). To honour *"exact text supporting the answer"*,
after grounding passes we **align the cited passage to the exact verbatim span
in the chunk** and store that. Result: **100% of shipped `source_passage` values
are byte-for-byte substrings of the parsed filing.** The rare pair whose evidence
spans two non-contiguous passages (e.g. a cross-year table comparison) is routed
to the rejected file by default (`require_verbatim_source`).

---

## 3. Running it

### Install
```bash
pip install -r requirements.txt
```

### Live run (generates a fresh dataset from any filing)
Needs an Anthropic API key. Generation and verification both call the API.
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --ticker AAPL                 # latest 10-K for Apple
python run.py --ticker MSFT --skip 1        # the *previous* 10-K
python run.py --cik 0000320193              # by CIK instead of ticker
```

### Reproduce the shipped sample offline (no API key, no network)
The captured model outputs are committed under `data/llm_runs/`, so the **entire
pipeline** (parse → chunk → grounding → verifier gate → snapping → dedupe →
output) runs locally against them:
```bash
python run.py --from-cache \
  --html data/raw/aapl-20250927.htm \
  --company "Apple Inc." --ticker AAPL --fiscal-year 2025 \
  --accession 0000320193-25-000079 --run-name apple_10k_fy2025
```

### Tests
```bash
python tests/test_chunk.py && python tests/test_verify.py
```

> **How the sample's model outputs were produced.** This environment had no API
> key, so the generation + independent-verification calls were executed with
> Claude (Sonnet 4.6 generating, Opus 4.8 verifying) using the *exact prompts in
> [`qa_pipeline/prompts.py`](qa_pipeline/prompts.py)*, and the responses were
> captured to `data/llm_runs/`. With `ANTHROPIC_API_KEY` set, `run.py` makes the
> identical calls itself via the Anthropic SDK. The two paths share one code
> path (`pipeline.run_pipeline`) — only the transport differs (`llm.py`).

---

## 4. Repository layout

```
qa_pipeline/
  config.py      tunables (models, chunk sizes, thresholds) — all env-overridable
  edgar.py       resolve + download a 10-K from SEC EDGAR (ticker/CIK)
  parse.py       table-aware HTML → clean text
  chunk.py       section detection ("Item N.") + windowing
  schema.py      Pydantic models (validates LLM output)
  prompts.py     generation + verification prompts (the source of dataset quality)
  llm.py         pluggable client: AnthropicClient (live) | ReplayClient (offline)
  generate.py    chunk → candidate Q&A
  verify.py      deterministic grounding + LLM verifier + passage snapping
  pipeline.py    orchestration: generate→verify→assemble, dedupe, write outputs
  merge_runs.py  merge captured generation+verification files for replay
run.py           CLI entrypoint
tests/           unit tests for chunking and verification
docs/SCALING.md  how to scale to many documents / 1000+ pairs
data/            raw filing, chunk manifest, captured LLM outputs (replay inputs)
output/          the shipped dataset (csv + jsonl), stats, rejected pairs
```

---

## 5. Design choices

- **Section-aware chunking over blind splitting.** A 10-K has a fixed skeleton
  of `Item N.` sections. Detecting them gives every chunk a meaningful `section`
  label and keeps related content together. TOC rows (which contain `|` from the
  source table) are distinguished from real headers, and repeating page footers
  are stripped.
- **Table-aware parsing.** Financial tables are rendered as
  `Label | col1 | col2` rows so numbers stay attached to their labels — most of
  the high-value numeric/comparison questions live in tables.
- **Two different models for generate vs verify.** Independence is the cheapest
  way to reduce correlated hallucinations. The verifier is also the *stronger*
  model, and re-derives arithmetic.
- **Cheap deterministic gate before the expensive one.** Fabricated quotes are
  caught for free, so the LLM verifier only spends tokens on plausible pairs.
- **Pluggable transport / full offline reproducibility.** One pipeline, two
  clients. The replay client makes the shipped dataset regenerate with no key,
  and the live client adds a response cache so re-runs are cheap and resumable.
- **Transparency.** Rejected pairs and per-row verification evidence are kept,
  not hidden.

## 6. Known limitations

- **Type mix skews to fact extraction (58%).** The generator produces what the
  text best supports; Risk Factors (the largest section) yields many factual
  questions. Hard/numeric/comparison questions are present but under-represented.
  The fix (steer generation toward under-filled type×difficulty cells) is
  described in [`docs/SCALING.md`](docs/SCALING.md).
- **Verification is high-precision, not perfect.** A single independent verifier
  catches clear errors; subtle misreadings could slip through. For a held-out
  benchmark split I'd use a 3-verifier majority (see scaling notes).
- **Table linearization loses 2-D structure.** Pipe-rendered tables are readable
  but a complex multi-column table can be ambiguous; XBRL cross-checking
  (proposed in the scaling notes) would make numeric answers bulletproof.
- **Non-contiguous evidence is dropped** by default rather than represented as a
  multi-span citation.
- **Tuned on Apple's (Workiva) HTML.** The parser handles the common filer
  formats and mis-declared encodings, but exotic layouts may need per-filer
  cleanup.

## 7. Scaling to 1000+ pairs / many documents

See **[docs/SCALING.md](docs/SCALING.md)** — covers durable-queue fan-out,
the Batch API, tiered models, global cross-document dedup via embeddings,
type/difficulty balancing, stratified filing sampling, and XBRL-based numeric
verification.
