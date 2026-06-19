# 10-K to Verified Q&A Dataset

Takes a SEC 10-K filing and builds a dataset of question/answer pairs from it.
Every answer goes through a verification step that checks it's actually supported
by the filing before it's kept, so generated-but-wrong pairs don't make it in.

Sample output: [`output/apple_10k_fy2025_dataset.csv`](output/apple_10k_fy2025_dataset.csv),
247 pairs built from Apple's FY2025 10-K (filed 2025-10-31, fiscal year ended
2025-09-27).

## Output format

One row per Q&A pair:

| column | meaning |
|---|---|
| `question` | the question, written to stand on its own (names the company/period) |
| `answer` | the answer |
| `source_passage` | the exact text from the filing the answer comes from |
| `question_type` | `fact_extraction`, `numeric_calculation`, `comparison`, or `multi_step_reasoning` |
| `difficulty` | `easy`, `medium`, `hard` |
| `section` | which 10-K section it came from |
| `company`, `ticker`, `fiscal_year`, `accession`, `chunk_id` | provenance, so any row traces back to the exact filing and passage |
| `verdict`, `verifier_confidence`, `grounding_overlap`, `source_verbatim` | verification details |

A few real rows from the shipped file:

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

## What's in the sample

247 verified pairs. 98.4% of generated candidates passed verification, mean
verifier confidence 0.98, and 100% of the source passages are exact substrings
of the filing.

- By type: fact_extraction 144, multi_step_reasoning 57, numeric_calculation 28, comparison 18
- By difficulty: easy 123, medium 97, hard 27
- Top sections: Risk Factors 98, Financial Statements 74, MD&A 24, Business 22

Machine-readable stats are in
[`output/apple_10k_fy2025_stats.json`](output/apple_10k_fy2025_stats.json), and
pairs that failed verification are kept in
[`output/apple_10k_fy2025_rejected.jsonl`](output/apple_10k_fy2025_rejected.jsonl)
rather than thrown away.

## How it works

```
10-K (HTML)  ->  parse  ->  chunk  ->  generate  ->  verify  ->  assemble  ->  CSV/JSONL
```

1. **Download** (`edgar.py`): pulls the 10-K from EDGAR by ticker or CIK, with a
   proper User-Agent and polite rate limiting.
2. **Parse** (`parse.py`): HTML to text. Tables are rendered as
   `Label | col | col` rows so the numbers stay tied to their labels. A lot of
   the good questions come out of tables, so this matters.
3. **Chunk** (`chunk.py`): a 10-K always has the same `Item N.` section
   skeleton, so it splits on those headers and packs each section into roughly
   6k-character windows. This filing produced 43 chunks, each labelled with its
   section.
4. **Generate** (`generate.py`): for each chunk an LLM writes a handful of Q&A
   pairs and quotes the passage that supports each one.
5. **Verify** (`verify.py`): two checks, below.
6. **Assemble** (`pipeline.py`): snap each passage to the exact filing text, drop
   duplicate questions, write the dataset, stats, and rejects.

### Verification

Two independent checks, cheapest first:

1. **Grounding check (no LLM).** Confirms the quoted `source_passage` is really
   present in the chunk, so the model can't invent a quote, and that numbers in a
   factual answer trace back to the source. This catches the worst failure mode,
   a fabricated passage, without spending anything.

2. **LLM check.** A different model (Opus) than the one that wrote the question
   (Sonnet) re-reads the chunk and decides whether the answer is correct and
   supported, recomputing the arithmetic on numeric questions. Using a separate
   model means a misread is less likely to slip through twice.

A pair is kept only if both checks pass and the verdict is `SUPPORTED`.

It catches real mistakes. In this run one pair claimed Apple's Services net sales
grew $13,989M; the verifier recomputed `109,158 - 96,169 = 12,989` and rejected
it for overstating growth by $1,000M. Of 251 candidates, 249 came back
`SUPPORTED`, 2 `PARTIAL`, 1 `NOT_SUPPORTED`.

### Exact source passages

LLMs tend to tidy up a quote (drop the table `|` separators, swap a curly
apostrophe for a straight one). Since the column is meant to be exact text, once
the grounding check passes the pipeline re-aligns the quote to the actual span in
the chunk and stores that. Every passage in the shipped file is a byte-for-byte
substring of the parsed filing. The rare pair whose evidence is split across two
non-adjacent rows (a cross-year table comparison, for example) is dropped by
default; the `require_verbatim_source` setting controls this.

## Running it

Install:

```bash
pip install -r requirements.txt
```

Live run against any filing (needs an Anthropic API key):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --ticker AAPL            # latest 10-K
python run.py --ticker MSFT --skip 1   # the previous one
python run.py --cik 0000320193         # by CIK
```

Reproduce the shipped sample offline, no key or network needed. The model
outputs are cached under `data/llm_runs/`, so the whole pipeline (parse, chunk,
grounding, verifier gate, snapping, dedupe, output) runs locally against them:

```bash
python run.py --from-cache \
  --html data/raw/aapl-20250927.htm \
  --company "Apple Inc." --ticker AAPL --fiscal-year 2025 \
  --accession 0000320193-25-000079 --run-name apple_10k_fy2025
```

Tests:

```bash
python tests/test_chunk.py && python tests/test_verify.py
```

The shipped run uses Claude (Sonnet 4.6 for generation, Opus 4.8 for
verification). Those responses are cached under `data/llm_runs/` so the dataset
rebuilds offline; with `ANTHROPIC_API_KEY` set, `run.py` makes the same calls
live through the Anthropic SDK. Both go through one code path
(`pipeline.run_pipeline`); only the client in `llm.py` differs.

## Layout

```
qa_pipeline/
  config.py      settings (models, chunk sizes, thresholds), all env-overridable
  edgar.py       download a 10-K from EDGAR
  parse.py       HTML to clean, table-aware text
  chunk.py       section detection + windowing
  schema.py      Pydantic models, validates the LLM output
  prompts.py     the generation and verification prompts
  llm.py         client: AnthropicClient (live) or ReplayClient (offline)
  generate.py    chunk -> candidate Q&A
  verify.py      grounding check + LLM verifier + passage snapping
  pipeline.py    orchestration, dedupe, write outputs
  merge_runs.py  merge captured outputs into the replay format
run.py           CLI
tests/           chunking and verification tests
docs/SCALING.md  scaling to many documents / 1000+ pairs
data/            raw filing, chunk manifest, captured model outputs
output/          the dataset, stats, rejected pairs
```

## Design notes

- **Section-aware chunking.** Splitting on the `Item N.` headers gives every
  chunk a real section label and keeps related text together. Table-of-contents
  rows (which contain `|` because they come from a table) are told apart from the
  real headers, and the repeating page footer is stripped.
- **Table-aware parsing.** Numbers stay attached to their row labels, which is
  where most numeric and comparison questions come from.
- **Two different models.** Generating with one and verifying with another is a
  cheap way to avoid correlated mistakes; the verifier is also the stronger model
  and redoes the math.
- **Cheap check before the expensive one.** Fabricated quotes are caught for free,
  so the LLM verifier only spends tokens on plausible pairs.
- **One pipeline, two clients.** The replay client lets the sample rebuild with no
  key; the live client adds a response cache so reruns are cheap and resumable.
- **Keep the rejects.** Failed pairs and per-row verification details stay in the
  output instead of being hidden.

## Limitations

- The type mix leans toward fact extraction (58%). Risk Factors is the biggest
  section and produces a lot of factual questions. Harder numeric and comparison
  questions are there but underrepresented; `docs/SCALING.md` covers how to steer
  generation toward the thinner buckets.
- One verifier is high precision but not perfect. A subtle misread could get
  through. For a benchmark split I'd use a 3-verifier majority (see the scaling
  notes).
- Table linearization loses the 2-D layout. Pipe-separated rows are readable but
  a complex multi-column table can be ambiguous. Cross-checking numbers against
  the filing's XBRL data would make numeric answers airtight.
- Evidence split across non-adjacent passages is dropped rather than cited as
  multiple spans.
- Mostly tested on Apple's (Workiva) HTML. The parser handles common filer
  formats and mis-declared encodings, but unusual layouts may need tweaks.

## Scaling

[`docs/SCALING.md`](docs/SCALING.md) covers running this over many filings and
into the thousands of pairs: queue-based fan-out, the Batch API, tiered models,
cross-document dedup with embeddings, type/difficulty balancing, stratified
filing sampling, and XBRL-based numeric checks.
