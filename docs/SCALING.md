# Scaling to many documents and 1000+ Q&A pairs

The pipeline in this repo runs one 10-K and produces ~100–200 verified pairs.
Nothing about the design assumes a single document — below is how I would take
it to thousands of filings and 100k+ pairs without losing the verification
guarantees.

## 1. The unit of work is a (chunk) job, and it is embarrassingly parallel

Every stage is a pure function of its input:

```
filing -> [chunks]            (deterministic, cheap, no LLM)
chunk  -> [candidate Q&A]     (1 LLM call)
(chunk, candidate) -> verdict (1 LLM call, independent model)
```

So the natural scaling unit is a **chunk job** and a **pair job**. A filing
with ~40 chunks is ~40 generation calls + ~200 verification calls. To reach
1000+ pairs you need ~8–10 filings; to reach 100k you need ~1,000 filings —
purely a throughput problem, not a design problem.

Concretely:

- **Fan-out by filing, then by chunk.** A queue (SQS / Celery / Cloud Tasks)
  holds one message per chunk. Workers pull, call the LLM, push candidate pairs
  onto a second “verify” queue. Verification workers pull from that. This is
  exactly the two-stage generate→verify flow in `pipeline.run_pipeline`, just
  backed by a durable queue instead of an in-process loop.
- **Idempotency & resume.** Each LLM call is keyed by a stable id
  (`gen::<chunk_id>`, `verify::<chunk_id>::<i>`) and cached
  (`AnthropicClient` already writes a response cache). Re-running a failed batch
  re-uses everything already done and only fills gaps — the same property that
  lets the shipped sample regenerate offline with `--from-cache`.
- **Rate limits.** EDGAR (10 req/s) and the Anthropic API (token/min limits)
  are the real ceilings. Use a token-bucket limiter, exponential backoff
  (already via `tenacity`), and the **Batch API** for generation/verification —
  ~50% cheaper and built for exactly this fire-and-collect pattern.

## 2. Cost control (the thing that actually bites at 100k pairs)

- **Tiered models.** Generate with a fast, cheap model (Haiku/Sonnet); verify
  with a stronger, independent one (Opus). The config already separates
  `generation_model` from `verification_model`.
- **Deterministic pre-filter before the LLM verifier.** The free grounding
  check (`deterministic_grounding`) rejects fabricated passages and unsupported
  numbers with zero tokens. Only pairs that pass reach the (expensive) LLM
  verifier. At scale this can drop verifier spend by 20–40%.
- **Prompt caching.** The generation system prompt is identical across every
  call; with Anthropic prompt caching the static instructions are billed once
  per cache window instead of per call.
- **Stop early per chunk.** `questions_per_chunk` and a global `target_pairs`
  cap let you buy exactly the dataset size you need.

## 3. Quality & diversity at scale

A bigger dataset is only useful if it is **diverse and clean**:

- **Global dedup, not just per-document.** The `normalized_question_key` dedup
  here is per-run; at scale, push question embeddings into a vector index and
  drop near-duplicates across the whole corpus (cosine > 0.9). Otherwise every
  filing yields the same “What was total net sales?” question.
- **Balance the mix.** Track the running distribution over
  (question_type × difficulty × section) and steer generation toward
  under-represented cells (e.g. ask specifically for `hard / numeric_calculation`
  from MD&A) instead of accepting whatever comes back.
- **Stratified sampling of filings.** Pull across sectors, sizes, and years so
  the benchmark is not all mega-cap tech. EDGAR’s full-text + submissions APIs
  make “N random 10-Ks per SIC code per year” a short query.
- **Second-pass adversarial verification for the held-out test split.** For the
  slice used as an actual benchmark, run 3 independent verifiers and keep only
  unanimous SUPPORTED pairs — higher precision where it matters most.

## 4. Generalizing beyond Apple / beyond 10-Ks

- **Section maps per form type.** `TEN_K_SECTIONS` is a lookup table; add
  `TEN_Q_SECTIONS`, `EIGHT_K_ITEMS`, etc. The parser/chunker are form-agnostic.
- **Filer-specific HTML quirks.** Workiva, Donnelley, and self-filed HTML differ.
  The table-aware parser handles the common cases; a small per-renderer cleanup
  registry covers the long tail. The encoding fallback chain already absorbs the
  most common breakage (mis-declared charsets).
- **XBRL as a cross-check.** 10-Ks ship structured XBRL financial facts. For
  numeric questions, validating the answer against the tagged XBRL value is an
  even stronger, fully-deterministic verifier than re-reading prose — a great
  high-precision addition for the financial-statement sections.

## 5. Operational concerns

- **Provenance on every row.** Each row already carries
  `company / accession / section / chunk_id`, so any pair is traceable back to
  the exact filing and passage — essential for audits and disputes.
- **Schema versioning.** Pin a dataset schema version; when prompts or the rubric
  change, bump it so downstream benchmarks know which generation produced a row.
- **Monitoring.** Track acceptance rate, verifier-confidence distribution, and
  type/difficulty balance per batch. A sudden acceptance-rate drop usually means
  a parser regression on a new filer, not a model problem.

### One-line summary

The hard part — *grounded generation + independent verification with full
provenance* — is already in place and stateless. Scaling is a matter of putting
those stateless stages behind a durable queue, adding global dedup + balancing,
and spending on the Batch API.
