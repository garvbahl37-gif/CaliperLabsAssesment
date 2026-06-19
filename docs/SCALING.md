# Scaling to many documents and 1000+ pairs

The pipeline here runs one 10-K and produces a couple hundred verified pairs.
None of the design assumes a single document, so the notes below are about
throughput, cost, and quality once you go from one filing to thousands.

## The work is already parallel

Every stage is a pure function of its input:

```
filing -> [chunks]              deterministic, cheap, no LLM
chunk  -> [candidate Q&A]       one LLM call
(chunk, candidate) -> verdict   one LLM call, independent model
```

So the natural unit of work is a chunk job and then a pair job. A filing with
~40 chunks is ~40 generation calls plus ~200 verification calls. Getting to
1000+ pairs means ~8-10 filings; 100k pairs means ~1,000 filings. That's a
throughput problem, not a design problem.

Concretely:

- **Fan out by filing, then by chunk.** Put one message per chunk on a queue
  (SQS, Celery, Cloud Tasks). Workers generate and push candidate pairs onto a
  second queue; verification workers read from that. It's the same two-stage
  generate-then-verify flow as `pipeline.run_pipeline`, just backed by a durable
  queue instead of an in-process loop.
- **Make it resumable.** Each LLM call has a stable key (`gen::<chunk_id>`,
  `verify::<chunk_id>::<i>`) and the live client already caches responses to
  disk. Re-running a failed batch reuses everything that finished and only fills
  the gaps. That's the same mechanism that lets the shipped sample rebuild
  offline.
- **Respect the rate limits.** EDGAR (about 10 req/s) and the API token limits
  are the real ceiling. Use a token bucket, exponential backoff (already wired
  through `tenacity`), and the Batch API for generation and verification, which
  is cheaper and built for this fire-and-collect pattern.

## Cost

- **Tiered models.** Generate with a cheaper model, verify with a stronger,
  independent one. `config.py` already separates `generation_model` from
  `verification_model`.
- **Filter before the LLM verifier.** The grounding check rejects fabricated
  passages and unsupported numbers for zero tokens, so only plausible pairs reach
  the expensive verifier. At scale that's a real chunk of the verifier bill.
- **Prompt caching.** The generation system prompt is identical on every call, so
  with prompt caching the static instructions are billed once per cache window
  instead of per call.
- **Stop when you have enough.** `questions_per_chunk` and a global
  `target_pairs` let you buy exactly the dataset size you need.

## Quality and diversity

A bigger dataset is only useful if it stays clean and varied.

- **Dedup across the whole corpus, not just per filing.** The dedup here is
  per-run. At scale, push question embeddings into a vector index and drop near
  duplicates across everything (cosine > 0.9), otherwise every filing gives you
  the same "what was total net sales?" question.
- **Balance the mix.** Track the running distribution over
  type x difficulty x section and steer generation toward the thin cells (for
  example, ask specifically for hard numeric questions from MD&A) instead of
  taking whatever comes back. This is also the fix for the fact-extraction skew
  in the current sample.
- **Sample filings on purpose.** Pull across sectors, sizes, and years so the
  benchmark isn't all mega-cap tech. EDGAR's submissions and full-text APIs make
  "N random 10-Ks per SIC code per year" a short query.
- **Stronger verification for a held-out split.** For the slice used as an actual
  benchmark, run three independent verifiers and keep only unanimous SUPPORTED
  pairs. Higher precision where it matters most.

## Beyond Apple and beyond 10-Ks

- **Section maps per form.** `TEN_K_SECTIONS` is just a lookup table; add ones
  for 10-Q, 8-K, and so on. The parser and chunker are form-agnostic.
- **Filer quirks.** Workiva, Donnelley, and self-filed HTML differ. The
  table-aware parser handles the common cases; a small per-renderer cleanup
  registry covers the long tail. The encoding fallback already absorbs the most
  common breakage (mis-declared charsets).
- **XBRL as a cross-check.** 10-Ks ship structured XBRL financial facts. For
  numeric questions, checking the answer against the tagged XBRL value is an even
  stronger, fully deterministic verifier than re-reading prose, and a good fit
  for the financial-statement sections.

## Operations

- **Provenance on every row.** Each row already carries
  company / accession / section / chunk_id, so any pair traces back to the exact
  filing and passage. That's necessary for audits and disputes.
- **Schema versioning.** Pin a dataset schema version and bump it when the
  prompts or rubric change, so downstream consumers know which generation
  produced a row.
- **Monitoring.** Watch acceptance rate, verifier-confidence distribution, and
  type/difficulty balance per batch. A sudden drop in acceptance usually means a
  parser regression on a new filer, not a model problem.

The hard part, grounded generation with independent verification and full
provenance, is already here and stateless. Scaling is mostly putting those
stages behind a queue, adding cross-document dedup and balancing, and moving the
bulk calls to the Batch API.
