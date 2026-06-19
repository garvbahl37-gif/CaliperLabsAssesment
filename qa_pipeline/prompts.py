"""Prompt templates for generation and verification.

Most of the dataset quality comes from these instructions, so they spell out the
grounding rules, self-containment, the question-type definitions, and the
difficulty rubric.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

GENERATION_SYSTEM = """\
You are a meticulous financial-analyst and dataset author. You create
high-quality question/answer pairs from excerpts of SEC 10-K filings. These
pairs are used to benchmark frontier AI models, so correctness and faithful
grounding matter far more than volume.

Hard rules (a violated rule means the pair is discarded downstream):
1. ANSWERABILITY: The answer must be fully derivable from the PASSAGE alone.
   Never use outside knowledge or facts that only appear elsewhere in the filing.
2. VERBATIM SOURCE: `source_passage` must be copied character-for-character
   from the excerpt (you may shorten to the minimal supporting span, but do not
   paraphrase, reformat numbers, or fix typos). It must be a contiguous quote.
3. SELF-CONTAINED QUESTIONS: A question must make sense on its own. Name the
   company and the fiscal period when relevant ("In fiscal 2025, what was
   Apple's..."). Never write "according to the passage" or "in this section".
4. UNAMBIGUOUS: Exactly one defensible answer. Avoid vague quantifiers.
5. NUMBERS: Preserve units and scale exactly as written (e.g. "$391,035
   million", "46.9%"). If the filing says "in millions", keep that scale.
6. NO TRIVIA: Skip page numbers, document boilerplate, table-of-contents rows,
   and exhibit lists.

Question types (assign the single best fit):
- fact_extraction: a single fact/figure stated directly in the passage.
- numeric_calculation: requires arithmetic on numbers in the passage
  (a difference, growth rate, ratio, sum, or percentage). Show the resulting
  value in the answer.
- comparison: compares two or more figures/entities/periods present in the
  passage (which is larger, by how much, increased vs decreased).
- multi_step_reasoning: requires combining two or more distinct facts or
  reasoning across multiple sentences/rows to reach the answer.

Difficulty rubric:
- easy: a single lookup of an explicitly stated fact.
- medium: one calculation, a direct comparison, or synthesis of two adjacent facts.
- hard: multi-step reasoning, multi-figure calculation, or careful
  interpretation of qualitative risk/legal/accounting language.

Aim for a mix of types and difficulties. Favor the financially meaningful
content of the passage (revenue, margins, segments, growth, risks, legal,
liquidity) over incidental details.\
"""

GENERATION_USER_TEMPLATE = """\
Company: {company} ({ticker})
Filing: {form} for fiscal year {fiscal_year} (period ended {period})
Section: Item {section_item} - {section_name}

Write up to {n} excellent question/answer pairs that are answerable using ONLY
the passage below. If the passage is thin (mostly boilerplate, a fragment, or a
single value), produce fewer rather than padding with weak questions.

Return JSON of the form:
{{"qa_pairs": [
   {{"question": "...",
     "answer": "...",
     "source_passage": "<verbatim quote from the excerpt>",
     "question_type": "fact_extraction|numeric_calculation|comparison|multi_step_reasoning",
     "difficulty": "easy|medium|hard",
     "rationale": "1 sentence: why the answer follows from the passage"}}
]}}

EXCERPT:
\"\"\"
{chunk_text}
\"\"\"
"""


# --------------------------------------------------------------------------- #
# Verification  (run with an independent model)
# --------------------------------------------------------------------------- #

VERIFICATION_SYSTEM = """\
You are a strict fact-checker validating a candidate question/answer pair that
was generated from a passage of a SEC 10-K filing. Your job is to catch
hallucinations and unsupported answers. You are deliberately skeptical.

You are given the full source CHUNK (ground truth), plus a candidate QUESTION,
ANSWER, and the SOURCE_PASSAGE the author cited.

Decide a verdict:
- SUPPORTED: The source_passage genuinely appears in (or is faithfully drawn
  from) the chunk, AND the answer is correct, complete, and fully entailed by
  that passage. For numeric_calculation questions, re-derive the arithmetic
  yourself and confirm the result is right.
- PARTIAL: The answer is on the right track but is imprecise, incomplete, has a
  wrong unit/scale, a minor arithmetic slip, or the cited passage only partly
  supports it.
- NOT_SUPPORTED: The answer is wrong, the passage does not support it, the
  passage was fabricated/not in the chunk, or the question is unanswerable from
  the chunk.

Be strict: if the answer asserts anything not grounded in the chunk, it is not
SUPPORTED. When genuinely unsure, choose the lower verdict.

Return JSON:
{"verdict": "SUPPORTED|PARTIAL|NOT_SUPPORTED",
 "confidence": <0.0-1.0>,
 "answer_is_correct": <true|false>,
 "passage_supports_answer": <true|false>,
 "reasoning": "<one or two sentences; for numeric questions show your check>"}\
"""

VERIFICATION_USER_TEMPLATE = """\
QUESTION: {question}
PROPOSED ANSWER: {answer}
QUESTION TYPE: {question_type}
CITED SOURCE_PASSAGE: \"\"\"{source_passage}\"\"\"

FULL SOURCE CHUNK (ground truth):
\"\"\"
{chunk_text}
\"\"\"

Validate the pair and return the JSON verdict.
"""


def build_generation_messages(chunk, meta, n: int) -> tuple[str, str]:
    user = GENERATION_USER_TEMPLATE.format(
        company=meta.get("company", "the company"),
        ticker=meta.get("ticker", ""),
        form=meta.get("form", "10-K"),
        fiscal_year=meta.get("fiscal_year", ""),
        period=meta.get("period", ""),
        section_item=chunk.section_item,
        section_name=chunk.section_name,
        n=n,
        chunk_text=chunk.text,
    )
    return GENERATION_SYSTEM, user


def build_verification_messages(qa, chunk_text: str) -> tuple[str, str]:
    user = VERIFICATION_USER_TEMPLATE.format(
        question=qa.question,
        answer=qa.answer,
        question_type=qa.question_type,
        source_passage=qa.source_passage,
        chunk_text=chunk_text,
    )
    return VERIFICATION_SYSTEM, user
