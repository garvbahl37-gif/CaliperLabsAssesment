"""
qa_pipeline
===========

An automated pipeline that turns a SEC 10-K filing into a large, verified
dataset of question / answer pairs for benchmarking LLMs on financial documents.

Stages
------
1. edgar    : resolve and download a 10-K from SEC EDGAR
2. parse    : HTML to clean, table-aware plain text
3. chunk    : split the filing into section-labelled chunks
4. generate : LLM writes grounded Q&A pairs for each chunk
5. verify   : a separate, independent check confirms each answer is supported
              by the source passage
6. assemble : dedupe and write a structured dataset

See README.md for the design write-up.
"""

__version__ = "0.1.0"
