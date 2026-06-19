"""Generation stage: turn one chunk into candidate Q&A pairs via the LLM."""

from __future__ import annotations

from typing import Optional

from pydantic import ValidationError

from .chunk import Chunk
from .config import Config, DEFAULT
from .llm import BaseClient
from .prompts import build_generation_messages
from .schema import GENERATION_JSON_SCHEMA, GeneratedQA
from .utils import get_logger

log = get_logger()


def generate_for_chunk(
    chunk: Chunk,
    meta: dict,
    client: BaseClient,
    cfg: Config = DEFAULT,
) -> list[GeneratedQA]:
    system, user = build_generation_messages(chunk, meta, cfg.questions_per_chunk)
    data = client.complete_json(
        model=cfg.generation_model,
        system=system,
        user=user,
        schema=GENERATION_JSON_SCHEMA,
        key=f"gen::{chunk.chunk_id}",
    )
    out: list[GeneratedQA] = []
    for raw in data.get("qa_pairs", []):
        try:
            out.append(GeneratedQA(**raw))
        except ValidationError as e:
            log.warning("Dropping malformed generated pair in %s: %s",
                        chunk.chunk_id, e.errors()[0].get("msg", "invalid"))
    return out
