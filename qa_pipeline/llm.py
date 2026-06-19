"""LLM transport layer.

The pipeline talks to exactly one tiny interface::

    client.complete_json(model, system, user, schema, key) -> dict

Two implementations share that interface so the *entire* pipeline (parsing,
generation parsing, verification, dedup, balancing, output) runs identically
whether or not a live API key is present:

* ``AnthropicClient``  -- calls the real Claude API with tool-forced structured
  output and a transparent on-disk response cache. This is the path used in
  production (``python run.py --ticker AAPL``).

* ``ReplayClient``     -- serves previously-captured model outputs from disk,
  keyed by a stable semantic id (``gen::<chunk>``, ``verify::<chunk>::<i>``).
  This lets the shipped sample dataset be regenerated end-to-end with no API
  key and no network, while still exercising every line of pipeline logic.

The ``key`` argument is what makes the two interchangeable: callers pass a
stable semantic id, the live client uses it only for caching, and the replay
client uses it to look up the captured response.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

from .config import Config, DEFAULT
from .utils import ensure_dir, get_logger, stable_id

log = get_logger()


class LLMError(RuntimeError):
    pass


class BaseClient:
    def complete_json(
        self, *, model: str, system: str, user: str, schema: dict, key: str
    ) -> dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Live Anthropic client
# --------------------------------------------------------------------------- #
class AnthropicClient(BaseClient):
    def __init__(self, cfg: Config = DEFAULT, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self._client = None  # lazy import so the SDK is optional

    def _anthropic(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError(
                    "anthropic SDK not installed. `pip install anthropic` or "
                    "run with --from-cache to use captured outputs."
                ) from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Set it, or run with "
                    "--from-cache to replay captured model outputs."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _cache_path(self, model: str, system: str, user: str) -> str:
        h = stable_id(model, system, user, length=24)
        return os.path.join(self.cfg.cache_dir, f"{h}.json")

    def complete_json(self, *, model, system, user, schema, key) -> dict:
        if self.use_cache:
            cp = self._cache_path(model, system, user)
            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as f:
                    return json.load(f)

        client = self._anthropic()
        # Force structured output via a single-tool definition. The model must
        # call `emit` with arguments matching `schema`, so we get valid JSON.
        tool = {
            "name": "emit",
            "description": "Return the structured result.",
            "input_schema": schema,
        }
        resp = client.messages.create(
            model=model,
            max_tokens=self.cfg.max_output_tokens,
            temperature=self.cfg.temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        data = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = block.input
                break
        if data is None:
            raise LLMError("Model did not return a tool_use block")

        if self.use_cache:
            cp = self._cache_path(model, system, user)
            ensure_dir(os.path.dirname(cp))
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return data


# --------------------------------------------------------------------------- #
# Offline replay client
# --------------------------------------------------------------------------- #
class ReplayClient(BaseClient):
    """Serves captured model outputs from ``data/llm_runs/<chunk_id>.json``.

    Each per-chunk file looks like::

        {"chunk_id": "...",
         "generation": {"qa_pairs": [...]},
         "verifications": [ {verdict, confidence, ...}, ... ]}

    and is mapped to the semantic keys the pipeline asks for.
    """

    def __init__(self, runs_dir: str = "data/llm_runs"):
        self.store: dict[str, dict] = {}
        files = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            cid = rec["chunk_id"]
            self.store[f"gen::{cid}"] = rec.get("generation", {"qa_pairs": []})
            for i, v in enumerate(rec.get("verifications", [])):
                self.store[f"verify::{cid}::{i}"] = v
        log.info("ReplayClient loaded %d captured responses from %s",
                 len(self.store), runs_dir)

    def complete_json(self, *, model, system, user, schema, key) -> dict:
        if key not in self.store:
            raise LLMError(
                f"No captured response for key {key!r}. The capture run may be "
                f"incomplete."
            )
        return self.store[key]


def build_client(cfg: Config = DEFAULT, from_cache: bool = False) -> BaseClient:
    if from_cache:
        return ReplayClient(os.path.join(cfg.data_dir, "llm_runs"))
    return AnthropicClient(cfg)
