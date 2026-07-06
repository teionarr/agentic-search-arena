"""Thin Anthropic-only LLM client shared by the reader and judge.

Reader and judge both default to the same Claude model (§9: ``reader.model = null`` ->
judge model). The optional OpenAI accuracy grader is the reused ``CorrectnessEvaluator``
as-is and does NOT go through here, so there is no dual-backend abstraction and no mandatory
OpenAI dependency on the core path.

Transport policy: bounded retry with exponential backoff (handles 429 under concurrency),
then give up and return ``None`` — callers treat ``None`` as "skip this call". The static
system prompts are cached (prompt caching) to cut input cost. Token usage is accumulated so a
run can report its real dollar cost. The model id is pinned + recorded in the run snapshot.
"""

import logging
import threading
import time
from typing import Any, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pinned default. Configurable via ArenaConfig; recorded in results.json for reproducibility.
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 4

# USD per 1M tokens (Sonnet-class defaults). Recorded with the run; edit if the model changes.
PRICING = {
    "input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30,
}


def _is_retryable(e: Exception) -> bool:
    """Retry only transient failures: 429 (rate limit), 5xx, and connection/timeouts.
    A 400 (bad request / insufficient credits), 401/403 (auth) will fail identically — don't
    waste attempts on them."""
    status = getattr(e, "status_code", None)
    if status is None:
        return True  # connection error / timeout (no HTTP status) -> transient
    return status == 429 or status >= 500


class LLMClient:
    """Wraps the Anthropic SDK: prompt caching, retry+backoff, structured tool-use, usage."""

    def __init__(self, model: str = DEFAULT_MODEL, client: Any = None, max_retries: int = MAX_RETRIES):
        self.model = model
        self.max_retries = max_retries
        self._client = client
        self._lock = threading.Lock()
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "calls": 0}

    def _get_client(self) -> Any:
        # Lock the lazy init: one LLMClient is shared across pipelined worker threads, so an
        # unlocked first-use could race and build several clients. Own the retry/timeout budget
        # here (the SDK's own default 10-min timeout + 2 retries would otherwise stack on top of
        # this class's bounded-retry loop and tie a worker up far longer than intended).
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import anthropic  # lazy: no key needed just to import arena
                    self._client = anthropic.Anthropic(timeout=60.0, max_retries=0)
        return self._client

    def _record_usage(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        with self._lock:
            self.usage["input"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output"] += getattr(u, "output_tokens", 0) or 0
            self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.usage["calls"] += 1

    def _create(self, system: str, user: str, max_tokens: int, tools=None, tool_choice=None) -> Optional[Any]:
        """One API call with bounded retry + exponential backoff. Returns the response or None."""
        cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        kwargs = dict(model=self.model, max_tokens=max_tokens, temperature=0.0,
                      system=cached_system, messages=[{"role": "user", "content": user}])
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._get_client().messages.create(**kwargs)
                self._record_usage(resp)
                return resp
            except Exception as e:
                last_err = e
                retryable = _is_retryable(e)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): "
                               f"{e.__class__.__name__}" + ("" if retryable else " (not retryable)"))
                if not retryable or attempt == self.max_retries - 1:
                    break
                time.sleep(min(2 ** attempt, 8))  # backoff for 429 / transient errors
        logger.error(f"LLM call gave up: {last_err}")
        return None

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
        """Free-text completion (used by the reader). None => skip."""
        resp = self._create(system, user, max_tokens)
        if resp is None:
            return None
        parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "text", None)]
        return "".join(parts).strip()

    def structured(self, system: str, user: str, schema: Type[BaseModel],
                   max_tokens: int = 1024) -> Optional[BaseModel]:
        """Structured output via forced tool-use, validated into a pydantic model. None => skip."""
        tool = {"name": "submit_verdict", "description": "Submit the structured verdict.",
                "input_schema": schema.model_json_schema()}
        resp = self._create(system, user, max_tokens, tools=[tool],
                            tool_choice={"type": "tool", "name": "submit_verdict"})
        if resp is None:
            return None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                try:
                    return schema.model_validate(block.input)
                except Exception as e:
                    logger.warning(f"Verdict validation failed: {e.__class__.__name__}")
                    return None
        return None

    def cost_usd(self) -> float:
        """Dollar cost of this client's calls so far, from accumulated usage × PRICING."""
        u = self.usage
        return round(
            u["input"] / 1e6 * PRICING["input"]
            + u["output"] / 1e6 * PRICING["output"]
            + u["cache_write"] / 1e6 * PRICING["cache_write"]
            + u["cache_read"] / 1e6 * PRICING["cache_read"], 4)
