"""Thin LLM clients shared by the reader and judge: Anthropic by default, OpenAI for §5.

Reader and judge both default to the same Claude model (§9: ``reader.model = null`` ->
judge model). The optional OpenAI accuracy grader is the reused ``CorrectnessEvaluator``
as-is and does NOT go through here, so there is no dual-backend abstraction and no mandatory
OpenAI dependency on the core path. The one cross-family entry point is the secondary judge
(§5): ``build_llm_client("openai:<model>")`` returns an :class:`OpenAIClient` with the same
interface, so native-answer pairs can be routed to a genuinely non-Claude judge.

Transport policy: bounded retry with exponential backoff (handles 429 under concurrency),
then give up and return ``None`` — callers treat ``None`` as "skip this call". The static
system prompts are cached (prompt caching) to cut input cost. Token usage is accumulated so a
run can report its real dollar cost. The model id is pinned + recorded in the run snapshot.
"""

import logging
import os
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


class OpenAIClient:
    """OpenAI-family drop-in for :class:`LLMClient` — the §5 cross-family secondary judge.

    Same interface (``complete`` / ``structured`` / ``cost_usd``) so judge and pipeline code
    cannot tell the families apart. Built on the base repo's existing ``langchain_openai``
    dependency (lazy import, like the Anthropic SDK above); no new dependency. Requires
    ``OPENAI_API_KEY`` at call time — a missing key raises immediately (config asked for a
    cross-family judge; silently skipping would fake the §5 mitigation).
    """

    def __init__(self, model: str, client: Any = None, max_retries: int = MAX_RETRIES):
        self.model = model
        self.max_retries = max_retries
        self._client = client
        self._lock = threading.Lock()
        self.usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "calls": 0}

    def _get_client(self) -> Any:
        # Same locked lazy init as LLMClient (shared across pipelined worker threads). Own the
        # retry budget here too: max_retries=0 so langchain's built-in retries don't stack on
        # this class's bounded-retry loop.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise RuntimeError(
                            "OPENAI_API_KEY is not set but the config requested an "
                            "OpenAI-family judge (judge.secondary: 'openai:...'). Add the key "
                            "to your .env, or use a Claude model id for the secondary judge.")
                    from langchain_openai import ChatOpenAI  # lazy: core path never needs OpenAI
                    self._client = ChatOpenAI(model=self.model, temperature=0.0,
                                              timeout=60.0, max_retries=0)
        return self._client

    def _invoke(self, runnable: Any, messages: list) -> Optional[Any]:
        """One call with bounded retry + backoff mirroring LLMClient._create. None => skip."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                result = runnable.invoke(messages)
                with self._lock:
                    self.usage["calls"] += 1
                return result
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
        """Free-text completion. None => skip. (Kept for interface parity; the secondary judge
        only calls structured().)"""
        client = self._get_client().bind(max_tokens=max_tokens)  # key check raises, retries don't mask it
        resp = self._invoke(client, [("system", system), ("user", user)])
        if resp is None:
            return None
        content = getattr(resp, "content", None)
        return content.strip() if isinstance(content, str) else None

    def structured(self, system: str, user: str, schema: Type[BaseModel],
                   max_tokens: int = 1024) -> Optional[BaseModel]:
        """Structured output via langchain's tool-calling path, as a validated pydantic model
        instance — the SAME schema instances the Anthropic client returns. None => skip."""
        runnable = self._get_client().with_structured_output(schema)
        result = self._invoke(runnable, [("system", system), ("user", user)])
        if result is None:
            return None
        if isinstance(result, schema):
            return result
        try:  # dict fallback (e.g. include_raw-style payloads); validate, never trust blindly
            return schema.model_validate(result)
        except Exception as e:
            logger.warning(f"Verdict validation failed: {e.__class__.__name__}")
            return None

    def cost_usd(self) -> float:
        """Always 0.0: OpenAI per-token cost tracking is not wired (PRICING above is
        Anthropic-only). Reporting 0 keeps the run's dollar figure honest-by-omission —
        the secondary judge's OpenAI spend is uncounted, never fabricated."""
        return 0.0


# Model-id prefix convention for cross-family judges (§5).
OPENAI_PREFIX = "openai:"
CLAUDE_PREFIX = "claude:"


def build_llm_client(model_id: str, **kwargs) -> Any:
    """Factory: route a (possibly prefixed) model id to the right client family.

    ``"openai:<model>"`` -> :class:`OpenAIClient`; ``"claude:<model>"`` or a bare id ->
    :class:`LLMClient`. Used wherever config supplies a judge model id (judge.secondary),
    so the §5 secondary judge can be a genuinely different model family.
    """
    if model_id.startswith(OPENAI_PREFIX):
        return OpenAIClient(model=model_id[len(OPENAI_PREFIX):], **kwargs)
    if model_id.startswith(CLAUDE_PREFIX):
        model_id = model_id[len(CLAUDE_PREFIX):]
    return LLMClient(model=model_id, **kwargs)
