"""Optional Langfuse tracing (§11) — OFF by default, redact-gated.

When Langfuse keys are present in secrets (``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
``LANGFUSE_HOST``) AND config ``langfuse.enabled=true``, each query becomes one trace with
three span types: ``provider.search`` (raw results + latency), ``reader.synthesize`` (the exact
context the model saw), and ``judge.compare`` (verdict + rationale, both order passes). Missing
keys silently disable tracing even when enabled — no error.

Security invariant: every span payload passes through the existing ``report.redact()`` boundary
before it reaches the Langfuse client, so no resolved secret value is ever exported. The pipeline
always calls the same thin interface; when tracing is off it holds a ``NullTracer`` whose spans
are inert, so there is exactly one code path with no per-call ``if tracing`` branching.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from arena import secrets
from arena.report import redact

logger = logging.getLogger(__name__)

# Secrets that gate real tracing. All three must be present (host has a client-side default
# but §11 keys it on presence, so we require it too — absence => silently disabled).
_LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


class Span:
    """A thin span handle. A ``NullSpan`` is inert; a ``LangfuseSpan`` wraps a real observation.

    ``child(name, ...)`` opens a nested span; ``end(output=...)`` closes it. Payloads are
    redacted at the boundary (see ``LangfuseSpan``), so callers pass plain dicts freely.
    """

    def child(self, name: str, input: Any = None, output: Any = None) -> "Span":
        return NullSpan()

    def end(self, output: Any = None) -> None:
        pass


class NullSpan(Span):
    """No-op span used when tracing is disabled — every method is a cheap return."""


class Tracer:
    """The interface the pipeline calls either way. Base impl is the no-op tracer."""

    enabled = False

    def trace(self, name: str, input: Any = None) -> Span:
        """Open one trace (root span) for a query. Returns a ``Span`` to nest under."""
        return NullSpan()

    def flush(self) -> None:
        pass


class NullTracer(Tracer):
    """Explicit no-op tracer. Holds no client and makes zero network calls."""


@dataclass
class LangfuseSpan(Span):
    """Wraps a real Langfuse observation. All input/output payloads are redacted here — this
    is the single boundary between arena data and the Langfuse client."""

    _obs: Any
    _secret_values: List[str] = field(default_factory=list)

    def child(self, name: str, input: Any = None, output: Any = None) -> "Span":
        try:
            obs = self._obs.start_observation(
                name=name,
                input=redact(input, self._secret_values) if input is not None else None,
                output=redact(output, self._secret_values) if output is not None else None,
            )
            return LangfuseSpan(_obs=obs, _secret_values=self._secret_values)
        except Exception as e:  # tracing must never break the run
            logger.warning(f"Langfuse child span failed ({e.__class__.__name__}); continuing")
            return NullSpan()

    def end(self, output: Any = None) -> None:
        try:
            if output is not None:
                self._obs.update(output=redact(output, self._secret_values))
            self._obs.end()
        except Exception as e:
            logger.warning(f"Langfuse span end failed ({e.__class__.__name__}); continuing")


@dataclass
class LangfuseTracer(Tracer):
    """Real tracer. Opens one trace per query; spans nest under it. Never raises to the caller."""

    _client: Any
    _secret_values: List[str] = field(default_factory=list)
    enabled: bool = True

    def trace(self, name: str, input: Any = None) -> Span:
        try:
            obs = self._client.start_observation(
                name=name,
                input=redact(input, self._secret_values) if input is not None else None,
            )
            return LangfuseSpan(_obs=obs, _secret_values=self._secret_values)
        except Exception as e:
            logger.warning(f"Langfuse trace start failed ({e.__class__.__name__}); continuing")
            return NullSpan()

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed ({e.__class__.__name__})")


def build_tracer(enabled: bool, client: Optional[Any] = None) -> Tracer:
    """Return a ``LangfuseTracer`` only when ``enabled`` AND all Langfuse keys are present.

    Any other case — disabled, missing keys, or an import/auth failure — yields a ``NullTracer``
    silently (no error), so the pipeline calls one identical interface either way. ``client`` is
    an injection seam for tests; production leaves it None and builds a real client.
    """
    if not enabled:
        return NullTracer()
    if client is None and not all(secrets.has(k) for k in _LANGFUSE_KEYS):
        # Enabled but keys absent -> silently disabled (§11).
        return NullTracer()

    secret_values = _resolved_secret_values()
    if client is None:
        try:
            from langfuse import Langfuse  # lazy: no dependency unless tracing is actually used
            client = Langfuse(
                public_key=secrets.get_secret("LANGFUSE_PUBLIC_KEY"),
                secret_key=secrets.get_secret("LANGFUSE_SECRET_KEY"),
                host=secrets.get_secret("LANGFUSE_HOST"),
            )
            # auth_check verifies credentials; a False/raising result -> disable silently.
            if hasattr(client, "auth_check") and not client.auth_check():
                logger.info("Langfuse auth check failed; tracing disabled")
                return NullTracer()
        except Exception as e:
            logger.info(f"Langfuse unavailable ({e.__class__.__name__}); tracing disabled")
            return NullTracer()

    return LangfuseTracer(_client=client, _secret_values=secret_values)


def _resolved_secret_values() -> List[str]:
    """Resolved secret values used to scrub span payloads at the redact boundary — provider
    keys plus the Langfuse keys themselves (so a key can never appear inside a span)."""
    import os
    vals = []
    for k, v in os.environ.items():
        if v and (k.endswith("_API_KEY") or k in ("OPENAI_API_KEY",) or k in _LANGFUSE_KEYS):
            vals.append(v)
    return vals
