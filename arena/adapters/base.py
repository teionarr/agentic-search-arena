"""Unified adapter shape and the composition shim over the base ProviderHandler.

The `UnifiedResult` dataclass is one of the two frozen, versioned contracts of the arena
(the other is the ``results.json`` schema in ``arena.report``). Everything downstream —
reader, judge, metrics, report — conforms to this shape, so it is defined once here.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Bump if the UnifiedResult shape changes in a backward-incompatible way.
UNIFIED_RESULT_VERSION = "1.0"


@dataclass
class EvidenceDoc:
    """One returned search result, normalized across providers."""

    url: str
    title: str
    content: str
    score: Optional[float] = None
    published_date: Optional[str] = None


@dataclass
class UnifiedResult:
    """The normalized output of a single provider search for a single query.

    ``answer`` holds a provider-native answer when present, but in M0 it is stripped
    before the reader/judge ever see it (the arena ranks the retrieval layer). ``results``
    is the evidence the reader synthesizes from.
    """

    results: List[EvidenceDoc] = field(default_factory=list)
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
    cost_units: Optional[float] = None
    raw: Any = None
    needs_synthesis: bool = True
    # Set by the adapter when the provider returned no usable evidence for this query.
    empty_evidence: bool = False


class HandlerAdapter:
    """Composes an existing ``ProviderHandler`` and normalizes its output.

    We reuse the handler's ``.search()`` for the network call, auth, and latency, then map
    the raw ``search_response`` into the ``UnifiedResult`` shape via a per-provider
    ``normalize_fn``. This is the single sanctioned provider-specific code path.
    """

    def __init__(self, name: str, handler: Any, normalize_fn) -> None:
        self.name = name
        self._handler = handler
        self._normalize_fn = normalize_fn
        self.min_interval_s = 0.0  # min seconds between requests (set from the registry spec)

    async def search(self, query: str) -> UnifiedResult:
        """Run the provider search and normalize it. Never raises past this boundary."""
        try:
            raw = await self._handler.search(query)
        except Exception as e:  # match the base's sentinel-not-raise idiom
            logger.error(f"[{self.name}] search raised: {e}")
            return UnifiedResult(raw={"error": str(e)}, empty_evidence=True)

        latency = raw.get("provider_latency") if isinstance(raw, dict) else None
        latency_ms = latency * 1000.0 if latency is not None else None

        docs = self._normalize_fn(raw)
        # Uniform per-provider log (the base handlers are inconsistent — Tavily/Exa are silent).
        lat = f"{latency_ms:.0f}ms" if latency_ms is not None else "no-latency"
        if docs:
            logger.info(f"[{self.name}] {len(docs)} results, {lat}")
        else:
            logger.warning(f"[{self.name}] returned no usable evidence, {lat}")
        return UnifiedResult(
            results=docs,
            answer=None,  # native answer intentionally discarded in M0 (forced synthesis)
            latency_ms=latency_ms,
            cost_units=None,
            raw=raw,
            needs_synthesis=True,
            empty_evidence=len(docs) == 0,
        )
