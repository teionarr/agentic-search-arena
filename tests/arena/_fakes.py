"""Shared fakes for arena Tier-A tests — no network, no real handlers, no API keys."""

import pytest

from arena.adapters.base import EvidenceDoc, UnifiedResult
from arena.judge import PairwiseVerdict


class FakeLLM:
    """Injectable LLM. ``complete`` and ``structured`` delegate to supplied callables."""

    def __init__(self, complete_fn=None, structured_fn=None):
        self._complete = complete_fn or (lambda s, u: "A synthesized answer grounded in the evidence provided.")
        self._structured = structured_fn or (lambda s, u, schema: PairwiseVerdict(winner="tie", rationale="equal"))
        self.complete_calls = []
        self.structured_calls = []

    def complete(self, system, user, max_tokens=1024):
        self.complete_calls.append(user)
        return self._complete(system, user)

    def structured(self, system, user, schema, max_tokens=1024):
        self.structured_calls.append(user)
        return self._structured(system, user, schema)


class FakeAdapter:
    """Adapter-like object with an async ``search`` returning a preset UnifiedResult."""

    def __init__(self, name, docs, answer="NATIVE ANSWER SHOULD BE STRIPPED", latency_ms=100.0):
        self.name = name
        self._docs = docs
        self._answer = answer
        self._latency = latency_ms

    async def search(self, query):
        return UnifiedResult(results=list(self._docs), answer=self._answer,
                             latency_ms=self._latency, empty_evidence=len(self._docs) == 0)


@pytest.fixture
def docs_two():
    return [EvidenceDoc(url="http://a", title="A", content="Paris is the capital of France."),
            EvidenceDoc(url="http://b", title="B", content="France's capital city is Paris.")]


def sync_gather(adapters, query):
    """Bypass the thread pool: run adapters' searches synchronously for e2e tests."""
    import asyncio
    out = {}
    for a in adapters:
        loop = asyncio.new_event_loop()
        try:
            out[a.name] = loop.run_until_complete(a.search(query))
        finally:
            loop.close()
    return out
