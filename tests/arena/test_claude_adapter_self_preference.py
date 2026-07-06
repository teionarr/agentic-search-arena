"""Claude web-search adapter (native-answer path) + self-preference labeling (§5/§6/§12).

Tier-A: deterministic, no AI, no live Anthropic calls (client is mocked with a canned payload).
"""

import asyncio
import json
import os

from arena.adapters import normalize
from arena.adapters.base import EvidenceDoc, HandlerAdapter, UnifiedResult
from arena.adapters.claude_search_handler import ClaudeSearchHandler, _serialize
from arena.adapters.registry import (REGISTRY, claude_family_providers,
                                     native_answer_providers)
from arena.judge import PairwiseVerdict, judge_pair
from arena.self_preference import POSSIBLE_SELF_PREFERENCE, self_preference_label


# ---- a canned Anthropic Messages response (dict blocks; no live call) ----

def _canned_response():
    return {
        "content": [
            {"type": "text", "text": "The capital of France is Paris."},
            {"type": "server_tool_use", "name": "web_search", "input": {"query": "capital of France"}},
            {"type": "web_search_tool_result", "content": [
                {"type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Paris",
                 "title": "Paris - Wikipedia", "page_age": "2026-01-01",
                 "encrypted_content": "ENCRYPTEDBLOB=="},
                {"type": "web_search_result", "url": "https://example.com/france",
                 "title": "France facts", "page_age": None,
                 "encrypted_content": "ENCRYPTEDBLOB2=="},
                {"type": "web_search_tool_result_error", "error_code": "unavailable"},
            ]},
        ],
    }


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def create(self, **kwargs):  # AsyncAnthropic: create is awaitable
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropic:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


# ---- _serialize: response blocks -> {answer, results} ----

def test_serialize_extracts_answer_and_results():
    flat = _serialize(_canned_response())
    assert flat["answer"] == "The capital of France is Paris."
    assert [r["url"] for r in flat["results"]] == [
        "https://en.wikipedia.org/wiki/Paris", "https://example.com/france"]  # error block skipped
    assert flat["results"][0]["page_age"] == "2026-01-01"


# ---- handler.search: canned payload -> {answer, search_response, provider_latency} ----

def test_claude_handler_maps_canned_payload():
    client = _FakeAnthropic(_canned_response())
    handler = ClaudeSearchHandler(client=client)
    raw = asyncio.run(handler.search("what is the capital of France?"))
    assert raw["answer"] == "The capital of France is Paris."
    assert raw["provider_latency"] is not None
    assert raw["search_response"]["answer"] == "The capital of France is Paris."
    assert len(raw["search_response"]["results"]) == 2
    # web_search tool wired into the create() call.
    assert client.messages.calls[0]["tools"][0]["name"] == "web_search"


def test_claude_handler_error_is_sentinel_not_raise():
    class Boom:
        class messages:
            @staticmethod
            async def create(**kwargs):
                raise RuntimeError("network down")
    handler = ClaudeSearchHandler(client=Boom())
    raw = asyncio.run(handler.search("q"))
    assert raw == {"answer": "", "search_response": None, "provider_latency": None}


# ---- adapter: native-answer path preserves answer + needs_synthesis=False ----

def test_native_answer_adapter_maps_to_unified_shape():
    client = _FakeAnthropic(_canned_response())
    handler = ClaudeSearchHandler(client=client)
    adapter = HandlerAdapter("claude_search", handler, normalize.normalize_claude_search,
                             native_answer=True)
    res = asyncio.run(adapter.search("q"))
    assert isinstance(res, UnifiedResult)
    assert res.answer == "The capital of France is Paris."   # native answer preserved
    assert res.needs_synthesis is False                       # native-answer path
    assert res.latency_ms is not None
    assert res.empty_evidence is False
    assert [d.url for d in res.results] == [
        "https://en.wikipedia.org/wiki/Paris", "https://example.com/france"]


def test_retrieval_adapters_still_strip_answer():
    # Regression: default (non-native) adapters must keep the primary path unchanged.
    class FakeHandler:
        async def search(self, q):
            return {"answer": "should be dropped",
                    "search_response": {"results": [{"url": "http://x", "content": "c"}]},
                    "provider_latency": 0.5}
    adapter = HandlerAdapter("tavily", FakeHandler(), normalize.normalize_tavily)
    res = asyncio.run(adapter.search("q"))
    assert res.answer is None and res.needs_synthesis is True


# ---- normalizer: claude search results -> EvidenceDoc[] ----

def test_normalize_claude_search():
    raw = {"search_response": {"answer": "a", "results": [
        {"url": "http://x", "title": "Ti", "page_age": "2026-01-01"}]},
        "provider_latency": 0.2}
    docs = normalize.normalize_claude_search(raw)
    assert docs == [EvidenceDoc(url="http://x", title="Ti", content="Ti",
                                published_date="2026-01-01")]


def test_normalize_claude_search_registered_and_none_safe():
    assert "claude_search" in normalize.NORMALIZERS
    assert normalize.normalize_claude_search({"search_response": None}) == []


def test_normalize_claude_search_keeps_url_only_results():
    # A result with a valid url but empty/missing title must be kept, not silently dropped.
    raw = {"search_response": {"results": [
        {"url": "http://x", "title": ""},        # empty title
        {"url": "http://y"},                       # missing title
        {"title": "no url"}]}}                      # no url -> dropped
    docs = normalize.normalize_claude_search(raw)
    assert [d.url for d in docs] == ["http://x", "http://y"]
    assert all(d.content == d.title for d in docs)


# ---- registry: one entry, marked native + claude-family ----

def test_registry_marks_claude_search_native_and_family():
    spec = REGISTRY["claude_search"]
    assert spec.native_answer is True
    assert spec.family == "claude"
    assert "claude_search" in native_answer_providers()
    assert "claude_search" in claude_family_providers()


# ---- self-preference labeling rule (§5/§6/§14) ----

def _label(a, b, a_native, b_native, judge_claude=True, secondary=False,
           family=frozenset({"claude_search"})):
    return self_preference_label(a, b, a_native, b_native, set(family), judge_claude, secondary)


def test_native_claude_pair_is_flagged():
    assert _label("claude_search", "tavily", True, False) == POSSIBLE_SELF_PREFERENCE


def test_primary_path_pair_is_never_flagged():
    # Same providers, but claude side is NOT on the native-answer path -> no label.
    assert _label("claude_search", "tavily", False, False) is None
    # Two non-claude providers, native or not -> no label.
    assert _label("tavily", "exa", True, True) is None


def test_secondary_judge_routes_away_no_label():
    assert _label("claude_search", "tavily", True, False, secondary=True) is None


def test_non_claude_judge_no_label():
    assert _label("claude_search", "tavily", True, False, judge_claude=False) is None


# ---- judge_pair carries the label through unchanged mechanics ----

def _pair():
    da = [EvidenceDoc(url="a", title="t", content="e")]
    db = [EvidenceDoc(url="b", title="t", content="e")]
    return ({"answer": "ans x", "docs": da}, {"answer": "ans y", "docs": db})


def test_judge_pair_attaches_self_preference_label():
    from _fakes import FakeLLM
    seq = iter([PairwiseVerdict(winner="A", rationale="r1"),
                PairwiseVerdict(winner="B", rationale="r2")])  # consistent x, no flip
    llm = FakeLLM(structured_fn=lambda s, u, sch: next(seq))
    x, y = _pair()
    out = judge_pair(llm, "q", x, y, nonce="n", order_swap=True,
                     self_preference=POSSIBLE_SELF_PREFERENCE)
    assert out["self_preference"] == POSSIBLE_SELF_PREFERENCE
    assert out["outcome"] == "x"  # labeling does not alter the verdict


def test_judge_pair_no_label_by_default():
    from _fakes import FakeLLM
    llm = FakeLLM(structured_fn=lambda s, u, sch: PairwiseVerdict(winner="tie", rationale="eq"))
    x, y = _pair()
    out = judge_pair(llm, "q", x, y, nonce="n")
    assert out["self_preference"] is None


# ---- pipeline: caveat gating on claude_native_mode (§5/§6) ----

def _native_adapter(name, docs):
    from _fakes import FakeAdapter
    a = FakeAdapter(name, docs)
    a.native_answer = True  # exercises the native-answer path in run_arena
    return a


def _run_arena(adapters, judge_primary="claude", secondary=None):
    from arena.config import ArenaConfig, Query
    from arena.pipeline import run_arena
    from arena.scope import INCLUDED, Scope, ScopeEntry
    from _fakes import FakeLLM, sync_gather
    cfg = ArenaConfig(evidence_budget_tokens=600, judge_primary=judge_primary,
                      judge_secondary=secondary)
    scope = Scope(entries=[ScopeEntry(a.name, INCLUDED) for a in adapters])
    q = [Query(query="capital of France?")] * 3
    return run_arena(cfg, q, adapters, scope, FakeLLM(), FakeLLM(), search_gatherer=sync_gather)


def test_caveat_fires_for_claude_native_provider():
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    result = _run_arena([_native_adapter("claude_search", docs), _native_adapter("tavily", docs)])
    j = result["judge"]
    assert j["native_mode"] is True and j["self_preference_caveat"] is True
    assert j["self_preference_flags"] >= 1  # the claude-native pair was flagged


def test_no_caveat_when_only_non_claude_native_provider():
    # A native-answer provider that is NOT Claude-family shares the native path but not the risk.
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    result = _run_arena([_native_adapter("tavily", docs), _native_adapter("brave", docs)])
    j = result["judge"]
    assert j["native_mode"] is False and j["self_preference_caveat"] is False
    assert j["self_preference_flags"] == 0


def test_secondary_judge_config_does_not_suppress_caveat():
    # A configured-but-unwired secondary judge must NOT silence the mitigation.
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    result = _run_arena([_native_adapter("claude_search", docs), _native_adapter("tavily", docs)],
                        secondary="gpt-4o")
    j = result["judge"]
    assert j["self_preference_caveat"] is True and j["self_preference_flags"] >= 1


def test_no_caveat_under_non_claude_judge():
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    result = _run_arena([_native_adapter("claude_search", docs), _native_adapter("tavily", docs)],
                        judge_primary="gpt-4o")
    assert result["judge"]["self_preference_caveat"] is False
