"""Tier-A normalization tests for the M1 roster adapters: You.com, Parallel, Gemini grounding.

No live calls. Each provider asserts: raw -> {answer, results[], latency_ms, cost_units, raw}
through the HandlerAdapter, the empty-evidence guard flags an empty response, and any native
answer is stripped on the default (forced-synthesis) reader path.
"""

import asyncio

from arena.adapters import normalize
from arena.adapters.base import EvidenceDoc, HandlerAdapter, UnifiedResult


def _wrap(search_response, provider_latency=0.5):
    return {"answer": "", "search_response": search_response, "provider_latency": provider_latency}


class _FixedHandler:
    """Async handler returning a preset base-shaped payload (no network)."""

    def __init__(self, payload):
        self._payload = payload

    async def search(self, query):
        return self._payload


# --- You.com -----------------------------------------------------------------

def test_youcom_web_and_news_results():
    raw = _wrap({"results": {
        "web": [{"url": "http://x", "title": "Ti", "description": "De",
                 "snippets": ["frag one", "frag two"], "page_age": "2025-06-25T11:41:00"}],
        "news": [{"url": "http://n", "title": "News", "description": "story"}]}})
    docs = normalize.normalize_youcom(raw)
    assert [d.url for d in docs] == ["http://x", "http://n"]
    assert docs[0].content == "frag one frag two"        # snippets joined
    assert docs[0].published_date == "2025-06-25T11:41:00"
    assert docs[1].content == "story"                    # falls back to description


def test_youcom_empty_when_no_web_or_news():
    assert normalize.normalize_youcom(_wrap({"results": {"web": [], "news": []}})) == []


# --- Parallel ----------------------------------------------------------------

def test_parallel_joins_excerpts():
    raw = _wrap({"results": [
        {"url": "http://x", "title": "Ti", "publish_date": "2024-01-15",
         "excerpts": ["Sample excerpt 1", "Sample excerpt 2"]}]})
    docs = normalize.normalize_parallel(raw)
    assert len(docs) == 1
    assert docs[0].content == "Sample excerpt 1 Sample excerpt 2"
    assert docs[0].published_date == "2024-01-15"


def test_parallel_empty_when_no_results():
    assert normalize.normalize_parallel(_wrap({"results": []})) == []


# --- Gemini grounding --------------------------------------------------------

def test_gemini_grounding_chunks_to_results():
    raw = _wrap({"candidates": [{
        "content": {"parts": [{"text": "answer"}]},
        "groundingMetadata": {
            "groundingChunks": [
                {"web": {"uri": "http://a", "title": "site-a"}},
                {"web": {"uri": "http://b", "title": "site-b"}}],
            "groundingSupports": [
                {"segment": {"text": "seg one"}, "groundingChunkIndices": [0]},
                {"segment": {"text": "seg two"}, "groundingChunkIndices": [0, 1]}]}}]})
    docs = normalize.normalize_gemini(raw)
    assert [d.url for d in docs] == ["http://a", "http://b"]
    assert docs[0].content == "seg one seg two"          # both supports reference chunk 0
    assert docs[1].content == "seg two"


def test_gemini_falls_back_to_title_when_no_support():
    # A chunk referenced by no groundingSupport still becomes evidence via its title.
    raw = _wrap({"candidates": [{
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": "http://a", "title": "site-a"}}],
            "groundingSupports": []}}]})
    docs = normalize.normalize_gemini(raw)
    assert len(docs) == 1 and docs[0].content == "site-a"


def test_gemini_empty_when_no_candidates():
    assert normalize.normalize_gemini(_wrap({"candidates": []})) == []


# --- Adapter-level guarantees (shared shape, answer-stripping, empty guard) ---

_NONEMPTY = {
    "youcom": _wrap({"results": {"web": [
        {"url": "http://x", "title": "Ti", "snippets": ["body"]}], "news": []}}),
    "parallel": _wrap({"results": [
        {"url": "http://x", "title": "Ti", "excerpts": ["body"]}]}),
    "gemini": _wrap({"candidates": [{
        "content": {"parts": [{"text": "native answer"}]},
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": "http://x", "title": "site"}}],
            "groundingSupports": [
                {"segment": {"text": "body"}, "groundingChunkIndices": [0]}]}}]}),
}

_EMPTY = {
    "youcom": _wrap({"results": {"web": [], "news": []}}),
    "parallel": _wrap({"results": []}),
    "gemini": _wrap({"candidates": []}),
}


def test_adapters_map_to_unified_shape_and_strip_answer():
    for name, payload in _NONEMPTY.items():
        adapter = HandlerAdapter(name, _FixedHandler(payload), normalize.NORMALIZERS[name])
        res = asyncio.run(adapter.search("q"))
        assert isinstance(res, UnifiedResult)
        assert res.results and all(isinstance(d, EvidenceDoc) for d in res.results)
        assert all(d.url and d.content for d in res.results)
        assert res.latency_ms == 500.0                   # 0.5s -> 500ms
        assert res.cost_units is None                    # cost mapping is downstream (M1 metrics)
        assert res.raw is payload                         # native payload preserved
        assert res.answer is None                         # native answer stripped (forced synthesis)
        assert res.empty_evidence is False


def test_adapters_flag_empty_evidence():
    for name, payload in _EMPTY.items():
        adapter = HandlerAdapter(name, _FixedHandler(payload), normalize.NORMALIZERS[name])
        res = asyncio.run(adapter.search("q"))
        assert res.results == []
        assert res.empty_evidence is True


def test_roster_none_search_response_is_empty():
    for name in ("youcom", "parallel", "gemini"):
        fn = normalize.NORMALIZERS[name]
        assert fn({"search_response": None, "provider_latency": None}) == []
