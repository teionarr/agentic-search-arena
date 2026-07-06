"""Adapter normalization: per-provider raw payloads -> unified shape (no live calls)."""

import asyncio

from arena.adapters import normalize
from arena.adapters.base import EvidenceDoc, HandlerAdapter, UnifiedResult


def _wrap(search_response):
    return {"answer": "", "search_response": search_response, "provider_latency": 0.5}


def test_tavily_normalizes():
    raw = _wrap({"results": [{"url": "http://x", "title": "T", "content": "body text", "score": 0.9}]})
    docs = normalize.normalize_tavily(raw)
    assert docs == [EvidenceDoc(url="http://x", title="T", content="body text", score=0.9)]


def test_exa_joins_highlights_list():
    raw = _wrap({"results": [{"url": "http://x", "highlights": ["frag one", "frag two"]}]})
    docs = normalize.normalize_exa(raw)
    assert len(docs) == 1 and docs[0].content == "frag one frag two"


def test_exa_empty_when_no_highlights():
    # The real risk: a highlight-less Exa response yields empty evidence (not an error).
    raw = _wrap({"results": [{"url": "http://x"}]})
    assert normalize.normalize_exa(raw) == []


def test_brave_nested_web_results():
    raw = _wrap({"web": {"results": [{"url": "http://x", "title": "Ti", "description": "De"}]}})
    docs = normalize.normalize_brave(raw)
    assert docs[0].content == "Ti\nDe"


def test_serper_organic_link_field():
    raw = _wrap({"organic": [{"link": "http://x", "title": "Ti", "snippet": "Sn"}]})
    docs = normalize.normalize_serper(raw)
    assert docs[0].url == "http://x" and docs[0].content == "Ti\nSn"


def test_perplexity_search_no_answer_key():
    # PerplexitySearchHandler.search() returns no 'answer' key — normalizer must not assume it.
    raw = {"search_response": {"results": [{"url": "http://x", "snippet": "Sn"}]},
           "provider_latency": 0.3}
    docs = normalize.normalize_perplexity_search(raw)
    assert docs[0].content == "Sn"


def test_firecrawl_v2_web_results():
    raw = _wrap({"success": True, "data": {"web": [
        {"url": "http://x", "title": "Ti", "description": "De"}]}, "creditsUsed": 1})
    docs = normalize.normalize_firecrawl(raw)
    assert docs[0].url == "http://x" and docs[0].content == "De"


def test_firecrawl_prefers_markdown_when_present():
    raw = _wrap({"data": {"web": [{"url": "http://x", "markdown": "full content", "description": "snip"}]}})
    assert normalize.normalize_firecrawl(raw)[0].content == "full content"


def test_linkup_search_results():
    raw = _wrap({"results": [
        {"type": "text", "name": "Ti", "url": "http://x", "content": "body"},
        {"type": "image", "name": "img", "url": "http://y"}]})  # image skipped
    docs = normalize.normalize_linkup(raw)
    assert len(docs) == 1 and docs[0].url == "http://x" and docs[0].content == "body"


def test_linkup_image_with_content_is_still_skipped():
    # An image result that DOES carry url+content must still be excluded by the type check —
    # otherwise the generic "if url and content" guard masks a broken image filter.
    raw = _wrap({"results": [
        {"type": "image", "name": "img", "url": "http://y", "content": "caption text"},
        {"type": "text", "name": "T", "url": "http://x", "content": "real"}]})
    docs = normalize.normalize_linkup(raw)
    assert [d.url for d in docs] == ["http://x"]


def test_none_search_response_is_empty():
    for fn in normalize.NORMALIZERS.values():
        assert fn({"search_response": None, "provider_latency": None}) == []


def test_normalizers_on_frozen_real_payloads():
    # Capture-real-then-freeze: the spike dumps one real raw payload per provider; assert each
    # normalizer still extracts usable docs from the real shape (catches provider API drift).
    import glob
    import json
    import os
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    fixtures = glob.glob(os.path.join(fixture_dir, "*_raw.json"))
    if not fixtures:
        import pytest
        pytest.skip("no frozen fixtures present (run `python -m arena.spike` to capture)")
    for path in fixtures:
        name = os.path.basename(path).replace("_raw.json", "")
        if name not in normalize.NORMALIZERS:
            continue
        raw = json.load(open(path))
        docs = normalize.NORMALIZERS[name](raw)
        assert docs, f"{name} normalizer extracted nothing from its real payload"
        assert all(d.url and d.content for d in docs), f"{name} produced a doc missing url/content"


def test_adapter_strips_native_answer_and_converts_latency():
    class FakeHandler:
        async def search(self, q):
            return _wrap({"results": [{"url": "http://x", "content": "c"}]})

    adapter = HandlerAdapter("tavily", FakeHandler(), normalize.normalize_tavily)
    res = asyncio.run(adapter.search("q"))
    assert isinstance(res, UnifiedResult)
    assert res.answer is None                 # native answer discarded (forced synthesis)
    assert res.latency_ms == 500.0            # 0.5s -> 500ms
    assert res.empty_evidence is False


def test_adapter_empty_evidence_flag():
    class EmptyHandler:
        async def search(self, q):
            return _wrap({"results": []})

    adapter = HandlerAdapter("tavily", EmptyHandler(), normalize.normalize_tavily)
    res = asyncio.run(adapter.search("q"))
    assert res.empty_evidence is True
