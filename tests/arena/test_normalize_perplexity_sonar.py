"""Perplexity Sonar adapter (native-answer path, the second after claude_search) — Tier-A.

No live calls. Asserts: the handler's answer extraction, the normalizer's
search_results/citations mapping, the adapter's native-answer preservation over a hand-frozen
fixture, and the registry entry (native, NOT Claude-family — the native path without the
self-preference caveat, proving that path isn't Claude-shaped).
"""

import asyncio
import json
import os

from arena.adapters import normalize
from arena.adapters.base import EvidenceDoc, HandlerAdapter, UnifiedResult
from arena.adapters.perplexity_sonar_handler import _extract_answer
from arena.adapters.registry import (REGISTRY, claude_family_providers,
                                     native_answer_providers)


def _fixture():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "perplexity_sonar_raw.json")
    return json.load(open(path))


class _FixedHandler:
    """Async handler returning a preset base-shaped payload (no network)."""

    def __init__(self, payload):
        self._payload = payload

    async def search(self, query):
        return self._payload


# --- normalizer: search_results / citations -> EvidenceDoc[] ------------------

def test_normalize_perplexity_maps_search_results():
    raw = {"search_response": {"search_results": [
        {"url": "http://x", "title": "Ti", "snippet": "Sn", "date": "2025-06-01"}]},
        "provider_latency": 0.2}
    docs = normalize.normalize_perplexity(raw)
    assert docs == [EvidenceDoc(url="http://x", title="Ti", content="Sn",
                                published_date="2025-06-01")]


def test_normalize_perplexity_snippetless_falls_back_to_title():
    raw = {"search_response": {"search_results": [
        {"url": "http://x", "title": "Ti"},        # no snippet -> title
        {"title": "no url"}]}}                      # no url -> dropped
    docs = normalize.normalize_perplexity(raw)
    assert len(docs) == 1 and docs[0].content == "Ti"


def test_normalize_perplexity_citations_only_fallback():
    # A citations-only response carries bare URLs; keep them as evidence (content = URL).
    raw = {"search_response": {"citations": ["http://a", "http://b"]}}
    docs = normalize.normalize_perplexity(raw)
    assert [d.url for d in docs] == ["http://a", "http://b"]
    assert all(d.content == d.url for d in docs)


def test_normalize_perplexity_registered_and_none_safe():
    assert "perplexity" in normalize.NORMALIZERS
    assert normalize.normalize_perplexity({"search_response": None}) == []


# --- handler answer extraction: chat completion choices -> answer text --------

def test_extract_answer_from_choices():
    fx = _fixture()
    assert _extract_answer(fx["search_response"]) == fx["answer"]


def test_extract_answer_empty_when_no_choices():
    assert _extract_answer({}) == ""


# --- adapter over the frozen fixture: unified shape, native answer preserved --

def test_sonar_adapter_maps_fixture_to_unified_shape():
    fx = _fixture()
    adapter = HandlerAdapter("perplexity", _FixedHandler(fx),
                             normalize.NORMALIZERS["perplexity"], native_answer=True)
    res = asyncio.run(adapter.search("q"))
    assert isinstance(res, UnifiedResult)
    assert res.answer == fx["answer"]                # native answer preserved
    assert res.needs_synthesis is False              # native-answer path
    assert res.latency_ms == fx["provider_latency"] * 1000.0
    assert res.cost_units is None                    # cost mapping is downstream (M1 metrics)
    assert res.raw is fx                             # native payload preserved
    assert res.empty_evidence is False
    assert [d.url for d in res.results] == [
        r["url"] for r in fx["search_response"]["search_results"]]
    assert all(isinstance(d, EvidenceDoc) and d.url and d.content for d in res.results)


def test_sonar_adapter_flags_empty_evidence_on_error_sentinel():
    payload = {"answer": "", "search_response": None, "provider_latency": None}
    adapter = HandlerAdapter("perplexity", _FixedHandler(payload),
                             normalize.NORMALIZERS["perplexity"], native_answer=True)
    res = asyncio.run(adapter.search("q"))
    assert res.results == [] and res.empty_evidence is True
    assert res.answer is None                        # empty answer coerced to None


# --- registry: native-answer entry, NOT Claude-family (no caveat) -------------

def test_registry_marks_perplexity_native_not_claude_family():
    spec = REGISTRY["perplexity"]
    assert spec.native_answer is True
    assert spec.required_env_keys == ["PERPLEXITY_API_KEY"]
    assert spec.family is None                       # family exists only for the Claude caveat
    assert "perplexity" in native_answer_providers()
    assert "perplexity" not in claude_family_providers()
