"""Parallel request compatibility through the real registry/handler; no network or keys."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from arena.adapters import parallel_handler
from arena.adapters.registry import REGISTRY
from arena.config import load_config


@pytest.fixture
def transport(monkeypatch):
    """Capture actual handler request arguments and return the existing canned response."""
    fixture = Path(__file__).parent / "fixtures" / "parallel_raw.json"
    state = {"calls": [], "status": 200, "error": None,
             "payload": json.loads(fixture.read_text())["search_response"]}
    monkeypatch.setenv("PARALLEL_API_KEY", "fixture-key")

    class Response:
        @property
        def status(self):
            return state["status"]

        async def __aenter__(self):
            if state["error"] is not None:
                raise state["error"]
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return state["payload"]

    class Session:
        def __init__(self, *, timeout):
            state["timeout"] = timeout.total

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, *, json, headers):
            state["calls"].append({"url": url, "body": copy.deepcopy(json), "headers": headers})
            return Response()

    monkeypatch.setattr(parallel_handler.aiohttp, "ClientSession", Session)
    return state


def _adapter(config):
    spec = REGISTRY["parallel"]
    return spec.build("parallel", {**spec.default_config, **config})


@pytest.mark.parametrize("config,limit,units", [
    ({"max_results": 5}, 5, 1.0),
    ({"advanced_settings": {"max_results": 30}}, 30, None),
    ({"max_results": 30, "advanced_settings": {"max_results": 5}}, 5, 1.0),
    ({"max_results": 5, "advanced_settings": {"max_results": 30}}, 30, None),
    ({"max_results": 30, "advanced_settings": {"max_results": None}}, None, 1.0),
    ({"max_results": 5, "advanced_settings": None}, 5, 1.0),
    ({"max_results": None}, None, 1.0),
])
def test_result_limits_reach_api_and_cost_guard(transport, tmp_path, config, limit, units):
    # Exercise the same file loader -> default/override merge -> registry build as the CLI.
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"providers": {"parallel": {"config": config}}}))
    loaded = load_config(str(path)).providers["parallel"]["config"]
    original = copy.deepcopy(loaded)
    adapter = _adapter(loaded)
    result = asyncio.run(adapter.search("2010 IEEE Frank Rosenblatt Award recipient"))
    body = transport["calls"][0]["body"]
    assert "max_results" not in body
    assert body["advanced_settings"]["max_results"] == limit
    assert body["mode"] == "advanced"
    assert loaded == original                         # no mutation of saved config
    assert result.cost_units == units
    assert len(result.results) == 2
    assert "Michio Sugeno" in result.results[0].content
    assert result.results[0].published_date == "2024-12-02"
    assert result.latency_ms is not None
    assert result.answer is None and result.needs_synthesis
    assert not result.empty_evidence


def test_default_request_keeps_api_defaults_and_headers(transport):
    asyncio.run(_adapter({}).search("query"))
    call = transport["calls"][0]
    assert call["url"] == "https://api.parallel.ai/v1/search"
    assert call["body"] == {"objective": "query", "search_queries": ["query"], "mode": "advanced"}
    assert call["headers"]["x-api-key"] == "fixture-key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["User-Agent"] == (
        "agentic-search-arena (https://github.com/teionarr/agentic-search-arena)")
    assert transport["timeout"] == 30


def test_legacy_limit_preserves_other_explicit_settings(transport):
    config = {"max_results": 5, "mode": "fast", "objective": "specific objective",
              "search_queries": ["specific keywords"],
              "advanced_settings": {"source_policy": {"include_domains": ["example.com"]}}}
    original = copy.deepcopy(config)
    asyncio.run(_adapter(config).search("query"))
    body = transport["calls"][0]["body"]
    assert body == {"mode": "fast", "objective": "specific objective",
                    "search_queries": ["specific keywords"],
                    "advanced_settings": {"max_results": 5,
                                          "source_policy": {"include_domains": ["example.com"]}}}
    assert config == original


@pytest.mark.parametrize("status,error", [
    (429, None),
    (500, None),
    (200, asyncio.TimeoutError()),
    (200, OSError("connection closed")),
])
def test_errors_keep_empty_evidence_sentinel(transport, status, error):
    transport.update(status=status, error=error)
    result = asyncio.run(_adapter({"max_results": 5}).search("query"))
    assert result.empty_evidence and result.results == []
    assert result.cost_units is None and result.latency_ms is None
    assert result.raw == {"answer": "", "search_response": None, "provider_latency": None}


def test_cancellation_propagates(transport):
    transport["error"] = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_adapter({"max_results": 5}).search("query"))
