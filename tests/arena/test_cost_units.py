"""Per-call billing units (§8.2): the registry declares each provider's deterministic unit
count (constant or config-dependent), the adapter emits it on success and never on empty/error,
and every unit-reporting provider has a matching pricing.yaml entry. No live calls, no AI.
"""

import asyncio

from arena.adapters.base import HandlerAdapter, UnifiedResult
from arena.adapters.registry import REGISTRY, resolve_cost_units
from arena.cost import load_pricing


# ---- adapter emission: spec units on success, None on empty/error ----

class _FixedHandler:
    def __init__(self, payload=None, raises=False):
        self._payload = payload
        self._raises = raises

    async def search(self, query):
        if self._raises:
            raise RuntimeError("boom")
        return self._payload


_DOC_PAYLOAD = {"answer": "", "provider_latency": 0.5,
                "search_response": {"results": [
                    {"url": "http://x", "title": "t", "content": "body"}]}}
_EMPTY_PAYLOAD = {"answer": "", "provider_latency": 0.5, "search_response": {"results": []}}


def _tavily_adapter(payload=None, raises=False):
    from arena.adapters import normalize
    return HandlerAdapter("tavily", _FixedHandler(payload, raises), normalize.NORMALIZERS["tavily"])


def test_adapter_emits_spec_units_on_success():
    adapter = _tavily_adapter(_DOC_PAYLOAD)
    adapter.cost_units_per_call = 2.0                    # as build() sets from the spec
    res = asyncio.run(adapter.search("q"))
    assert isinstance(res, UnifiedResult) and res.results
    assert res.cost_units == 2.0


def test_adapter_emits_no_units_on_empty_evidence():
    adapter = _tavily_adapter(_EMPTY_PAYLOAD)
    adapter.cost_units_per_call = 2.0
    res = asyncio.run(adapter.search("q"))
    assert res.empty_evidence and res.cost_units is None


def test_adapter_emits_no_units_on_error():
    adapter = _tavily_adapter(raises=True)
    adapter.cost_units_per_call = 2.0
    res = asyncio.run(adapter.search("q"))
    assert res.empty_evidence and res.cost_units is None


def test_adapter_defaults_to_no_units():
    # An adapter built without a spec (bare composition) keeps the honest blank (§8.2).
    res = asyncio.run(_tavily_adapter(_DOC_PAYLOAD).search("q"))
    assert res.results and res.cost_units is None


# ---- config-dependent unit resolution (registry) ----

def _units(name, **overrides):
    spec = REGISTRY[name]
    return resolve_cost_units(spec, {**spec.default_config, **overrides})


def test_tavily_units_depend_on_search_depth():
    assert _units("tavily") == 2.0                            # default config: advanced
    assert _units("tavily", search_depth="basic") == 1.0
    assert _units("tavily", search_depth="unknown") is None   # unpublished depth → blank


def test_linkup_units_depend_on_depth_and_output_type():
    assert _units("linkup") == 1.0                            # standard + searchResults
    assert _units("linkup", depth="deep") == 10.0             # $0.05 = 10× the $0.005 unit
    assert _units("linkup", outputType="sourcedAnswer") == 1.2
    assert _units("linkup", outputType="unknown") is None


def test_firecrawl_units_flat_only_without_scraping():
    assert _units("firecrawl") == 2.0                         # 2 credits per <=10-result search
    assert _units("firecrawl", limit=25) is None              # rounded-up on returned results
    assert _units("firecrawl", scrapeOptions={"formats": ["markdown"]}) is None


def test_serper_and_parallel_units_gate_on_result_count():
    assert _units("serper") == 1.0
    assert _units("serper", num=100) is None                  # credit count unpublished above 10
    assert _units("parallel") == 1.0
    assert _units("parallel", max_results=30) is None         # extra results bill per result


def test_flat_per_request_providers():
    assert _units("brave") == 1.0
    assert _units("perplexity_search") == 1.0


def test_token_based_or_unpublished_providers_report_no_units():
    # Honesty rule (§8.2): token-based (perplexity Sonar, gemini, claude_search), per-page
    # variable (exa) and unpublished (youcom) billing must stay blank — never fabricated.
    for name in ("perplexity", "gemini", "claude_search", "exa", "youcom"):
        assert _units(name) is None, f"{name} must not report fabricated units"


# ---- pricing consistency: every unit-reporting provider is priced ----

def test_every_unit_reporting_provider_has_a_pricing_entry():
    pricing = load_pricing("configs/pricing.yaml")
    assert pricing["as_of"], "pricing file must record an as_of date (§8.2)"
    for name, spec in REGISTRY.items():
        if resolve_cost_units(spec, dict(spec.default_config)) is not None:
            entry = pricing["providers"].get(name)
            assert entry and entry.get("usd_per_unit", 0) > 0, \
                f"{name} reports cost units but has no usable pricing.yaml entry"
