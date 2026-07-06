"""Cost-per-query column (§8.2, §14 'Metrics math'): pricing_map × units_consumed, normalized
to $/query, with the pricing ``as_of`` date surfaced; blank cost when a provider reports no
units, and its weight dropped + remaining weights renormalized. Deterministic, no AI, no network.
"""

from arena.adapters.base import EvidenceDoc, UnifiedResult
from arena.config import ArenaConfig, Query
from arena.cost import (attach_cost, cost_block, cost_per_query, effective_weights,
                        load_pricing, present_cost_providers)
from arena.metrics import renormalize_weights
from arena.pipeline import run_arena
from arena.report import _cost_as_of, render_cli_summary, write_results
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


PRICING = {"as_of": "2026-07-01",
           "providers": {"tavily": {"unit": "credit", "usd_per_unit": 0.008},
                         "serper": {"unit": "request", "usd_per_unit": 0.0003}}}


# ---- pure cost math ----

def test_cost_per_query_is_price_times_units():
    # tavily: $0.008/credit × 2 credits = $0.016/query.
    assert cost_per_query(PRICING, "tavily", 2.0) == 0.016
    # serper: $0.0003/request × 1 request = $0.0003/query.
    assert abs(cost_per_query(PRICING, "serper", 1.0) - 0.0003) < 1e-12


def test_cost_blank_when_no_units():
    assert cost_per_query(PRICING, "tavily", None) is None


def test_cost_blank_when_provider_unpriced():
    assert cost_per_query(PRICING, "exa", 3.0) is None  # exa not in this pricing map


def test_cost_block_surfaces_as_of_even_when_blank():
    blank = cost_block(PRICING, "tavily", None)
    assert blank["usd_per_query"] is None
    assert blank["as_of"] == "2026-07-01"       # date surfaced regardless of cost presence
    priced = cost_block(PRICING, "serper", 1.0)
    assert priced["usd_per_query"] and priced["as_of"] == "2026-07-01"
    assert priced["unit"] == "request"


# ---- shipped pricing file ----

def test_load_shipped_pricing_has_as_of_and_all_providers():
    p = load_pricing("configs/pricing.yaml")
    assert p["as_of"], "pricing file must record an as_of date (§8.2)"
    for prov in ("tavily", "exa", "brave", "serper", "perplexity_search", "firecrawl", "linkup"):
        assert prov in p["providers"], f"{prov} missing a default price"
        assert p["providers"][prov]["usd_per_unit"] > 0


def test_load_pricing_missing_file_is_blank_not_error():
    p = load_pricing("configs/does_not_exist.yaml")
    assert p == {"as_of": None, "providers": {}}


def test_load_pricing_never_aborts_on_bad_file(tmp_path):
    # Pricing is optional: invalid YAML, a non-mapping root, and a non-mapping `providers`
    # must all leave cost blank (empty shape) rather than raising and killing the run.
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("as_of: [unclosed\n  : broken")           # invalid YAML
    non_mapping_root = tmp_path / "list.yaml"
    non_mapping_root.write_text("- just\n- a\n- list\n")           # root is a list
    bad_providers = tmp_path / "badprov.yaml"
    bad_providers.write_text("as_of: \"2026-07-01\"\nproviders: 42\n")  # providers not a mapping
    for path in (bad_yaml, non_mapping_root, bad_providers):
        p = load_pricing(str(path))
        assert p == {"as_of": None, "providers": {}}
        assert "as_of" in p and isinstance(p["providers"], dict)


# ---- weight drop + renormalize when cost absent/partial (§8) ----

def test_cost_weight_dropped_and_renormalized_when_absent():
    metrics = {"tavily": {"cost": {"usd_per_query": None}},
               "serper": {"cost": {"usd_per_query": None}}}
    present = present_cost_providers(metrics)              # nobody has a cost
    weights = {"cost": 0.2, "latency": 0.4, "coverage": 0.4}
    metric_axes = ["latency", "coverage"] + (["cost"] if present else [])
    renorm = renormalize_weights(weights, metric_axes)
    assert "cost" not in renorm                            # dropped
    assert abs(sum(renorm.values()) - 1.0) < 1e-9          # remainder renormalized
    assert abs(renorm["latency"] - 0.5) < 1e-9             # 0.4 / 0.8


def test_cost_weight_kept_when_present():
    metrics = {"tavily": {"cost": {"usd_per_query": 0.016}}}
    assert present_cost_providers(metrics) == ["tavily"]
    weights = {"cost": 0.2, "latency": 0.8}
    renorm = renormalize_weights(weights, ["cost", "latency"])
    assert "cost" in renorm and abs(sum(renorm.values()) - 1.0) < 1e-9


def test_effective_weights_drops_cost_when_blank():
    # latency + coverage are populated; only cost is blank → cost drops, the rest renormalize.
    metrics = {"tavily": {"cost": {"usd_per_query": None},
                          "latency": {"p50": 100.0},
                          "coverage": {"avg_tokens_per_result": 50.0}}}
    ew = effective_weights({"cost": 0.2, "latency": 0.4, "coverage": 0.4}, metrics)
    assert "cost" not in ew
    assert abs(sum(ew.values()) - 1.0) < 1e-9 and abs(ew["latency"] - 0.5) < 1e-9


def test_effective_weights_keeps_cost_when_present():
    metrics = {"tavily": {"cost": {"usd_per_query": 0.016}, "latency": {"p50": 100.0}}}
    ew = effective_weights({"cost": 0.2, "latency": 0.8}, metrics)
    assert "cost" in ew and abs(sum(ew.values()) - 1.0) < 1e-9


def test_effective_weights_unified_cost_and_freshness_drop_together():
    # The coordinator's interaction: cost AND freshness both absent → both weights drop and
    # latency/coverage renormalize through the one shared path (no competing renormalization).
    metrics = {"tavily": {"cost": {"usd_per_query": None},
                          "latency": {"p50": 100.0},
                          "coverage": {"avg_tokens_per_result": 50.0}}}  # no freshness block at all
    ew = effective_weights({"cost": 0.25, "freshness": 0.25, "latency": 0.25, "coverage": 0.25},
                           metrics)
    assert "cost" not in ew and "freshness" not in ew
    assert abs(sum(ew.values()) - 1.0) < 1e-9
    assert abs(ew["latency"] - 0.5) < 1e-9 and abs(ew["coverage"] - 0.5) < 1e-9


# ---- pipeline: units → $/query normalization + attach ----

class _CostAdapter:
    """Adapter-like fake that reports ``cost_units`` per search (drives the cost column)."""

    def __init__(self, name, docs, cost_units):
        self.name = name
        self._docs = docs
        self._units = cost_units

    async def search(self, query):
        from arena.adapters.base import EvidenceDoc  # local import: fixtures live per-test
        docs = [EvidenceDoc(**d) for d in self._docs]
        return UnifiedResult(results=docs, answer=None, latency_ms=100.0,
                             cost_units=self._units, empty_evidence=len(docs) == 0)


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(adapters, pricing_path, n_queries=2):
    cfg = ArenaConfig(evidence_budget_tokens=600, pricing_path=pricing_path)
    queries = [Query(query=f"capital of France? {i}") for i in range(n_queries)]
    scope = _scope([a.name for a in adapters])
    return run_arena(cfg, queries, adapters, scope, FakeLLM(), FakeLLM(),
                     search_gatherer=sync_gather)


def _write_pricing(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text("as_of: \"2026-07-01\"\n"
                 "providers:\n"
                 "  tavily: { unit: credit, usd_per_unit: 0.008 }\n"
                 "  serper: { unit: request, usd_per_unit: 0.0003 }\n")
    return str(p)


def test_pipeline_normalizes_units_to_cost_per_query(tmp_path):
    doc = [{"url": "u", "title": "t", "content": "Paris is the capital of France."}]
    # tavily reports 2 credits per query (constant across queries) → $/query = 0.008 × 2 = 0.016.
    tavily = _CostAdapter("tavily", doc, cost_units=2.0)
    serper = _CostAdapter("serper", doc, cost_units=1.0)
    result = _run([tavily, serper], _write_pricing(tmp_path))
    ct = result["metrics"]["tavily"]["cost"]
    cs = result["metrics"]["serper"]["cost"]
    assert ct["usd_per_query"] == 0.016            # normalized to $/query, not summed over queries
    assert abs(cs["usd_per_query"] - 0.0003) < 1e-12
    assert ct["as_of"] == "2026-07-01"             # as_of surfaced with the cost column


def test_pipeline_drops_cost_weight_when_no_units_reported(tmp_path):
    # A run where no adapter reports units (FakeAdapter reports none — like a real provider
    # whose billing is token-based/unknowable, §8.2) leaves cost blank for every provider →
    # the pipeline must drop the cost weight from the effective weights and renormalize the
    # remainder (the wired §8 renormalization path).
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    cfg = ArenaConfig(evidence_budget_tokens=600, pricing_path=_write_pricing(tmp_path),
                      weights={"cost": 0.2, "latency": 0.4, "coverage": 0.4})
    queries = [Query(query="capital of France?")]
    scope = _scope(["tavily", "serper"])
    result = run_arena(cfg, queries, [FakeAdapter("tavily", docs), FakeAdapter("serper", docs)],
                       scope, FakeLLM(), FakeLLM(), search_gatherer=sync_gather)
    ew = result["weights_effective"]
    assert "cost" not in ew                                  # dropped: no provider reported units
    assert abs(sum(ew.values()) - 1.0) < 1e-9               # remainder renormalized
    assert abs(ew["latency"] - 0.5) < 1e-9 and abs(ew["coverage"] - 0.5) < 1e-9


def test_pipeline_keeps_cost_weight_when_units_reported(tmp_path):
    doc = [{"url": "u", "title": "t", "content": "Paris is the capital of France."}]
    cfg = ArenaConfig(evidence_budget_tokens=600, pricing_path=_write_pricing(tmp_path),
                      weights={"cost": 0.2, "latency": 0.8})
    queries = [Query(query="capital of France?")]
    scope = _scope(["tavily"])
    result = run_arena(cfg, queries, [_CostAdapter("tavily", doc, cost_units=2.0)],
                       scope, FakeLLM(), FakeLLM(), search_gatherer=sync_gather)
    ew = result["weights_effective"]
    assert "cost" in ew and abs(sum(ew.values()) - 1.0) < 1e-9  # cost survives, weights sum to 1


def test_pipeline_blank_cost_when_no_units(tmp_path):
    doc = [{"url": "u", "title": "t", "content": "Paris is the capital of France."}]
    tavily = _CostAdapter("tavily", doc, cost_units=2.0)
    serper = _CostAdapter("serper", doc, cost_units=None)   # reports no units → blank cost
    result = _run([tavily, serper], _write_pricing(tmp_path))
    assert result["metrics"]["serper"]["cost"]["usd_per_query"] is None
    assert result["metrics"]["tavily"]["cost"]["usd_per_query"] == 0.016
    # partial-cost run: cost weight must drop, remaining renormalize.
    present = present_cost_providers(result["metrics"])
    assert present == ["tavily"]


# ---- report surfaces the cost column + as_of ----

def _doc_with_cost():
    return {
        "timestamp": "t", "model_id": "m", "n_queries": 2, "cost_usd": 0.1,
        "degenerate_run": False, "n_decided_comparisons": 2, "n_excluded_comparisons": 0,
        "judge": {"swap_consistency": 0.9},
        "scope": {"tavily": {"status": "included", "detail": ""}},
        "stage_status": {"judge": {"status": "green", "reason": "ok"}},
        "ranking": [{"provider": "tavily", "win_rate": 0.6, "ci_low": 0.5, "ci_high": 0.7,
                     "n_comparisons": 2, "status": "ranked", "rank": 1, "tie_group": 0},
                    {"provider": "serper", "win_rate": 0.4, "ci_low": 0.3, "ci_high": 0.5,
                     "n_comparisons": 2, "status": "ranked", "rank": 2, "tie_group": 1}],
        "metrics": {
            "tavily": {"coverage": {"avg_tokens_per_result": 100},
                       "cost": {"usd_per_query": 0.016, "unit": "credit",
                                "units_consumed": 2.0, "as_of": "2026-07-01"}},
            "serper": {"coverage": {"avg_tokens_per_result": 90},
                       "cost": {"usd_per_query": None, "unit": "request",
                                "units_consumed": None, "as_of": "2026-07-01"}}},
    }


def test_cli_surfaces_cost_and_as_of():
    text = render_cli_summary(_doc_with_cost())
    assert "pricing as of 2026-07-01" in text     # as_of printed alongside the cost column
    assert "$0.0160/q" in text                     # tavily's computed cost
    assert "cost n/a" in text                       # serper blank cost, not fabricated


def test_cost_as_of_helper_reads_from_metrics():
    assert _cost_as_of(_doc_with_cost()) == "2026-07-01"
    assert _cost_as_of({"metrics": {}}) is None


def test_csv_includes_cost_columns(tmp_path):
    paths = write_results(_doc_with_cost(), str(tmp_path))
    rows = open(paths["csv"]).read().splitlines()
    assert "cost_usd_per_query" in rows[0] and "cost_as_of" in rows[0]
    assert "0.016" in rows[1] and "2026-07-01" in rows[1]        # tavily row
