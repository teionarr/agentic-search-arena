"""Per-category ranking (§8 use-case segmentation): same aggregation, per category slice."""

from arena.adapters.base import EvidenceDoc
from arena.aggregate import per_category_rankings
from arena.config import ArenaConfig, Query
from arena.pipeline import run_arena
from arena.report import render_cli_summary
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


# ---- pure slicing (aggregate) ----

def test_per_category_opposite_rankings():
    # p1 dominates in 'news', p2 dominates in 'code' -> the slices must rank them oppositely.
    comps = ([{"a": "p1", "b": "p2", "winner": "p1", "category": "news"}] * 10
             + [{"a": "p1", "b": "p2", "winner": "p2", "category": "code"}] * 10)
    out = per_category_rankings(comps, ["p1", "p2"], seed=0)
    news = {s.provider: s for s in out["news"].scores}
    code = {s.provider: s for s in out["code"].scores}
    assert news["p1"].rank == 1 and news["p2"].rank == 2
    assert code["p2"].rank == 1 and code["p1"].rank == 2


def test_per_category_untagged_comparisons_excluded_from_slices():
    comps = ([{"a": "p1", "b": "p2", "winner": "p1", "category": "news"}] * 5
             + [{"a": "p1", "b": "p2", "winner": "p2", "category": None}] * 5)
    out = per_category_rankings(comps, ["p1", "p2"], seed=0)
    assert set(out) == {"news"}
    assert out["news"].n_decided == 5  # untagged rows contribute only to the overall ranking


def test_per_category_small_slice_unranked():
    # A slice below the min-comparisons floor yields unranked, never a forced order.
    comps = [{"a": "p1", "b": "p2", "winner": "p1", "category": "rare"}]
    out = per_category_rankings(comps, ["p1", "p2"], seed=0)
    assert all(s.status == "unranked" for s in out["rare"].scores)


def test_per_category_empty_when_no_categories():
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}] * 3
    assert per_category_rankings(comps, ["p1", "p2"], seed=0) == {}


# ---- end-to-end through the pipeline ----

def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run_with_categories():
    docs = [EvidenceDoc(url="u", title="t", content="evidence content about the topic here")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    queries = ([Query(query=f"news q{i}?", category="news") for i in range(3)]
               + [Query(query=f"code q{i}?", category="code") for i in range(3)]
               + [Query(query="untagged q?")])
    cfg = ArenaConfig(evidence_budget_tokens=600)
    return run_arena(cfg, queries, adapters, _scope([a.name for a in adapters]),
                     FakeLLM(), FakeLLM(), search_gatherer=sync_gather)


def test_pipeline_emits_per_category_block():
    result = _run_with_categories()
    assert set(result["per_category"]) == {"news", "code"}
    for blk in result["per_category"].values():
        assert blk["n_decided_comparisons"] == 3  # one comparison per query in the slice
        providers = {s["provider"] for s in blk["ranking"]}
        assert providers == {"tavily", "brave"}
    # Overall ranking still spans all 7 queries' comparisons.
    assert result["n_decided_comparisons"] == 7


def test_pipeline_no_categories_gives_empty_block():
    docs = [EvidenceDoc(url="u", title="t", content="evidence content about the topic here")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    cfg = ArenaConfig(evidence_budget_tokens=600)
    result = run_arena(cfg, [Query(query="q?")], adapters, _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(), search_gatherer=sync_gather)
    assert result["per_category"] == {}


def test_cli_renders_by_category_section():
    from arena.report import build_document
    result = _run_with_categories()
    doc = build_document(result, ["q"], config_snapshot={}, model_id="test-model")
    text = render_cli_summary(doc)
    assert "BY CATEGORY" in text
    assert "news" in text and "code" in text


def test_cli_omits_by_category_when_absent():
    doc = {
        "timestamp": "t", "model_id": "m", "n_queries": 1, "cost_usd": None,
        "degenerate_run": False, "n_decided_comparisons": 1, "n_excluded_comparisons": 0,
        "judge": {}, "scope": {}, "stage_status": {},
        "ranking": [], "metrics": {}, "per_category": {},
    }
    assert "BY CATEGORY" not in render_cli_summary(doc)
