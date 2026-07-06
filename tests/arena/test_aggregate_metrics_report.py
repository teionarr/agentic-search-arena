"""Aggregation (win-rate + bootstrap CI), metrics math, and report safety."""

from arena.aggregate import aggregate
from arena.metrics import evidence_coverage, latency_percentiles, renormalize_weights
from arena.report import _csv_safe, query_set_hash, redact


# ---- aggregate ----

def _round_robin(winner_map, providers, repeats):
    """Build comparisons where winner_map[(a,b)] wins each of `repeats` times."""
    comps = []
    for (a, b), winner in winner_map.items():
        for _ in range(repeats):
            comps.append({"a": a, "b": b, "winner": winner})
    return comps


def test_aggregate_deterministic():
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}] * 5 + [{"a": "p1", "b": "p2", "winner": "p2"}] * 2
    a1 = aggregate(comps, ["p1", "p2"], seed=0)
    a2 = aggregate(comps, ["p1", "p2"], seed=0)
    assert [(_s.provider, _s.win_rate, _s.ci_low, _s.ci_high) for _s in a1.scores] == \
           [(_s.provider, _s.win_rate, _s.ci_low, _s.ci_high) for _s in a2.scores]


def test_aggregate_separation_70_30():
    # p1 beats p2 70/30 at significant n -> non-overlapping CIs (must not manufacture a tie).
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}] * 70 + [{"a": "p1", "b": "p2", "winner": "p2"}] * 30
    agg = aggregate(comps, ["p1", "p2"], seed=0)
    s = {x.provider: x for x in agg.scores}
    assert s["p1"].win_rate > s["p2"].win_rate
    assert s["p1"].ci_low > s["p2"].ci_high  # non-overlapping -> not tied
    assert agg.tie_groups == [["p1"], ["p2"]]


def test_aggregate_ties_when_equal():
    comps = [{"a": "p1", "b": "p2", "winner": "tie"}] * 30
    agg = aggregate(comps, ["p1", "p2"], seed=0)
    assert len(agg.tie_groups) == 1 and set(agg.tie_groups[0]) == {"p1", "p2"}


def test_aggregate_unranked_insufficient():
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}]  # p3 never compared
    agg = aggregate(comps, ["p1", "p2", "p3"], seed=0, min_comparisons=2)
    s = {x.provider: x for x in agg.scores}
    assert s["p3"].status == "unranked"


def test_aggregate_min_comparisons_boundary():
    # Exactly min_comparisons -> ranked; one fewer -> unranked (guards the < vs <= off-by-one).
    comps = ([{"a": "p1", "b": "p2", "winner": "p1"}, {"a": "p1", "b": "p2", "winner": "p2"}]
             + [{"a": "p3", "b": "p1", "winner": "p1"}])  # p2 has exactly 2, p3 has exactly 1
    agg = aggregate(comps, ["p1", "p2", "p3"], seed=0, min_comparisons=2)
    s = {x.provider: x for x in agg.scores}
    assert s["p2"].n_comparisons == 2 and s["p2"].status == "ranked"   # boundary is inclusive
    assert s["p3"].n_comparisons == 1 and s["p3"].status == "unranked"


def test_aggregate_ci_is_a_real_interval():
    # A non-degenerate provider must get an actual interval (ci_low < win_rate < ci_high),
    # not a point estimate collapsed onto the win-rate.
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}] * 60 + [{"a": "p1", "b": "p2", "winner": "p2"}] * 40
    agg = aggregate(comps, ["p1", "p2"], seed=0)
    s = {x.provider: x for x in agg.scores}
    assert s["p1"].ci_high > s["p1"].ci_low                    # non-zero width
    assert s["p1"].ci_low < s["p1"].win_rate < s["p1"].ci_high


def test_redact_drops_key_fields_by_name_even_without_value_match():
    # Key-bearing fields must be dropped by NAME, independent of the value-scrub path.
    out = redact({"api_key": "not-a-known-secret", "authorization": "Bearer zzz",
                  "x-api-key": "abc", "note": "harmless"}, secret_values=[])
    assert out["api_key"] == "«redacted»"
    assert out["authorization"] == "«redacted»"
    assert out["x-api-key"] == "«redacted»"
    assert out["note"] == "harmless"


def test_aggregate_excludes_none_winner():
    comps = [{"a": "p1", "b": "p2", "winner": None}] * 5 + [{"a": "p1", "b": "p2", "winner": "p1"}] * 3
    agg = aggregate(comps, ["p1", "p2"], seed=0)
    assert agg.n_excluded == 5 and agg.n_decided == 3


# ---- metrics ----

def test_latency_excludes_none():
    out = latency_percentiles([100.0, None, 200.0, None])
    assert out["n"] == 2 and out["p50"] == 150.0


def test_latency_all_none():
    assert latency_percentiles([None, None]) == {"p50": None, "p95": None, "n": 0}


def test_coverage():
    assert evidence_coverage([10, 20, 30])["avg_tokens_per_result"] == 20.0
    assert evidence_coverage([])["avg_tokens_per_result"] is None


def test_renormalize_weights_drops_absent():
    w = renormalize_weights({"accuracy": 0.5, "latency": 0.25, "coverage": 0.25},
                            present_metrics=["latency", "coverage"])
    assert abs(sum(w.values()) - 1.0) < 1e-9 and "accuracy" not in w


# ---- report safety ----

def test_redact_scrubs_secret_value_and_key_fields():
    doc = {"raw": {"api_key": "SEKRET", "note": "value SEKRET appears"}, "x": ["SEKRET"]}
    out = redact(doc, secret_values=["SEKRET"])
    assert out["raw"]["api_key"] == "«redacted»"
    assert "SEKRET" not in out["raw"]["note"]
    assert out["x"] == ["«redacted»"]


def test_csv_safe_prefixes_formula():
    assert _csv_safe("=cmd()") == "'=cmd()"
    assert _csv_safe("+1") == "'+1"
    assert _csv_safe("normal") == "normal"


def test_query_set_hash_stable():
    assert query_set_hash(["a", "b"]) == query_set_hash(["a", "b"])
    assert query_set_hash(["a", "b"]) != query_set_hash(["b", "a"])


def _sample_doc():
    return {
        "timestamp": "t", "model_id": "m", "n_queries": 3, "cost_usd": 0.1,
        "degenerate_run": False, "n_decided_comparisons": 2, "n_excluded_comparisons": 1,
        "judge": {"swap_consistency": 0.71},
        "scope": {"tavily": {"status": "included", "detail": ""},
                  "firecrawl": {"status": "excluded — user choice", "detail": ""}},
        "stage_status": {"judge": {"status": "red", "reason": "swap-consistency 0.71"}},
        "ranking": [{"provider": "tavily", "win_rate": 0.6, "ci_low": 0.5, "ci_high": 0.7,
                     "n_comparisons": 2, "status": "ranked", "rank": 1, "tie_group": 0},
                    {"provider": "exa", "win_rate": None, "ci_low": None, "ci_high": None,
                     "n_comparisons": 1, "status": "unranked", "rank": None, "tie_group": None}],
        "metrics": {"tavily": {"coverage": {"avg_tokens_per_result": 100}}, "exa": {"coverage": {}}},
    }


def test_write_results_emits_json_and_csv_only(tmp_path):
    from arena.report import write_results
    paths = write_results(_sample_doc(), str(tmp_path))
    assert set(paths) == {"json", "csv"}          # no HTML artifact
    import os
    assert not any(f.endswith(".html") for f in os.listdir(tmp_path))


def test_cli_dashboard_renders_bars_and_status():
    from arena.report import render_cli_summary
    text = render_cli_summary(_sample_doc())
    assert "SEARCH ARENA" in text
    assert "█" in text and "│" in text                       # win-rate bar + 0.50 marker
    assert "clear leader" in text                            # sole-member tie group tagged
    assert "unranked — insufficient valid comparisons" in text
    assert "❌ judge" in text and "swap-consistency 0.71" in text
