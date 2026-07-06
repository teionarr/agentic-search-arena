"""Cost per successful outcome: $/query ÷ accuracy rate, only where anchors exist (§8)."""

import pytest

from arena.cost import attach_cost_per_success


def _m(usd_per_query=None, acc_rate=None, acc_total=0, with_cost=True):
    m = {"accuracy": {"rate": acc_rate, "total": acc_total,
                      "correct": int((acc_rate or 0) * acc_total)}}
    if with_cost:
        m["cost"] = {"usd_per_query": usd_per_query, "as_of": "2026-07-01"}
    return m


def test_derived_where_cost_and_anchor_exist():
    metrics = {"tavily": _m(usd_per_query=0.01, acc_rate=0.5, acc_total=10)}
    attach_cost_per_success(metrics)
    # Half the answers are correct -> each correct answer effectively costs double.
    assert metrics["tavily"]["cost"]["usd_per_correct"] == pytest.approx(0.02)


def test_blank_without_accuracy_anchor():
    metrics = {"tavily": _m(usd_per_query=0.01, acc_rate=None)}
    attach_cost_per_success(metrics)
    assert metrics["tavily"]["cost"]["usd_per_correct"] is None


def test_blank_without_cost():
    metrics = {"tavily": _m(usd_per_query=None, acc_rate=0.8, acc_total=5)}
    attach_cost_per_success(metrics)
    assert metrics["tavily"]["cost"]["usd_per_correct"] is None


def test_blank_when_never_correct():
    # rate == 0 -> no finite $/correct; blank, not infinity and not fabricated.
    metrics = {"tavily": _m(usd_per_query=0.01, acc_rate=0.0, acc_total=5)}
    attach_cost_per_success(metrics)
    assert metrics["tavily"]["cost"]["usd_per_correct"] is None


def test_missing_cost_block_tolerated():
    metrics = {"tavily": _m(with_cost=False)}
    attach_cost_per_success(metrics)  # must not raise
    assert "cost" not in metrics["tavily"]


def test_csv_and_cli_render_usd_per_correct(tmp_path):
    from arena.report import render_cli_summary, write_results
    doc = {
        "timestamp": "t", "model_id": "m", "n_queries": 1, "cost_usd": None,
        "degenerate_run": False, "n_decided_comparisons": 2, "n_excluded_comparisons": 0,
        "judge": {}, "scope": {}, "stage_status": {},
        "ranking": [{"provider": "tavily", "win_rate": 0.6, "ci_low": 0.5, "ci_high": 0.7,
                     "n_comparisons": 2, "status": "ranked", "rank": 1, "tie_group": 0}],
        "metrics": {"tavily": {"coverage": {"avg_tokens_per_result": 100},
                               "cost": {"usd_per_query": 0.01, "usd_per_correct": 0.02,
                                        "as_of": "2026-07-01"}}},
    }
    paths = write_results(doc, str(tmp_path))
    import csv as csvmod
    with open(paths["csv"]) as f:
        row = next(csvmod.DictReader(f))
    assert row["cost_usd_per_correct"] == "0.02"
    assert "$0.0200/correct" in render_cli_summary(doc)
