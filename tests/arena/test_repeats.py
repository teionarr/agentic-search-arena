"""--repeats N (statistical honesty): real re-runs per query + per-repeat variance signal."""

import pytest

from arena.adapters.base import EvidenceDoc
from arena.aggregate import point_winrates
from arena.config import ArenaConfig, Query
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(repeats, n_queries=2):
    docs = [EvidenceDoc(url="u", title="t", content="evidence content about the topic here")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    cfg = ArenaConfig(evidence_budget_tokens=600, repeats=repeats)
    queries = [Query(query=f"q{i}?") for i in range(n_queries)]
    return run_arena(cfg, queries, adapters, _scope([a.name for a in adapters]),
                     FakeLLM(), FakeLLM(), search_gatherer=sync_gather)


# ---- pure helper ----

def test_point_winrates():
    comps = ([{"a": "p1", "b": "p2", "winner": "p1"}] * 3
             + [{"a": "p1", "b": "p2", "winner": "tie"}])
    wr = point_winrates(comps, ["p1", "p2", "p3"])
    assert wr["p1"] == pytest.approx(3.5 / 4)
    assert wr["p2"] == pytest.approx(0.5 / 4)
    assert wr["p3"] is None  # never compared -> None, not 0


# ---- config ----

def test_repeats_default_and_validation():
    assert ArenaConfig().repeats == 1
    with pytest.raises(ValueError):
        ArenaConfig(repeats=0)


def test_repeats_from_config_file(tmp_path):
    from arena.config import load_config
    p = tmp_path / "arena.yaml"
    p.write_text("repeats: 3\n")
    assert load_config(str(p)).repeats == 3


# ---- pipeline ----

def test_repeats_multiply_comparisons_but_not_n_queries():
    r1 = _run(repeats=1)
    r3 = _run(repeats=3)
    assert r1["n_queries"] == r3["n_queries"] == 2  # unique queries, not expanded count
    assert r3["n_decided_comparisons"] == 3 * r1["n_decided_comparisons"]


def test_repeats_block_present_with_variance():
    r = _run(repeats=2)
    blk = r["repeats"]
    assert blk["n"] == 2
    assert set(blk["per_repeat_win_rates"]) == {"0", "1"}
    # Deterministic fakes -> identical repeats -> zero spread.
    assert blk["win_rate_spread"] == {"tavily": 0.0, "brave": 0.0}


def test_repeats_one_has_no_variance_keys():
    r = _run(repeats=1)
    assert r["repeats"] == {"n": 1}


def test_cli_shows_repeats_and_spread():
    from arena.report import build_document, render_cli_summary
    r = _run(repeats=2)
    doc = build_document(r, ["q0?", "q1?"], config_snapshot={}, model_id="test-model")
    text = render_cli_summary(doc)
    assert "× 2 repeats" in text
    assert "win-rate spread across repeats" in text


def test_cli_no_spread_line_single_run():
    from arena.report import build_document, render_cli_summary
    r = _run(repeats=1)
    doc = build_document(r, ["q0?", "q1?"], config_snapshot={}, model_id="test-model")
    text = render_cli_summary(doc)
    assert "repeats" not in text
