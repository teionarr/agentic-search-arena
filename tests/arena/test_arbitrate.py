"""Tier-2 arbitration (M4): pivotal-tie selection, blinding, re-aggregation. No IO, no AI."""

import json

from arena.arbitrate import (arbitrate_interactively, blind_order, comparisons_from_rationale,
                             load_run, reapply, render_rankings, select_pivotal)


def _doc():
    """Two statistically-tied providers (p1,p2) + a clear third (p3); one excluded and one
    low-confidence verdict between the tied pair, plus decided ones."""
    rationale = (
        [{"query": f"q{i}", "a": "p1", "b": "p2", "winner": "p1",
          "flipped": False, "low_confidence": False} for i in range(3)]
        + [{"query": f"q{i}", "a": "p1", "b": "p2", "winner": "p2",
            "flipped": False, "low_confidence": False} for i in range(3, 6)]
        + [{"query": "q_flip", "a": "p1", "b": "p2", "winner": None,
            "flipped": True, "low_confidence": True}]
        + [{"query": "q_low", "a": "p1", "b": "p2", "winner": "p1",
            "flipped": False, "low_confidence": True}]
        + [{"query": f"q{i}", "a": "p3", "b": "p1", "winner": "p3",
            "flipped": False, "low_confidence": False} for i in range(6, 12)]
    )
    return {
        "ranking": [
            {"provider": "p3", "rank": 1, "win_rate": 0.9, "status": "ranked", "tie_group": 0},
            {"provider": "p1", "rank": 2, "win_rate": 0.5, "status": "ranked", "tie_group": 1},
            {"provider": "p2", "rank": 3, "win_rate": 0.5, "status": "ranked", "tie_group": 1},
        ],
        "tie_groups": [["p3"], ["p1", "p2"]],
        "rationale_log": rationale,
        "aggregation_method": "bradley_terry",
    }


# ---- selection ----

def test_selects_only_tied_pair_excluded_or_low_confidence():
    items, meta = select_pivotal(_doc())
    assert {i["query"] for i in items} == {"q_flip", "q_low"}   # not the decided q0..q5
    assert all({i["a"], i["b"]} == {"p1", "p2"} for i in items)  # p3 is not tied with anyone
    assert meta["n_left_out"] == 0


def test_cap_reports_left_out():
    items, meta = select_pivotal(_doc(), max_items=1)
    assert len(items) == 1 and meta["n_left_out"] == 1           # no silent truncation


def test_no_ties_selects_nothing():
    doc = _doc()
    doc["tie_groups"] = [["p3"], ["p1"], ["p2"]]                 # everyone separated
    items, _ = select_pivotal(doc)
    assert items == []


# ---- blinding ----

def test_blind_order_deterministic_and_covers_both_orders():
    assert blind_order("q", "a", "b") == blind_order("q", "a", "b")
    orders = {blind_order(f"q{i}", "a", "b") for i in range(40)}
    assert orders == {("a", "b"), ("b", "a")}                    # not always the same side


# ---- re-aggregation ----

def test_human_verdicts_break_the_tie():
    doc = _doc()
    base = comparisons_from_rationale(doc["rationale_log"])
    providers = ["p3", "p1", "p2"]
    from arena.aggregate import aggregate
    before = aggregate(base, providers, seed=0)
    assert {s.provider for s in before.scores if s.tie_group is not None}  # sanity
    # Human arbitrates the two pivotal items for p1, at high weight.
    adjudications = [{"a": "p1", "b": "p2", "winner": "p1"},
                     {"a": "p1", "b": "p2", "winner": "p1"}]
    after = reapply(base, adjudications, providers, weight=5.0)
    s = {x.provider: x for x in after.scores}
    assert s["p1"].win_rate > s["p2"].win_rate                   # tie broken in p1's favor


def test_skipped_verdicts_contribute_nothing():
    base = [{"a": "p1", "b": "p2", "winner": "p1"}] * 4
    same = reapply(base, [{"a": "p1", "b": "p2", "winner": None}], ["p1", "p2"])
    baseline = reapply(base, [], ["p1", "p2"])
    assert [(s.provider, s.win_rate) for s in same.scores] == \
           [(s.provider, s.win_rate) for s in baseline.scores]


def test_render_shows_before_after():
    base = ([{"a": "p1", "b": "p2", "winner": "p1"}] * 2
            + [{"a": "p1", "b": "p2", "winner": "p2"}] * 2)
    from arena.aggregate import aggregate
    before = aggregate(base, ["p1", "p2"], seed=0)
    after = reapply(base, [{"a": "p1", "b": "p2", "winner": "p1"}] * 3, ["p1", "p2"])
    text = render_rankings(before, after)
    assert "ARBITRATION" in text and "p1" in text and "→" in text


# ---- interactive loop (scripted) ----

def test_interactive_loop_records_blinded_verdicts():
    items = [{"query": "q_flip", "a": "p1", "b": "p2"},
             {"query": "q_low", "a": "p1", "b": "p2"}]
    answers = {"q_flip": {"p1": "answer one", "p2": "answer two"},
               "q_low": {"p1": "answer one", "p2": "answer two"}}
    script = iter(["x", "1", "t"])                               # invalid retried, then 1, tie
    out = []
    adjs = arbitrate_interactively(items, answers, ask=lambda _: next(script),
                                   echo=out.append)
    assert len(adjs) == 2
    assert adjs[0]["winner"] == adjs[0]["shown_first"]           # '1' maps through the blinding
    assert adjs[1]["winner"] == "tie"
    assert any("enter 1, 2, t, s or q" in str(line) for line in out)


def test_interactive_quit_stops_early():
    items = [{"query": "q1", "a": "p1", "b": "p2"}, {"query": "q2", "a": "p1", "b": "p2"}]
    adjs = arbitrate_interactively(items, {}, ask=lambda _: "q", echo=lambda *_: None)
    assert adjs == []


# ---- run loading ----

def test_load_run_requires_traces(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_doc()))
    import pytest
    with pytest.raises(FileNotFoundError, match="save-traces"):
        load_run(str(tmp_path))


def test_load_run_reads_answers_from_traces(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_doc()))
    tdir = tmp_path / "traces"
    tdir.mkdir()
    (tdir / "query_0000.json").write_text(json.dumps(
        {"query": "q_flip", "providers": {"p1": {"reader_answer": "A1"},
                                          "p2": {"reader_answer": "A2"}}}))
    doc, answers = load_run(str(tmp_path))
    assert doc["tie_groups"] == [["p3"], ["p1", "p2"]]
    assert answers["q_flip"] == {"p1": "A1", "p2": "A2"}
