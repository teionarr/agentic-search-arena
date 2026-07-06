"""Bradley-Terry aggregation (§6.3) + secondary judge / reliability weighting (§5, §6.3, §6.4).

Tier-A: deterministic, no AI. Judge is mocked via FakeLLM.
"""

from arena.aggregate import aggregate
from arena.reliability import (cohens_kappa, consensus_agreements, judge_weights)


# ---------------------------------------------------------------------------
# Bradley-Terry aggregation
# ---------------------------------------------------------------------------

def _chain(winner_map, repeats):
    comps = []
    for (a, b), winner in winner_map.items():
        for _ in range(repeats):
            comps.append({"a": a, "b": b, "winner": winner})
    return comps


def test_bt_deterministic():
    # Identical fixture -> identical ranking + CIs across runs (seeded).
    comps = _chain({("p1", "p2"): "p1", ("p2", "p3"): "p2", ("p1", "p3"): "p1"}, 8)
    a1 = aggregate(comps, ["p1", "p2", "p3"], seed=0, method="bradley_terry")
    a2 = aggregate(comps, ["p1", "p2", "p3"], seed=0, method="bradley_terry")
    key = lambda agg: [(s.provider, s.win_rate, s.ci_low, s.ci_high, s.rank, s.tie_group)
                       for s in agg.scores]
    assert key(a1) == key(a2)
    assert a1.method == "bradley_terry"


def test_bt_default_method_is_bradley_terry():
    comps = [{"a": "p1", "b": "p2", "winner": "p1"}] * 4 + [{"a": "p1", "b": "p2", "winner": "p2"}] * 2
    assert aggregate(comps, ["p1", "p2"]).method == "bradley_terry"


def test_bt_transitive_ranking():
    # A>B, B>C, A>C consistently -> strict order A, B, C by BT strength.
    comps = _chain({("A", "B"): "A", ("B", "C"): "B", ("A", "C"): "A"}, 12)
    agg = aggregate(comps, ["A", "B", "C"], seed=0, method="bradley_terry")
    ranked = [s.provider for s in agg.scores if s.status == "ranked"]
    assert ranked == ["A", "B", "C"]


def test_bt_separation_70_30_non_overlapping():
    # A beats B 70/30 at significant n -> non-overlapping CIs, NOT a manufactured tie.
    comps = [{"a": "A", "b": "B", "winner": "A"}] * 70 + [{"a": "A", "b": "B", "winner": "B"}] * 30
    agg = aggregate(comps, ["A", "B"], seed=0, method="bradley_terry")
    s = {x.provider: x for x in agg.scores}
    assert s["A"].win_rate > s["B"].win_rate
    assert s["A"].ci_low > s["B"].ci_high              # non-overlapping
    assert agg.tie_groups == [["A"], ["B"]]


def test_bt_ci_widens_as_n_shrinks():
    def width(n):
        comps = ([{"a": "A", "b": "B", "winner": "A"}] * int(0.6 * n)
                 + [{"a": "A", "b": "B", "winner": "B"}] * (n - int(0.6 * n)))
        agg = aggregate(comps, ["A", "B"], seed=0, method="bradley_terry")
        a = {x.provider: x for x in agg.scores}["A"]
        return a.ci_high - a.ci_low
    assert width(200) < width(40) < width(12)          # monotone: fewer games -> wider CI


def test_bt_overlapping_ci_tie_grouping():
    # Near-even outcomes -> overlapping CIs -> a single tie group (not falsely ordered).
    comps = [{"a": "A", "b": "B", "winner": "A"}] * 11 + [{"a": "A", "b": "B", "winner": "B"}] * 10
    agg = aggregate(comps, ["A", "B"], seed=0, method="bradley_terry")
    assert len(agg.tie_groups) == 1 and set(agg.tie_groups[0]) == {"A", "B"}


def test_bt_unranked_insufficient_comparisons():
    comps = [{"a": "A", "b": "B", "winner": "A"}] * 4  # C never compared
    agg = aggregate(comps, ["A", "B", "C"], seed=0, method="bradley_terry", min_comparisons=2)
    s = {x.provider: x for x in agg.scores}
    assert s["C"].status == "unranked" and s["C"].win_rate is None
    assert s["A"].status == "ranked" and s["B"].status == "ranked"


def test_bt_excludes_none_winner():
    comps = [{"a": "A", "b": "B", "winner": None}] * 5 + [{"a": "A", "b": "B", "winner": "A"}] * 3
    agg = aggregate(comps, ["A", "B"], seed=0, method="bradley_terry")
    assert agg.n_excluded == 5 and agg.n_decided == 3


def test_bt_default_run_is_unweighted():
    # No 'weight' key on any comparison => plain unweighted BT (the default single-judge path).
    # Assert byte-identical output to explicitly weight=1.0 on every comparison.
    comps = _chain({("A", "B"): "A", ("B", "C"): "B", ("A", "C"): "A"}, 6)
    plain = aggregate(comps, ["A", "B", "C"], seed=0, method="bradley_terry")
    w1 = aggregate([{**c, "weight": 1.0} for c in comps], ["A", "B", "C"], seed=0,
                   method="bradley_terry")
    key = lambda agg: [(s.provider, s.win_rate, s.ci_low, s.ci_high) for s in agg.scores]
    assert key(plain) == key(w1)


def test_winrate_method_still_available():
    # The M0 estimator is kept intact and selectable.
    comps = [{"a": "A", "b": "B", "winner": "A"}] * 6 + [{"a": "A", "b": "B", "winner": "B"}] * 4
    agg = aggregate(comps, ["A", "B"], seed=0, method="winrate")
    s = {x.provider: x for x in agg.scores}
    assert agg.method == "winrate"
    assert s["A"].win_rate == 0.6 and s["B"].win_rate == 0.4   # raw win-rate, exact


def test_aggregate_rejects_unknown_method():
    import pytest
    with pytest.raises(ValueError):
        aggregate([{"a": "A", "b": "B", "winner": "A"}], ["A", "B"], method="elo")


def test_aggregate_rejects_invalid_weights():
    # nan/inf/zero/negative weights would corrupt weighted wins/games/CIs — reject at the boundary.
    import pytest
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            aggregate([{"a": "A", "b": "B", "winner": "A", "weight": bad}], ["A", "B"])


# ---------------------------------------------------------------------------
# Cohen's kappa (§6.4)
# ---------------------------------------------------------------------------

def test_kappa_perfect_agreement():
    a = ["x", "y", "tie", "x", "y"]
    assert cohens_kappa(a, list(a)) == 1.0


def test_kappa_known_value():
    # Classic 2x2 fixture: 20 both-yes, 15 both-no, 5 + 10 disagree (n=50).
    # po = 35/50 = 0.70; pe = (25/50)(30/50)+(25/50)(20/50) = 0.30+0.20 = 0.50
    # kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert abs(cohens_kappa(a, b) - 0.40) < 1e-9


def test_kappa_chance_level_is_zero():
    # Independent-at-chance raters -> kappa ~ 0. Exactly 0 on a balanced product fixture.
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert abs(cohens_kappa(a, b)) < 1e-9


def test_kappa_empty_is_none():
    assert cohens_kappa([], []) is None


# ---------------------------------------------------------------------------
# Judge-reliability weighting (§6.3)
# ---------------------------------------------------------------------------

def test_no_weighting_single_judge():
    # A single judge (no per-judge signal) => empty weights => plain unweighted BT.
    assert judge_weights({"primary": 0.9}) == {}
    assert judge_weights({}) == {}


def test_weighting_off_mode():
    assert judge_weights({"primary": 0.9, "secondary": 0.5}, mode="off") == {}


def test_unreliable_judge_downweighted():
    # Secondary agrees with the ensemble less than primary => strictly smaller weight.
    w = judge_weights({"primary": 0.9, "secondary": 0.5})
    assert w["secondary"] < w["primary"]
    assert abs((w["primary"] + w["secondary"]) / 2 - 1.0) < 1e-9   # mean-normalized


def test_consensus_agreements_direction():
    # Primary matches the majority every time; secondary dissents on half the items.
    per_judge = {
        "primary":   ["A", "A", "B", "B"],
        "secondary": ["A", "tie", "B", "tie"],
        "third":     ["A", "A", "B", "B"],
    }
    agree = consensus_agreements(per_judge)
    assert agree["primary"] == 1.0 and agree["third"] == 1.0
    assert agree["secondary"] == 0.5
    w = judge_weights(agree)
    assert w["secondary"] < w["primary"]


def test_downweighting_shifts_bt_toward_reliable_verdicts():
    # Crafted fixture: a reliable judge says A>B; an unreliable judge says B>A. With the
    # unreliable judge down-weighted, BT must place A above B (the reliable verdict wins).
    reliable = [{"a": "A", "b": "B", "winner": "A", "weight": 2.0}] * 10
    unreliable = [{"a": "A", "b": "B", "winner": "B", "weight": 0.2}] * 10
    agg = aggregate(reliable + unreliable, ["A", "B"], seed=0, method="bradley_terry")
    s = {x.provider: x for x in agg.scores}
    assert s["A"].win_rate > s["B"].win_rate

    # Sanity: unweighted (equal votes) is a dead tie by BT expected-score.
    equal = ([{"a": "A", "b": "B", "winner": "A"}] * 10
             + [{"a": "A", "b": "B", "winner": "B"}] * 10)
    s2 = {x.provider: x for x in aggregate(equal, ["A", "B"], seed=0).scores}
    assert abs(s2["A"].win_rate - s2["B"].win_rate) < 1e-6


def test_two_judge_consensus_is_symmetric_no_lexicographic_bias():
    # With only two judges, every disagreement is a 1-1 tie. consensus_agreements must NOT let a
    # judge look more reliable for picking "x" over "y" (i.e. by provider-name order): both judges
    # get the SAME symmetric pairwise-agreement rate regardless of which label each chose.
    per_judge = {"primary": ["x", "x", "tie"], "secondary": ["y", "x", "y"]}  # agree on 1 of 3
    agree = consensus_agreements(per_judge)
    assert agree["primary"] == agree["secondary"] == 1 / 3
    # Swapping which side each judge took must not change the (symmetric) result.
    swapped = {"primary": ["y", "x", "y"], "secondary": ["x", "x", "tie"]}
    assert consensus_agreements(swapped)["primary"] == 1 / 3


def test_two_judge_run_stays_unweighted():
    # Equal agreements (the two-judge case) carry no signal about WHICH judge is right -> no
    # weighting. This keeps a pure two-judge, no-gold run as plain unweighted BT (report κ only).
    assert judge_weights({"primary": 0.7, "secondary": 0.7}) == {}
    # And end-to-end via the consensus helper:
    per_judge = {"primary": ["x", "y", "tie"], "secondary": ["y", "x", "tie"]}
    assert judge_weights(consensus_agreements(per_judge)) == {}


def test_three_judge_consensus_still_discriminates():
    # 3 judges DO form a real majority, so an outlier judge is down-weighted (the signal exists).
    per_judge = {"primary": ["x", "x", "x"], "secondary": ["x", "x", "x"], "third": ["y", "y", "y"]}
    agree = consensus_agreements(per_judge)
    assert agree["primary"] == 1.0 and agree["third"] == 0.0
    w = judge_weights(agree)
    assert w["third"] < w["primary"]


def test_consensus_agreements_rejects_misaligned_lengths():
    import pytest
    with pytest.raises(ValueError):
        consensus_agreements({"primary": ["x", "y"], "secondary": ["x"]})


# ---------------------------------------------------------------------------
# BT connected-components (§6.3 identifiability)
# ---------------------------------------------------------------------------

def test_bt_disconnected_components_normalized_independently():
    # Two disjoint sub-tournaments that never share an opponent: {A>B} and {C>D}. Strength is only
    # identifiable within a component, so the two winners (A, C) must land at the SAME strength and
    # be grouped as a tie — not forced into an arbitrary cross-component order by a shared scale.
    comps = [{"a": "A", "b": "B", "winner": "A"}] * 10 + [{"a": "C", "b": "D", "winner": "C"}] * 10
    agg = aggregate(comps, ["A", "B", "C", "D"], seed=0, method="bradley_terry")
    s = {x.provider: x for x in agg.scores}
    assert abs(s["A"].win_rate - s["C"].win_rate) < 1e-6   # winners tie across components
    assert abs(s["B"].win_rate - s["D"].win_rate) < 1e-6   # losers tie across components
    assert {frozenset(g) for g in agg.tie_groups} == {frozenset(["A", "C"]), frozenset(["B", "D"])}


def test_bt_connected_transitive_unaffected():
    # The common fully-connected path must be unchanged: A>B>C stays strictly ordered.
    comps = _chain({("A", "B"): "A", ("B", "C"): "B", ("A", "C"): "A"}, 10)
    agg = aggregate(comps, ["A", "B", "C"], seed=0, method="bradley_terry")
    assert [s.provider for s in agg.scores if s.status == "ranked"] == ["A", "B", "C"]
