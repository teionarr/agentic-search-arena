"""Secondary-judge wiring through the pipeline (§5, §6.3, §6.4) — mocked judges, no AI."""

from arena.config import ArenaConfig, Query
from arena.judge import PairwiseVerdict, judge_pair, route_native_self_preference
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry
from arena.adapters.base import EvidenceDoc
from _fakes import FakeAdapter, FakeLLM, sync_gather


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _verdict(winner):
    return lambda system, user, schema: PairwiseVerdict(winner=winner, rationale="r")


def _docs():
    return [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]


def test_single_judge_no_secondary_is_unweighted():
    # Default path: no secondary judge => no per-judge signal => BT stays unweighted, κ is None.
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600)
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(), search_gatherer=sync_gather)
    assert result["reliability_weighted"] is False
    assert result["judge"]["inter_judge_kappa"] is None
    assert result["aggregation_method"] == "bradley_terry"


def test_secondary_judge_surfaces_kappa():
    # Both judges always tie => perfect agreement => κ == 1.0 surfaced in judge block + stage.
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600)
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(structured_fn=_verdict("tie")),
                       search_gatherer=sync_gather,
                       secondary_judge_llm=FakeLLM(structured_fn=_verdict("tie")))
    assert result["judge"]["inter_judge_kappa"] == 1.0
    assert "κ" in result["stage_status"]["judge"]["reason"]


def test_secondary_judge_disagreement_lowers_kappa():
    # Primary always picks A, secondary always picks B => the two judges never agree.
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600, order_swap=False)  # no swap-exclusion noise
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(structured_fn=_verdict("A")),
                       search_gatherer=sync_gather,
                       secondary_judge_llm=FakeLLM(structured_fn=_verdict("B")))
    assert result["judge"]["inter_judge_kappa"] is not None
    assert result["judge"]["inter_judge_kappa"] < 0.5


def test_secondary_judge_two_judge_run_is_unweighted():
    # A pure two-judge run has no signal about WHICH judge is right (cross-agreement is symmetric)
    # => BT stays unweighted even though κ is reported (§6.3 spec-correct default).
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600, order_swap=False)
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(structured_fn=_verdict("A")),
                       search_gatherer=sync_gather,
                       secondary_judge_llm=FakeLLM(structured_fn=_verdict("B")))
    assert result["reliability_weighted"] is False              # unweighted
    assert result["judge"]["inter_judge_kappa"] is not None     # but κ still reported


def test_secondary_judge_failure_surfaced_in_stage_status():
    # A configured secondary judge that always skips (returns None) produces NO paired labels.
    # This must NOT stay silently green off primary-only swap consistency — the judge stage goes
    # red and names the broken secondary path (CodeRabbit pipeline.py:306).
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600)
    dead_secondary = FakeLLM(structured_fn=lambda system, user, schema: None)
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(structured_fn=_verdict("tie")),
                       search_gatherer=sync_gather, secondary_judge_llm=dead_secondary)
    assert result["judge"]["inter_judge_kappa"] is None
    assert result["stage_status"]["judge"]["status"] == "red"
    assert "secondary" in result["stage_status"]["judge"]["reason"]


def test_single_judge_no_false_secondary_failure():
    # No secondary configured => no secondary-failure flag; stage stays healthy on a good judge.
    a = FakeAdapter("tavily", _docs())
    b = FakeAdapter("brave", _docs())
    cfg = ArenaConfig(evidence_budget_tokens=600)
    result = run_arena(cfg, [Query(query="q?")] * 3, [a, b], _scope(["tavily", "brave"]),
                       FakeLLM(), FakeLLM(structured_fn=_verdict("tie")), search_gatherer=sync_gather)
    assert "secondary" not in result["stage_status"]["judge"]["reason"]


def test_self_preference_routing_hook_noop_when_not_native():
    # In M0/M1 no pair is native, so the hook never routes even with a secondary configured.
    assert route_native_self_preference(False, False, secondary_configured=True) is False
    assert route_native_self_preference(True, False, secondary_configured=True) is True    # future path
    assert route_native_self_preference(True, True, secondary_configured=False) is False   # no secondary


def test_routing_gated_to_claude_family_natives():
    # The pipeline gates the route on ``native AND in claude_family`` before calling the hook.
    # A non-Claude native (native=True but not Claude-family) must NOT route: caller passes False.
    claude_family = {"claude_search"}

    def route_for(prov, native):
        gated = native and prov in claude_family
        return route_native_self_preference(gated, False, secondary_configured=True)

    assert route_for("claude_search", native=True) is True     # Claude native -> routes
    assert route_for("perplexity", native=True) is False       # non-Claude native -> no route
    assert route_for("claude_search", native=False) is False   # synthesized -> no route


def test_judge_pair_records_both_judge_labels():
    x = {"answer": "ax", "docs": _docs()}
    y = {"answer": "ay", "docs": _docs()}
    out = judge_pair(FakeLLM(structured_fn=_verdict("A")), "q?", x, y, "nonce",
                     order_swap=False, secondary_llm=FakeLLM(structured_fn=_verdict("B")))
    assert out["judge_labels"]["primary"] == "x"
    assert out["judge_labels"]["secondary"] == "y"
    assert out["decided_by"] == "primary"   # not routed


def test_judge_pair_routes_to_secondary_when_requested():
    x = {"answer": "ax", "docs": _docs()}
    y = {"answer": "ay", "docs": _docs()}
    out = judge_pair(FakeLLM(structured_fn=_verdict("A")), "q?", x, y, "nonce",
                     order_swap=False, secondary_llm=FakeLLM(structured_fn=_verdict("B")),
                     route_to_secondary=True)
    assert out["outcome"] == "y"            # secondary's verdict is the reported outcome
    assert out["decided_by"] == "secondary"
