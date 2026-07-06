"""Tier-1 free anchors (§3 Tier 1): consensus silver labels + machine-verifiable checks.

All deterministic — no AI, no network. Consensus forms only at >= N convergence; normalization
canonicalizes case/numbers/dates; machine-verify matchers are exact; arena-vs-consensus
agreement is computed; the anchor column stays blank where unanchored.
"""

import pytest

from arena.adapters.base import EvidenceDoc
from arena.anchors import (
    compute_anchors,
    consensus_label,
    machine_verify,
    normalize_answer,
    verify_date,
    verify_number,
    verify_string_contains,
)
from arena.config import ArenaConfig, Query
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


# ---- normalization ----

def test_normalize_case_trim_punctuation():
    assert normalize_answer("  Paris. ") == normalize_answer("paris") == "paris"
    assert normalize_answer("New York!") == normalize_answer("new york") == "new york"


def test_normalize_canonicalizes_numbers():
    # thousands separators and a trailing .0 collapse to the same token
    assert normalize_answer("1,000") == normalize_answer("1000") == normalize_answer("1000.0")
    assert "1000" in normalize_answer("about 1,000 people")


def test_normalize_canonicalizes_dates():
    # equivalent date spellings normalize to the same ISO string
    base = normalize_answer("2020-01-05")
    assert normalize_answer("January 5, 2020") == base
    assert normalize_answer("Jan 5 2020") == base
    assert normalize_answer("5 January 2020") == base


# ---- consensus (>= N convergence only) ----

def test_consensus_forms_at_n():
    answers = {"a": "Paris", "b": "paris.", "c": "PARIS"}
    assert consensus_label(answers, min_providers=3) == "paris"


def test_consensus_absent_at_n_minus_one():
    # Exactly N-1 converge -> no silver label (never fabricate at the boundary).
    answers = {"a": "Paris", "b": "paris", "c": "Lyon"}
    assert consensus_label(answers, min_providers=3) is None


def test_consensus_ignores_empty_answers():
    answers = {"a": "Paris", "b": "paris", "c": None, "d": ""}
    assert consensus_label(answers, min_providers=3) is None
    assert consensus_label(answers, min_providers=2) == "paris"


def test_consensus_ambiguous_tie_yields_no_label():
    # A 3-vs-3 split: both answers reach min_providers=3, so there is NO unique convergence.
    # A false consensus here would corrupt silver labels -> must return None.
    answers = {"a": "Paris", "b": "paris", "c": "PARIS", "d": "Lyon", "e": "lyon", "f": "LYON"}
    assert consensus_label(answers, min_providers=3) is None


# ---- machine-verify matchers ----

def test_verify_string_contains():
    assert verify_string_contains("The capital is Paris.", "Paris") is True
    assert verify_string_contains("The capital is Lyon.", "Paris") is False


def test_verify_number_exact():
    assert verify_number("There are 1,000 items.", "1000") is True
    assert verify_number("There are 999 items.", "1000") is False
    assert verify_number("some text", "not-a-number") is None  # not checkable as a number


def test_verify_date_exact():
    assert verify_date("It happened on January 5, 2020.", "2020-01-05") is True
    assert verify_date("It happened on 2019-01-05.", "2020-01-05") is False
    assert verify_date("some text", "not-a-date") is None


def test_machine_verify_dispatch_and_blank():
    assert machine_verify("The capital is Paris.", "Paris") is True   # string
    assert machine_verify("Total is 42.", "42") is True               # number
    assert machine_verify("Dated 2020-01-05.", "Jan 5 2020") is True  # date
    assert machine_verify(None, "Paris") is None                      # unanchored -> blank
    assert machine_verify("answer", None) is None


# ---- compute_anchors: coverage + arena-vs-consensus agreement ----

def test_compute_anchors_coverage_and_agreement():
    # q0: 3 providers converge on "paris" (consensus); the arena winner also matches consensus.
    per_query_answers = [{"a": "Paris", "b": "paris", "c": "PARIS", "d": "Lyon"}]
    per_query_comparisons = [[
        {"a": "a", "b": "d", "winner": "a"},   # consensus side (a) wins -> agrees
        {"a": "b", "b": "d", "winner": "b"},   # consensus side (b) wins -> agrees
    ]]
    anc = compute_anchors(per_query_answers, per_query_comparisons,
                          ["a", "b", "c", "d"], min_providers=3).as_dict()
    assert anc["n_consensus_queries"] == 1 and anc["consensus_coverage"] == 1.0
    assert anc["arena_vs_consensus_agreement"] == 1.0 and anc["n_arena_checked"] == 1


def test_compute_anchors_no_consensus_blank_agreement():
    per_query_answers = [{"a": "Paris", "b": "Lyon", "c": "Rome"}]  # no convergence
    anc = compute_anchors(per_query_answers, [[]], ["a", "b", "c"], min_providers=3).as_dict()
    assert anc["n_consensus_queries"] == 0 and anc["consensus_coverage"] == 0.0
    assert anc["arena_vs_consensus_agreement"] is None and anc["n_arena_checked"] == 0


def test_compute_anchors_disagreement():
    # 3 converge on "paris"; the arena winner picks the NON-consensus side -> disagreement.
    per_query_answers = [{"a": "Paris", "b": "paris", "c": "PARIS", "d": "Lyon"}]
    per_query_comparisons = [[{"a": "a", "b": "d", "winner": "d"}]]  # non-consensus side wins
    anc = compute_anchors(per_query_answers, per_query_comparisons,
                          ["a", "b", "c", "d"], min_providers=3).as_dict()
    assert anc["arena_vs_consensus_agreement"] == 0.0 and anc["n_arena_checked"] == 1


# ---- e2e: machine-verify is a FREE accuracy anchor (no OpenAI, no LLM grader) ----

def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def test_machine_verify_feeds_accuracy_without_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    a, b = FakeAdapter("tavily", docs), FakeAdapter("brave", docs)
    reader = FakeLLM(complete_fn=lambda s, u: "The capital is Paris.")
    # No grader_llm passed: the ONLY way accuracy populates is the deterministic machine check.
    result = run_arena(ArenaConfig(evidence_budget_tokens=600),
                       [Query(query="capital of France?", expected_answer="Paris")],
                       [a, b], _scope(["tavily", "brave"]), reader, FakeLLM(),
                       search_gatherer=sync_gather)
    for prov in ("tavily", "brave"):
        acc = result["metrics"][prov]["accuracy"]
        assert acc["total"] == 1 and acc["correct"] == 1 and acc["rate"] == 1.0
        av = result["anchors"]["auto_verify"][prov]
        assert av["total"] == 1 and av["correct"] == 1  # counted as a free anchor


def test_anchor_column_blank_when_unanchored():
    # No expected_answer and (with only 2 providers) no consensus at the default min=3.
    docs = [EvidenceDoc(url="u", title="t", content="evidence content for the query")]
    result = run_arena(ArenaConfig(evidence_budget_tokens=600),
                       [Query(query="q?")], [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)],
                       _scope(["tavily", "brave"]), FakeLLM(), FakeLLM(), search_gatherer=sync_gather)
    anc = result["anchors"]
    assert anc["consensus_coverage"] == 0.0            # no consensus reached
    assert anc["arena_vs_consensus_agreement"] is None  # nothing to check against
    for prov in ("tavily", "brave"):
        assert result["anchors"]["auto_verify"][prov]["rate"] is None  # blank, not fabricated


def test_consensus_reached_with_three_providers_e2e():
    # 3 providers, identical reader answer -> consensus label forms at the default min=3.
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    adapters = [FakeAdapter(n, docs) for n in ("tavily", "brave", "serper")]
    reader = FakeLLM(complete_fn=lambda s, u: "The capital is Paris.")
    result = run_arena(ArenaConfig(evidence_budget_tokens=600),
                       [Query(query="capital of France?")], adapters,
                       _scope(["tavily", "brave", "serper"]), reader, FakeLLM(),
                       search_gatherer=sync_gather)
    anc = result["anchors"]
    assert anc["n_consensus_queries"] == 1 and anc["consensus_coverage"] == 1.0
    assert anc["min_providers"] == 3


# ---- config validation ----

def test_config_rejects_bad_consensus_min_providers():
    for bad in (1, 0, True, "3"):
        with pytest.raises(ValueError):
            ArenaConfig(consensus_min_providers=bad)
    ArenaConfig(consensus_min_providers=2)  # boundary is valid


# ---- zero-valued expected_answer is a valid gold answer, not "missing" ----

def test_zero_expected_answer_is_machine_verified(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = [EvidenceDoc(url="u", title="t", content="the count is zero")]
    a, b = FakeAdapter("tavily", docs), FakeAdapter("brave", docs)
    reader = FakeLLM(complete_fn=lambda s, u: "The count is 0 items.")
    result = run_arena(ArenaConfig(evidence_budget_tokens=600),
                       [Query(query="how many?", expected_answer="0")],
                       [a, b], _scope(["tavily", "brave"]), reader, FakeLLM(),
                       search_gatherer=sync_gather)
    for prov in ("tavily", "brave"):
        # "0" is falsy but a valid numeric gold -> machine-verified, not skipped.
        assert result["anchors"]["auto_verify"][prov]["total"] == 1
        assert result["metrics"][prov]["accuracy"]["correct"] == 1
