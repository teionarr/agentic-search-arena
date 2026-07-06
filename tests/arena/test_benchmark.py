"""Benchmark-suite mode (M2, §7): connectors into the common schema, calibration agreement,
marketing-claims ledger delta, sample sizing, and the arena-vs-benchmark cross-signal.

All Tier-A: deterministic, NO AI, NO network — loaders read fixtures; the calibration path is
driven by mocked reader/judge/grader LLMs (same fakes the calibration tests use).
"""

import os

import pytest

from arena.adapters.base import EvidenceDoc
from arena.benchmark import (build_ledger, cross_signal, load_benchmark, load_frames,
                             load_freshqa, load_published_claims, load_simpleqa,
                             run_benchmark_suite)
from arena.config import ArenaConfig, Query, load_config
from arena.grade import _GradeResult
from arena.judge import PairwiseVerdict
from _fakes import FakeAdapter, FakeLLM, sync_gather

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


# ---- connectors: load into the common Query schema (loader test, no network) ----

def test_load_simpleqa_from_vendored_csv():
    rows = load_simpleqa(3)
    assert len(rows) == 3
    assert all(isinstance(q, Query) and q.query for q in rows)
    assert all(q.expected_answer for q in rows)       # SimpleQA carries gold
    assert all(q.category == "simpleqa" for q in rows)


def test_load_frames_fixture_maps_prompt_answer():
    rows = load_frames(10, path=os.path.join(FIX, "frames_sample.csv"))
    assert len(rows) == 3
    assert rows[0].query.startswith("What is the capital")
    assert rows[0].expected_answer == "Tokyo"
    assert all(q.category == "frames" for q in rows)


def test_load_freshqa_fixture_jsonl_and_freshness_tag():
    rows = load_freshqa(10, path=os.path.join(FIX, "freshqa_sample.jsonl"))
    assert len(rows) == 2
    assert rows[0].expected_answer == "Sam Altman"
    assert rows[1].expected_answer is None            # empty answer -> None (unanchored row)
    assert all(q.freshness_need for q in rows)         # freshness tag always populated
    assert all(q.category == "freshqa" for q in rows)


def test_load_benchmark_sample_sizing_honored():
    assert len(load_benchmark("simpleqa", sample_size=5)) == 5
    assert len(load_benchmark("simpleqa", sample_size=1)) == 1


def test_load_benchmark_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        load_benchmark("browsecomp", sample_size=10)


def test_load_benchmark_rejects_nonpositive_sample():
    with pytest.raises(ValueError):
        load_benchmark("simpleqa", sample_size=0)


def test_missing_data_file_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        load_frames(10, path=os.path.join(FIX, "does_not_exist.csv"))


# ---- config parsing of modes.benchmark_suite ----

def test_config_benchmark_suite_off_by_default():
    cfg = load_config(None)
    assert cfg.benchmark_suite is False
    assert cfg.benchmark_datasets == ["simpleqa"] and cfg.benchmark_sample_size == 300


def test_config_benchmark_suite_parsed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("modes:\n  benchmark_suite:\n    enabled: true\n"
                 "    datasets: [simpleqa, frames]\n    sample_size: 50\n"
                 "    published_claims_path: configs/published_claims.yaml\n")
    cfg = load_config(str(p))
    assert cfg.benchmark_suite is True
    assert cfg.benchmark_datasets == ["simpleqa", "frames"]
    assert cfg.benchmark_sample_size == 50
    assert cfg.published_claims_path == "configs/published_claims.yaml"


# ---- marketing-claims ledger (pure): delta + trace, no accusation ----

def test_build_ledger_delta_from_published_vs_rerun():
    rerun = {"tavily": {"rate": 0.88, "correct": 88, "total": 100},
             "exa": {"rate": 0.80, "correct": 80, "total": 100}}
    published = {"tavily": {"score": 0.93, "as_of": "2025-01-15", "source": "blog"}}
    rows = build_ledger("simpleqa", rerun, published, "2026-07-06T00:00:00")
    by = {r["provider"]: r for r in rows}
    # tavily: neutral re-run 0.88 vs published 0.93 -> delta -0.05, trace preserved.
    assert by["tavily"]["delta"] == pytest.approx(-0.05)
    assert by["tavily"]["published_as_of"] == "2025-01-15"
    assert by["tavily"]["published_source"] == "blog"
    assert by["tavily"]["rerun_as_of"] == "2026-07-06T00:00:00"
    # exa: no published number -> delta None, but the neutral re-run row is still present (§7).
    assert by["exa"]["delta"] is None
    assert by["exa"]["rerun_rate"] == 0.80


def test_build_ledger_rerun_only_when_no_published():
    rows = build_ledger("frames", {"tavily": {"rate": 0.5, "correct": 5, "total": 10}}, {}, "t")
    assert len(rows) == 1 and rows[0]["delta"] is None and rows[0]["published_score"] is None


def test_load_published_claims_example_file():
    claims = load_published_claims("configs/published_claims.example.yaml")
    assert "simpleqa" in claims
    assert claims["simpleqa"]["tavily"]["score"] == 0.93
    assert claims["simpleqa"]["tavily"]["as_of"] == "2025-01-15"


def test_load_published_claims_missing_file_is_empty():
    assert load_published_claims(None) == {}
    assert load_published_claims("configs/nope.yaml") == {}


# ---- cross-signal ----

def test_cross_signal_none_without_arena():
    assert cross_signal(None, {"datasets": {}}) is None


def test_cross_signal_orders_both_sides():
    arena = [{"provider": "tavily", "rank": 1}, {"provider": "exa", "rank": 2}]
    report = {"datasets": {"simpleqa": {"benchmark_rank": [
        {"provider": "exa", "rank": 1}, {"provider": "tavily", "rank": 2}]}}}
    cs = cross_signal(arena, report)
    assert cs["arena_order"] == ["tavily", "exa"]
    assert cs["benchmark_order"]["simpleqa"] == ["exa", "tavily"]  # disagreement is the insight


# ---- calibration agreement % via the shared arena run (mocked LLMs, no network) ----

def _reader_fn(system, user):
    return "The capital is Paris." if "Paris" in user else "The capital is Lyon."


def _grader_fn(system, user, schema):
    pred = user.split("Predicted answer:")[-1]
    return _GradeResult(correct="Paris" in pred)


def test_run_benchmark_suite_calibration_and_ledger(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # force Claude grader path
    good = FakeAdapter("tavily", [EvidenceDoc(url="a", title="t",
                                              content="Paris is the capital of France.")])
    bad = FakeAdapter("brave", [EvidenceDoc(url="b", title="t",
                                            content="Lyon is a city in France.")])
    # A one-row benchmark file so we exercise the real loader path too.
    data = tmp_path / "mini.csv"
    data.write_text("problem,answer\ncapital of France?,Paris\n")
    monkeypatch.setitem(__import__("arena.benchmark", fromlist=["DATASET_PATHS"]).DATASET_PATHS,
                        "simpleqa", str(data))

    published = {"simpleqa": {"tavily": {"score": 1.0, "as_of": "2025-01-01", "source": "s"}}}
    report = run_benchmark_suite(
        ["simpleqa"], sample_size=50, adapters=[good, bad],
        reader_llm=FakeLLM(complete_fn=_reader_fn),
        # Judge prefers whichever Answer block cites Paris (the correct fact).
        judge_llm=FakeLLM(structured_fn=lambda s, u, sch: PairwiseVerdict(
            winner="A" if "Paris" in u[u.find("### Answer A"):u.find("### Answer B")] else "B",
            rationale="x")),
        grader_llm=FakeLLM(structured_fn=_grader_fn),
        config=ArenaConfig(evidence_budget_tokens=600),
        published=published, search_gatherer=sync_gather,
    )
    d = report["datasets"]["simpleqa"]
    # One decidable pair, judge picked the correct answer -> perfect agreement (§6.5).
    assert d["calibration"]["agreement"] == 1.0
    assert d["calibration"]["n_decidable_pairs"] == 1
    assert d["calibration"]["grader"] == "claude"
    # Ledger: tavily graded correct (rate 1.0) vs published 1.0 -> delta 0.0, trace preserved.
    led = {r["provider"]: r for r in d["ledger"]}
    assert led["tavily"]["rerun_rate"] == 1.0
    assert led["tavily"]["delta"] == pytest.approx(0.0)
    assert led["tavily"]["published_source"] == "s"


def test_run_benchmark_suite_without_published_still_produces_rerun(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    good = FakeAdapter("tavily", [EvidenceDoc(url="a", title="t",
                                              content="Paris is the capital of France.")])
    bad = FakeAdapter("brave", [EvidenceDoc(url="b", title="t",
                                            content="Lyon is a city in France.")])
    data = tmp_path / "mini.csv"
    data.write_text("problem,answer\ncapital of France?,Paris\n")
    monkeypatch.setitem(__import__("arena.benchmark", fromlist=["DATASET_PATHS"]).DATASET_PATHS,
                        "simpleqa", str(data))
    report = run_benchmark_suite(
        ["simpleqa"], sample_size=50, adapters=[good, bad],
        reader_llm=FakeLLM(complete_fn=_reader_fn),
        judge_llm=FakeLLM(structured_fn=lambda s, u, sch: PairwiseVerdict(winner="tie", rationale="x")),
        grader_llm=FakeLLM(structured_fn=_grader_fn),
        config=ArenaConfig(evidence_budget_tokens=600),
        published={}, search_gatherer=sync_gather,
    )
    d = report["datasets"]["simpleqa"]
    assert len(d["ledger"]) == 2                       # neutral re-run row per provider
    assert all(r["published_score"] is None and r["delta"] is None for r in d["ledger"])
