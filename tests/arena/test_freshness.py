"""Freshness scoring (§8.3): dated-in-window share + disclosed date coverage.

Tier-A, deterministic, no AI. A fixed ``now`` keeps every window assertion reproducible;
undated results are excluded from the score but counted in coverage; low coverage flags the
score low-confidence; and when no query is time-sensitive, freshness is absent so its weight is
dropped and the remaining weights renormalize (reusing ``renormalize_weights``).
"""

from datetime import datetime, timezone

from arena.adapters.base import EvidenceDoc
from arena.config import ArenaConfig, Query
from arena.metrics import (aggregate_freshness, freshness_score,
                           parse_freshness_window_days, parse_reliable_date,
                           renormalize_weights)
from arena.pipeline import run_arena
from arena.report import build_document, write_results
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather

NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


def _doc(published=None, content="body text"):
    return EvidenceDoc(url="http://x", title="t", content=content, published_date=published)


# ---- window parsing ----

def test_window_parsing_aliases_and_integers():
    assert parse_freshness_window_days("week") == 7
    assert parse_freshness_window_days("month") == 30
    assert parse_freshness_window_days("year") == 365
    assert parse_freshness_window_days("14") == 14      # bare integer = days
    assert parse_freshness_window_days("7d") == 7
    assert parse_freshness_window_days("3 days") == 3


def test_window_parsing_falls_back_to_default():
    assert parse_freshness_window_days("", default_days=30) == 30
    assert parse_freshness_window_days("whenever", default_days=30) == 30
    assert parse_freshness_window_days(None, default_days=45) == 45


# ---- date parsing (never guessed) ----

def test_parse_reliable_date_prefers_published_then_content():
    assert parse_reliable_date(_doc(published="2026-07-01"), now=NOW).day == 1
    # No published_date -> parse an ISO date embedded in content.
    d = parse_reliable_date(_doc(published=None, content="Posted 2026-06-15 by staff"), now=NOW)
    assert d is not None and d.month == 6 and d.day == 15


def test_parse_reliable_date_none_when_absent_or_future():
    assert parse_reliable_date(_doc(published=None, content="no dates in this text"), now=NOW) is None
    # A future date is unreliable, not counted (guards against garbage timestamps).
    assert parse_reliable_date(_doc(published="2099-01-01"), now=NOW) is None


def test_free_text_iso_ignores_version_like_strings():
    # A date-shaped substring inside a version/build/ID string must NOT be read as a publish date
    # (the free-text last-resort match is flanked-digit/dot/dash guarded).
    for junk in ("build 1.2026-01-01", "id 2026-01-01.5", "release v3-2026-01-01"):
        assert parse_reliable_date(_doc(published=None, content=junk), now=NOW) is None
    # A genuinely standalone date in content is still picked up.
    assert parse_reliable_date(_doc(published=None, content="Published 2026-06-15."), now=NOW) is not None


# ---- score on known dates within/outside the window ----

def test_score_counts_only_dated_in_window():
    docs = [
        _doc(published="2026-07-01"),   # 5 days old  -> in a 30d window
        _doc(published="2026-06-20"),   # 16 days old -> in window
        _doc(published="2026-01-01"),   # ~half a year -> OUT of a 30d window
    ]
    out = freshness_score(docs, window_days=30, now=NOW)
    assert out["dated"] == 3 and out["n_results"] == 3
    assert out["coverage"] == 1.0
    assert out["in_window"] == 2
    assert out["score"] == 2 / 3
    assert out["low_confidence"] is False


def test_undated_excluded_from_score_but_counted_in_coverage():
    docs = [
        _doc(published="2026-07-01"),                  # dated, in window
        _doc(published=None, content="no date here"),  # undated -> coverage only
        _doc(published=None, content="also undated"),  # undated -> coverage only
    ]
    out = freshness_score(docs, window_days=30, now=NOW)
    assert out["dated"] == 1                 # undated excluded from the score's denominator
    assert out["in_window"] == 1
    assert out["score"] == 1.0               # 1/1 dated results are in-window
    assert out["coverage"] == 1 / 3          # only 1 of 3 carried a usable date
    assert out["low_confidence"] is True     # coverage < 0.5 -> score is low-confidence


def test_all_undated_score_is_none_not_zero():
    docs = [_doc(published=None, content="x"), _doc(published=None, content="y")]
    out = freshness_score(docs, window_days=30, now=NOW)
    assert out["score"] is None              # never fabricate a 0 for a missing date
    assert out["coverage"] == 0.0
    assert out["low_confidence"] is True


def test_high_coverage_is_not_low_confidence():
    docs = [_doc(published="2026-07-01"), _doc(published="2026-06-25"),
            _doc(published=None, content="undated")]
    out = freshness_score(docs, window_days=30, now=NOW)
    assert out["coverage"] == 2 / 3          # >= 0.5 bar
    assert out["low_confidence"] is False


# ---- aggregation across time-sensitive queries ----

def test_aggregate_freshness_pools_counts():
    q1 = freshness_score([_doc(published="2026-07-01"), _doc(published="2026-01-01")], 30, now=NOW)
    q2 = freshness_score([_doc(published="2026-06-30")], 30, now=NOW)
    agg = aggregate_freshness([q1, q2])
    assert agg["n_results"] == 3 and agg["dated"] == 3
    assert agg["in_window"] == 2 and agg["score"] == 2 / 3
    assert agg["n_queries"] == 2


def test_aggregate_freshness_absent_when_no_timesensitive_results():
    assert aggregate_freshness([]) is None
    assert aggregate_freshness([{"n_results": 0, "dated": 0, "in_window": 0}]) is None


# ---- renormalization: freshness absent -> weight dropped + remaining renormalized (§8) ----

def test_freshness_absent_weight_dropped_and_renormalized():
    w = renormalize_weights({"latency": 0.5, "coverage": 0.3, "freshness": 0.2},
                            present_metrics=["latency", "coverage"])  # freshness absent
    assert "freshness" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["latency"] - 0.625) < 1e-9   # 0.5 / 0.8
    assert abs(w["coverage"] - 0.375) < 1e-9  # 0.3 / 0.8


def test_freshness_present_kept_in_weights():
    w = renormalize_weights({"latency": 0.5, "freshness": 0.5},
                            present_metrics=["latency", "freshness"])
    assert set(w) == {"latency", "freshness"} and abs(sum(w.values()) - 1.0) < 1e-9


# ---- pipeline wiring: freshness flows into metrics/results.json only on time-sensitive runs ----

def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(adapters, queries):
    cfg = ArenaConfig(evidence_budget_tokens=600)
    scope = _scope([a.name for a in adapters])
    return run_arena(cfg, queries, adapters, scope, FakeLLM(), FakeLLM(),
                     search_gatherer=sync_gather)


def test_pipeline_emits_freshness_on_timesensitive_query():
    # Old date -> out of the 7-day window; low date coverage flagged low-confidence.
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.",
                        published_date="2020-01-01"),
            EvidenceDoc(url="u2", title="t2", content="No date in this evidence text at all."),
            EvidenceDoc(url="u3", title="t3", content="Still no date to be found here.")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    result = _run(adapters, [Query(query="latest news?", freshness_need="week")])
    fr = result["metrics"]["tavily"]["freshness"]
    assert fr["dated"] == 1 and fr["n_results"] == 3      # one dated, two undated
    assert fr["score"] == 0.0                             # dated result is out of window
    assert fr["coverage"] == 1 / 3
    assert fr["low_confidence"] is True                   # coverage below the bar
    # Freshness survives into the canonical results.json document.
    doc = build_document(result, ["q"], config_snapshot={}, model_id="m")
    assert doc["metrics"]["tavily"]["freshness"]["score"] == 0.0


def test_pipeline_omits_freshness_when_no_timesensitive_query():
    docs = [EvidenceDoc(url="u", title="t", content="evidence content", published_date="2026-07-01")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    result = _run(adapters, [Query(query="plain query?")])  # no freshness_need
    assert "freshness" not in result["metrics"]["tavily"]


def test_freshness_columns_written_to_csv(tmp_path):
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital.",
                        published_date="2026-07-01")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    result = _run(adapters, [Query(query="latest?", freshness_need="month")])
    doc = build_document(result, ["q"], config_snapshot={}, model_id="m")
    paths = write_results(doc, str(tmp_path))
    header = open(paths["csv"]).readline()
    assert "freshness_score" in header and "freshness_coverage" in header
    assert "freshness_low_confidence" in header
