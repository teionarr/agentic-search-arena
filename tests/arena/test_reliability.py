"""Reliability column: errored calls are counted apart from genuine empty results."""

from arena.adapters.base import EvidenceDoc, UnifiedResult
from arena.config import ArenaConfig, Query
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


class RawAdapter:
    """Adapter returning a preset UnifiedResult verbatim (for error-shape fixtures)."""

    def __init__(self, name, result):
        self.name = name
        self._result = result

    async def search(self, query):
        return self._result


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(adapters, n_queries=2):
    cfg = ArenaConfig(evidence_budget_tokens=600)
    queries = [Query(query=f"q{i}?") for i in range(n_queries)]
    return run_arena(cfg, queries, adapters, _scope([a.name for a in adapters]),
                     FakeLLM(), FakeLLM(), search_gatherer=sync_gather)


def test_errored_provider_counted_as_error_not_just_empty():
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    erroring = RawAdapter("brave", UnifiedResult(raw={"error": "boom"}, empty_evidence=True))
    result = _run([good, erroring])
    rel = result["metrics"]["brave"]["reliability"]
    assert rel["success_rate"] == 0.0
    assert rel["error_rate"] == 1.0 and rel["errors"] == 2
    assert result["metrics"]["tavily"]["reliability"] == {
        "success_rate": 1.0, "error_rate": 0.0, "errors": 0}


def test_swallowed_error_shape_counts_as_error():
    # Base handlers swallow provider errors into raw={"search_response": None}.
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    swallowed = RawAdapter("brave", UnifiedResult(raw={"search_response": None}, empty_evidence=True))
    result = _run([good, swallowed])
    assert result["metrics"]["brave"]["reliability"]["error_rate"] == 1.0


def test_genuine_empty_is_not_an_error():
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    empty = FakeAdapter("brave", [])  # found nothing; did not error
    result = _run([good, empty])
    rel = result["metrics"]["brave"]["reliability"]
    assert rel["success_rate"] == 0.0     # still unsuccessful...
    assert rel["error_rate"] == 0.0       # ...but weak, not unreliable
    assert result["metrics"]["brave"]["empty_evidence_rate"] == 1.0


def test_csv_has_reliability_columns(tmp_path):
    import csv as csvmod
    from arena.report import build_document, write_results
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    erroring = RawAdapter("brave", UnifiedResult(raw={"error": "boom"}, empty_evidence=True))
    doc = build_document(_run([good, erroring]), ["q"], config_snapshot={}, model_id="m")
    paths = write_results(doc, str(tmp_path))
    with open(paths["csv"]) as f:
        rows = list(csvmod.DictReader(f))
    by_prov = {r["provider"]: r for r in rows}
    assert by_prov["brave"]["error_rate"] == "1.0"
    assert by_prov["tavily"]["success_rate"] == "1.0"


def test_cli_flags_only_deviating_reliability():
    from arena.report import build_document, render_cli_summary
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    erroring = RawAdapter("brave", UnifiedResult(raw={"error": "boom"}, empty_evidence=True))
    doc = build_document(_run([good, erroring]), ["q"], config_snapshot={}, model_id="m")
    text = render_cli_summary(doc)
    assert "ok 0%" in text            # the failing provider is flagged
    assert "ok 100%" not in text      # healthy providers stay unannotated
