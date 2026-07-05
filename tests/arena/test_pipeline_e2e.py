"""End-to-end pipeline on fakes: symmetry, well-formed output, stage_status, secret canary."""

import json

from arena.adapters.base import EvidenceDoc
from arena.config import ArenaConfig, Query
from arena.judge import PairwiseVerdict
from arena.pipeline import run_arena
from arena.report import build_document
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(adapters, judge_structured_fn=None, queries=None):
    cfg = ArenaConfig(evidence_budget_tokens=600)
    queries = queries or [Query(query="what is the capital of France?")]
    reader_llm = FakeLLM()  # returns a fixed non-degenerate answer
    judge_llm = FakeLLM(structured_fn=judge_structured_fn) if judge_structured_fn else FakeLLM()
    scope = _scope([a.name for a in adapters])
    return run_arena(cfg, queries, adapters, scope, reader_llm, judge_llm,
                     search_gatherer=sync_gather)


def test_symmetry_identical_evidence_ties():
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    a = FakeAdapter("tavily", docs)
    b = FakeAdapter("brave", docs)
    # Enough queries to clear the min-comparisons floor; identical evidence for both providers
    # must yield identical scores and a single tie group (the neutrality invariant).
    result = _run([a, b], queries=[Query(query="what is the capital of France?")] * 3)
    scores = {s["provider"]: s for s in result["ranking"]}
    assert scores["tavily"]["win_rate"] == scores["brave"]["win_rate"] == 0.5
    assert result["tie_groups"] and set(result["tie_groups"][0]) == {"tavily", "brave"}


def test_wellformed_ranking_and_stage_status():
    docs = [EvidenceDoc(url="u", title="t", content="some evidence content about the topic")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs), FakeAdapter("serper", docs)]
    result = _run(adapters)
    assert set(result["stage_status"].keys()) == {
        "secrets", "adapters", "reader", "judge", "aggregate", "pipeline"}
    assert result["degenerate_run"] is False  # 3 providers
    assert len(result["ranking"]) == 3
    assert "swap_consistency" in result["judge"]


def test_degenerate_flag_two_providers():
    docs = [EvidenceDoc(url="u", title="t", content="evidence content here for the query")]
    result = _run([FakeAdapter("tavily", docs), FakeAdapter("brave", docs)])
    assert result["degenerate_run"] is True  # <3 providers


def test_empty_evidence_provider_not_forced_to_lose():
    good = FakeAdapter("tavily", [EvidenceDoc(url="u", title="t", content="real evidence content")])
    empty = FakeAdapter("brave", [])  # returns nothing
    result = _run([good, empty])
    m = result["metrics"]
    assert m["brave"]["empty_evidence_rate"] == 1.0
    assert m["brave"]["cells_succeeded"] == 0


def test_accuracy_column_from_gold(monkeypatch):
    from arena.grade import _GradeResult
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # force the Claude grader path
    docs = [EvidenceDoc(url="u", title="t", content="Paris is the capital of France.")]
    a = FakeAdapter("tavily", docs)
    b = FakeAdapter("brave", docs)
    cfg = ArenaConfig(evidence_budget_tokens=600)
    reader_llm = FakeLLM(complete_fn=lambda s, u: "The capital is Paris.")
    judge_llm = FakeLLM()  # ties
    grader_llm = FakeLLM(structured_fn=lambda s, u, sch: _GradeResult(correct="Paris" in u))
    scope = _scope(["tavily", "brave"])
    result = run_arena(cfg, [Query(query="capital of France?", expected_answer="Paris")],
                       [a, b], scope, reader_llm, judge_llm, search_gatherer=sync_gather,
                       grader_llm=grader_llm)
    for prov in ("tavily", "brave"):
        acc = result["metrics"][prov]["accuracy"]
        assert acc["total"] == 1 and acc["correct"] == 1 and acc["rate"] == 1.0


def test_accuracy_blank_without_expected_answer():
    docs = [EvidenceDoc(url="u", title="t", content="evidence content for the query here")]
    result = _run([FakeAdapter("tavily", docs), FakeAdapter("brave", docs)])  # queries have no gold
    assert result["metrics"]["tavily"]["accuracy"]["rate"] is None
    assert result["metrics"]["tavily"]["accuracy"]["total"] == 0


def test_secret_canary_absent_from_serialized_doc(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "SENTINEL_SECRET_123")
    docs = [EvidenceDoc(url="u", title="t", content="content mentioning SENTINEL_SECRET_123 here")]
    result = _run([FakeAdapter("tavily", docs), FakeAdapter("brave", docs)])
    doc = build_document(result, ["q"], config_snapshot={}, model_id="test-model")
    blob = json.dumps(doc)
    assert "SENTINEL_SECRET_123" not in blob  # redaction boundary scrubbed it


def test_pipelined_path_no_gatherer_runs(monkeypatch):
    # Exercise the real threaded orchestration (_pipelined_run) — no injected gatherer.
    # FakeAdapter.search is async, so it goes through the search-stream + LLM-worker threads.
    docs = [EvidenceDoc(url="u", title="t", content="evidence content about the topic here")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs), FakeAdapter("serper", docs)]
    cfg = ArenaConfig(evidence_budget_tokens=600, max_concurrency=4)
    scope = _scope([a.name for a in adapters])
    queries = [Query(query=f"q{i}?") for i in range(4)]
    result = run_arena(cfg, queries, adapters, scope, FakeLLM(), FakeLLM())  # default judge ties
    assert set(result["stage_status"].keys()) == {
        "secrets", "adapters", "reader", "judge", "aggregate", "pipeline"}
    assert len(result["ranking"]) == 3
    assert all(m["cells_succeeded"] == 4 for m in result["metrics"].values())  # every cell searched


def test_judge_flip_excluded_end_to_end():
    # Judge that always flips (A both passes) -> all comparisons excluded -> providers unranked.

    def flip_fn(system, user, schema):
        return PairwiseVerdict(winner="A", rationale="always A")

    docs = [EvidenceDoc(url="u", title="t", content="evidence content for the topic")]
    result = _run([FakeAdapter("tavily", docs), FakeAdapter("brave", docs)], judge_structured_fn=flip_fn)
    assert result["n_excluded_comparisons"] >= 1
    # every comparison flipped -> no decided comparisons -> both unranked
    assert all(s["status"] == "unranked" for s in result["ranking"])
