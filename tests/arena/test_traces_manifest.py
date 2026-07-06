"""Audit traces (--save-traces) and the run manifest (§15 reproducibility)."""

import json
import os

from arena.adapters.base import EvidenceDoc
from arena.config import ArenaConfig, Query, load_config
from arena.pipeline import run_arena
from arena.report import build_document, run_manifest, write_traces
from arena.scope import INCLUDED, Scope, ScopeEntry
from _fakes import FakeAdapter, FakeLLM, sync_gather


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(save_traces):
    docs = [EvidenceDoc(url="http://u", title="t", content="evidence content about the topic")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    cfg = ArenaConfig(evidence_budget_tokens=600, save_traces=save_traces)
    queries = [Query(query="q0?", category="news"), Query(query="q1?")]
    return run_arena(cfg, queries, adapters, _scope([a.name for a in adapters]),
                     FakeLLM(), FakeLLM(), search_gatherer=sync_gather)


# ---- traces ----

def test_traces_collected_when_enabled():
    result = _run(save_traces=True)
    traces = result["traces"]
    assert len(traces) == 2
    t0 = traces[0]
    assert t0["query"] == "q0?" and t0["category"] == "news" and t0["repeat"] == 0
    for prov in ("tavily", "brave"):
        entry = t0["providers"][prov]
        assert entry["n_results"] == 1
        assert entry["evidence"][0]["url"] == "http://u"     # the exact docs the reader saw
        assert entry["reader_answer"]                        # and what it wrote from them


def test_traces_absent_when_disabled():
    result = _run(save_traces=False)
    assert result["traces"] is None


def test_write_traces_files_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "SENTINEL_SECRET_456")
    traces = [{"query": "q?", "providers": {"tavily": {
        "raw": {"api_key": "whatever", "note": "mentions SENTINEL_SECRET_456"}}}}]
    paths = write_traces(traces, str(tmp_path))
    assert paths == [os.path.join(str(tmp_path), "traces", "query_0000.json")]
    blob = open(paths[0]).read()
    assert "SENTINEL_SECRET_456" not in blob                 # value scrubbed
    assert json.loads(blob)["providers"]["tavily"]["raw"]["api_key"] == "«redacted»"


def test_save_traces_from_config_file(tmp_path):
    p = tmp_path / "arena.yaml"
    p.write_text("output:\n  save_traces: true\n")
    assert load_config(str(p)).save_traces is True
    assert ArenaConfig().save_traces is False


# ---- manifest ----

def test_manifest_shape_and_git_commit():
    m = run_manifest()
    assert set(m) == {"git_commit", "python", "packages"}
    # This repo is a git checkout, so the commit must resolve to a 40-hex sha.
    assert m["git_commit"] and len(m["git_commit"]) == 40
    assert m["python"]
    assert m["packages"].get("numpy")                        # installed dep resolves a version


def test_manifest_lands_in_results_document():
    result = _run(save_traces=False)
    doc = build_document(result, ["q0?", "q1?"], config_snapshot={}, model_id="m")
    assert doc["manifest"]["git_commit"]
    assert "traces" not in doc                               # traces stay out of results.json
