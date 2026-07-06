"""Config parsing, queries loading, and scope attribution."""


import pytest

from arena import scope as scope_mod
from arena.config import load_config, load_queries


def test_load_queries_csv(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("query,expected_answer,category\nwho is x?,X,people\nwhat is y?,,\n")
    qs = load_queries(str(p))
    assert len(qs) == 2
    assert qs[0].query == "who is x?" and qs[0].expected_answer == "X"
    assert qs[1].expected_answer is None  # blank optional col -> None


def test_load_queries_jsonl(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"query": "a?"}\n{"query": "b?", "category": "c"}\n')
    qs = load_queries(str(p))
    assert [q.query for q in qs] == ["a?", "b?"] and qs[1].category == "c"


def test_load_queries_missing_query_errors(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("notquery\nx\n")
    with pytest.raises(ValueError):
        load_queries(str(p))


def test_config_rejects_yaml_object_injection(tmp_path):
    # safe_load must reject !!python/object tags rather than execute them.
    p = tmp_path / "c.yaml"
    p.write_text("providers: {}\nevil: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception):
        load_config(str(p))


def test_config_unknown_provider_key_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("providers:\n  not_a_provider: {enabled: true}\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_config_defaults_zero_config():
    cfg = load_config(None)
    assert cfg.order_swap is True and cfg.evidence_budget_tokens > 0


def test_config_rejects_nonpositive_budget_and_concurrency(tmp_path):
    from arena.config import ArenaConfig
    with pytest.raises(ValueError):
        ArenaConfig(evidence_budget_tokens=0)          # would silently disable the evidence cap
    with pytest.raises(ValueError):
        ArenaConfig(max_concurrency=0)
    p = tmp_path / "c.yaml"
    p.write_text("evidence_budget_tokens: -5\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def _clear_provider_keys(monkeypatch):
    for k in ["TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY", "SERPER_API_KEY", "PERPLEXITY_API_KEY"]:
        monkeypatch.delenv(k, raising=False)


def test_scope_no_keys_all_excluded(monkeypatch):
    _clear_provider_keys(monkeypatch)
    sc = scope_mod.resolve_scope({})
    assert sc.included == []
    assert all(e.status == scope_mod.NO_KEY for e in sc.entries)


def test_scope_key_present_included(monkeypatch):
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    sc = scope_mod.resolve_scope({})
    assert "tavily" in sc.included
    assert sc.as_dict()["exa"]["status"] == scope_mod.NO_KEY


def test_scope_user_choice_excluded(monkeypatch):
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    sc = scope_mod.resolve_scope({"tavily": {"enabled": False}})
    assert "tavily" not in sc.included
    assert sc.as_dict()["tavily"]["status"] == scope_mod.USER_CHOICE


def test_scope_mark_runtime_error(monkeypatch):
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    sc = scope_mod.resolve_scope({})
    assert "tavily" in sc.included
    sc.mark_runtime_error("tavily", "boom: HTTP 500")
    assert "tavily" not in sc.included                       # drops out of the run
    entry = sc.as_dict()["tavily"]
    assert entry["status"] == scope_mod.RUNTIME_ERROR and "boom" in entry["detail"]
