"""Cold-start CLI behavior: --demo, the ANTHROPIC_API_KEY preflight, friendly errors for
bad input (no tracebacks at the CLI boundary), and the .env.example key inventory."""

import json
import logging
import os

import pytest

import compare_runs
import run_arena
from arena.adapters.registry import REGISTRY

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _clear_api_keys(monkeypatch):
    """Simulate a keyless environment (also drops GOOGLE_API_KEY via the _API_KEY suffix)."""
    for k in list(os.environ):
        if k.endswith("_API_KEY"):
            monkeypatch.delenv(k, raising=False)


def _no_op_load_secrets(monkeypatch):
    """Keep tests hermetic: never read the developer's real .env or call Doppler."""
    monkeypatch.setattr(run_arena.secrets, "load_secrets", lambda *a, **kw: None)


# ---- --demo: zero keys, zero network ---------------------------------------------------------

def test_demo_renders_without_any_keys(monkeypatch, capsys):
    _clear_api_keys(monkeypatch)
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--demo"])
    assert run_arena.main() == 0
    out = capsys.readouterr().out
    assert "DEMO — a real committed run" in out
    assert "$15.47" in out
    assert "SEARCH ARENA" in out          # went through the real renderer
    assert "RANKING" in out


def test_demo_works_from_any_cwd(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)           # demo file resolves relative to run_arena.py, not cwd
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--demo"])
    assert run_arena.main() == 0
    assert "SEARCH ARENA" in capsys.readouterr().out


def test_queries_still_required_without_demo(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_arena.py"])
    with pytest.raises(SystemExit) as exc:
        run_arena.main()
    assert exc.value.code == 2
    assert "--queries is required (or use --demo" in capsys.readouterr().err


# ---- ANTHROPIC_API_KEY preflight -------------------------------------------------------------

def test_missing_anthropic_key_fails_fast_with_named_key(monkeypatch, caplog, tmp_path):
    _no_op_load_secrets(monkeypatch)
    _clear_api_keys(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "present-but-not-enough")
    q = tmp_path / "q.csv"
    q.write_text("query\nwho is x?\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--queries", str(q)])
    with caplog.at_level(logging.ERROR):
        assert run_arena.main() == 1
    assert "ANTHROPIC_API_KEY" in caplog.text
    assert "judge and reader run on the Anthropic" in caplog.text


def test_zero_provider_error_mentions_anthropic_key(monkeypatch, caplog, tmp_path):
    _no_op_load_secrets(monkeypatch)
    _clear_api_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    q = tmp_path / "q.csv"
    q.write_text("query\nwho is x?\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("providers:\n  claude_search: {enabled: false}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--queries", str(q), "--config", str(cfg)])
    with caplog.at_level(logging.ERROR):
        assert run_arena.main() == 1
    assert "No providers are enabled with a key present" in caplog.text
    assert "ANTHROPIC_API_KEY is additionally required" in caplog.text


# ---- friendly errors at the CLI boundary (message, not traceback) ----------------------------

@pytest.fixture
def _keyed_env(monkeypatch, tmp_path):
    """ANTHROPIC key present, hermetic secrets, cwd off the repo (no configs/arena.yaml)."""
    _no_op_load_secrets(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)


def test_missing_queries_file_prints_clean_error(_keyed_env, monkeypatch, caplog, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--queries", str(tmp_path / "nope.csv")])
    with caplog.at_level(logging.ERROR):
        assert run_arena.main() == 1
    assert "Queries file not found" in caplog.text


def test_malformed_queries_row_prints_clean_error(_keyed_env, monkeypatch, caplog, tmp_path):
    q = tmp_path / "q.csv"
    q.write_text("notquery\nx\n")     # no 'query' column -> ValueError in load_queries
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--queries", str(q)])
    with caplog.at_level(logging.ERROR):
        assert run_arena.main() == 1
    assert "missing a non-empty 'query' field" in caplog.text


def test_missing_config_file_prints_clean_error(_keyed_env, monkeypatch, caplog, tmp_path):
    q = tmp_path / "q.csv"
    q.write_text("query\nwho is x?\n")
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--queries", str(q),
                                     "--config", str(tmp_path / "nope.yaml")])
    with caplog.at_level(logging.ERROR):
        assert run_arena.main() == 1
    assert "Config file not found" in caplog.text


def test_compare_runs_missing_file_prints_clean_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["compare_runs.py", str(tmp_path / "b.json"),
                                     str(tmp_path / "a.json")])
    assert compare_runs.main() == 1
    assert "Results file not found" in capsys.readouterr().err


def test_compare_runs_bad_json_prints_clean_error(monkeypatch, capsys, tmp_path):
    b, a = tmp_path / "b.json", tmp_path / "a.json"
    b.write_text("not json {")
    a.write_text("{}")
    monkeypatch.setattr("sys.argv", ["compare_runs.py", str(b), str(a)])
    assert compare_runs.main() == 1
    assert "Not a valid results.json" in capsys.readouterr().err


# ---- cheap-first-run guidance (--help epilog) -------------------------------------------------

def test_help_epilog_has_first_run_recipe(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_arena.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        run_arena.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "first run?" in out
    assert "--demo" in out
    assert "claude-haiku-4-5-20251001" in out
    assert "$15.47" in out                # the measured number, not an estimate


# ---- .env.example inventory -------------------------------------------------------------------

def test_env_example_covers_every_registry_key():
    text = open(os.path.join(_ROOT, ".env.example")).read()
    keys = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"}
    for spec in REGISTRY.values():
        keys.update(spec.required_env_keys)
        keys.update(spec.any_of_env_keys)
    missing = sorted(k for k in keys if k not in text)
    assert not missing, f".env.example is missing: {missing}"
