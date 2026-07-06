"""Tier-3 downstream hook (§3): the user's own loop, exit 0 = success. Real subprocesses
(tiny python -c commands) — deterministic, fast, no AI."""

import subprocess
import sys

import pytest

from arena.config import ArenaConfig, load_config
from arena.downstream import attach_downstream, run_downstream


def test_success_and_failure_counted():
    # Succeeds only for provider 'tavily' — via the {provider} substitution.
    cmd = f'{sys.executable} -c "import sys; sys.exit(0 if \'{{provider}}\' == \'tavily\' else 1)"'
    out = run_downstream(cmd, ["tavily", "brave"], runs=3, timeout_s=30)
    assert out["tavily"] == {"success_rate": 1.0, "successes": 3, "n": 3}
    assert out["brave"] == {"success_rate": 0.0, "successes": 0, "n": 3}


def test_provider_reaches_command_via_env():
    cmd = f'{sys.executable} -c "import os,sys; sys.exit(0 if os.environ[\'ARENA_PROVIDER\'] == \'exa\' else 1)"'
    out = run_downstream(cmd, ["exa"], runs=1, timeout_s=30)
    assert out["exa"]["success_rate"] == 1.0


def test_timeout_counts_as_failure():
    def fake_runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
    out = run_downstream("whatever {provider}", ["tavily"], runs=2, timeout_s=1,
                         runner=fake_runner)
    assert out["tavily"] == {"success_rate": 0.0, "successes": 0, "n": 2}


def test_unspawnable_command_counts_as_failure_not_crash():
    out = run_downstream("/no/such/binary-xyz {provider}", ["tavily"], runs=1, timeout_s=5)
    assert out["tavily"]["success_rate"] == 0.0


def test_attach_is_additive_and_absent_stays_absent():
    metrics = {"tavily": {"latency": {"p50": 100}}, "brave": {"latency": {"p50": 90}}}
    attach_downstream(metrics, {"tavily": {"success_rate": 0.8, "successes": 4, "n": 5}})
    assert metrics["tavily"]["downstream"]["success_rate"] == 0.8
    assert "downstream" not in metrics["brave"]           # absent, not zero (§8)
    assert metrics["tavily"]["latency"]["p50"] == 100     # untouched


def test_downstream_weight_axis_renormalizes():
    from arena.cost import effective_weights
    metrics = {"tavily": {"downstream": {"success_rate": 0.8}},
               "brave": {}}
    with_ds = effective_weights({"downstream": 0.5, "latency": 0.5}, metrics)
    assert "downstream" in with_ds                         # populated -> weight survives
    without = effective_weights({"downstream": 0.5, "latency": 0.5},
                                {"tavily": {}, "brave": {}})
    assert "downstream" not in without                     # absent -> dropped, renormalized


def test_config_parses_downstream_block(tmp_path):
    p = tmp_path / "arena.yaml"
    p.write_text('downstream:\n  command: "run.sh {provider}"\n  runs: 2\n  timeout_s: 60\n')
    cfg = load_config(str(p))
    assert cfg.downstream_command == "run.sh {provider}"
    assert cfg.downstream_runs == 2 and cfg.downstream_timeout_s == 60
    assert ArenaConfig().downstream_command is None        # off by default


def test_config_rejects_bad_runs():
    with pytest.raises(ValueError):
        ArenaConfig(downstream_runs=0)


def test_config_rejects_bad_timeout_and_command_type():
    with pytest.raises(ValueError):
        ArenaConfig(downstream_timeout_s=0)
    with pytest.raises(ValueError):
        ArenaConfig(downstream_command=["not", "a", "string"])


def test_unparseable_command_returns_empty_not_crash():
    # Config error, not provider signal: column stays absent (§8), run doesn't crash.
    out = run_downstream('bad "unclosed quote {provider}', ["tavily"], runs=2, timeout_s=5)
    assert out == {}


def test_csv_and_cli_render_downstream(tmp_path):
    import csv as csvmod
    from arena.report import render_cli_summary, write_results
    doc = {
        "timestamp": "t", "model_id": "m", "n_queries": 1, "cost_usd": None,
        "degenerate_run": False, "n_decided_comparisons": 2, "n_excluded_comparisons": 0,
        "judge": {}, "scope": {}, "stage_status": {},
        "ranking": [{"provider": "tavily", "win_rate": 0.6, "ci_low": 0.5, "ci_high": 0.7,
                     "n_comparisons": 2, "status": "ranked", "rank": 1, "tie_group": 0}],
        "metrics": {"tavily": {"coverage": {"avg_tokens_per_result": 100},
                               "downstream": {"success_rate": 0.8, "successes": 4, "n": 5}}},
    }
    paths = write_results(doc, str(tmp_path))
    with open(paths["csv"]) as f:
        row = next(csvmod.DictReader(f))
    assert row["downstream_success_rate"] == "0.8" and row["downstream_n"] == "5"
    assert "ds 80% (4/5)" in render_cli_summary(doc)
