"""Tier-A tests for the post-hoc weighted re-ranking (§8/§10): arena/rerank.py engine +
CLI, and the WEIGHTED dashboard section in arena/report.py. Deterministic, no AI, no APIs."""

import copy
import json
import os

import pytest

from arena.rerank import (AXES, effective_axis_weights, main, render_rerank,
                          weighted_scores)
from arena.report import render_cli_summary

EXAMPLE_RESULTS = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "example-run", "results.json"))


def _rank_entry(provider, rank, win_rate, status="ranked"):
    ci_low = max(0.0, win_rate - 0.05) if win_rate is not None else None
    ci_high = min(1.0, win_rate + 0.05) if win_rate is not None else None
    return {"provider": provider, "win_rate": win_rate, "ci_low": ci_low,
            "ci_high": ci_high, "n_comparisons": 40,
            "status": status, "rank": rank, "tie_group": 0}


def make_doc():
    """Three providers; latency and freshness discriminate; no cost/downstream data."""
    return {
        "ranking": [
            _rank_entry("alpha", 1, 0.8),
            _rank_entry("beta", 2, 0.6),
            _rank_entry("gamma", 3, 0.2),
        ],
        "metrics": {
            "alpha": {"latency": {"p50": 100.0}, "freshness": {"score": 0.2},
                      "accuracy": {"rate": 1.0}},
            "beta": {"latency": {"p50": 300.0}, "freshness": {"score": 1.0},
                     "accuracy": {"rate": 1.0}},
            "gamma": {"latency": {"p50": 500.0}, "freshness": {"score": 0.6},
                      "accuracy": {"rate": 1.0}},
        },
    }


def per_axis(scored, provider):
    return next(e for e in scored if e["provider"] == provider)["per_axis"]


# ---- normalization + inversion -------------------------------------------------------------

def test_minmax_normalization_and_inversion():
    scored = weighted_scores(make_doc(), {"latency": 0.5, "freshness": 0.5})
    # latency is lower-is-better: 100 -> 1.0, 300 -> 0.5, 500 -> 0.0
    assert per_axis(scored, "alpha")["latency"]["normalized"] == 1.0
    assert per_axis(scored, "beta")["latency"]["normalized"] == 0.5
    assert per_axis(scored, "gamma")["latency"]["normalized"] == 0.0
    # freshness is higher-is-better: 0.2 -> 0.0, 0.6 -> 0.5, 1.0 -> 1.0
    assert per_axis(scored, "alpha")["freshness"]["normalized"] == 0.0
    assert per_axis(scored, "beta")["freshness"]["normalized"] == 1.0
    assert per_axis(scored, "gamma")["freshness"]["normalized"] == pytest.approx(0.5)
    # raw values are carried through
    assert per_axis(scored, "alpha")["latency"]["value"] == 100.0


def test_weighted_score_formula_and_order():
    scored = weighted_scores(make_doc(), {"latency": 0.75, "freshness": 0.25})
    by_p = {e["provider"]: e for e in scored}
    assert by_p["alpha"]["weighted_score"] == pytest.approx(0.75 * 1.0 + 0.25 * 0.0)
    assert by_p["beta"]["weighted_score"] == pytest.approx(0.75 * 0.5 + 0.25 * 1.0)
    assert by_p["gamma"]["weighted_score"] == pytest.approx(0.25 * 0.5)
    assert [e["provider"] for e in scored] == ["alpha", "beta", "gamma"]
    assert [e["rank"] for e in scored] == [1, 2, 3]


def test_degenerate_axis_all_equal_scores_one():
    # accuracy is 1.0 for everyone -> the axis cannot discriminate; every holder gets 1.0
    scored = weighted_scores(make_doc(), {"accuracy": 1.0})
    assert all(e["per_axis"]["accuracy"]["normalized"] == 1.0 for e in scored)


# ---- §8 honesty: missing value on a present axis -> per-provider renormalization ------------

def test_missing_value_renormalizes_weights_for_that_provider():
    doc = make_doc()
    del doc["metrics"]["beta"]["freshness"]  # axis still present via alpha/gamma
    scored = weighted_scores(doc, {"latency": 0.5, "freshness": 0.5})
    beta = per_axis(scored, "beta")
    # beta is unscored on freshness (never zero-scored) ...
    assert beta["freshness"]["value"] is None
    assert beta["freshness"]["normalized"] is None
    assert beta["freshness"]["weight"] is None
    # ... and its remaining weight renormalizes to 1.0 on latency
    assert beta["latency"]["weight"] == pytest.approx(1.0)
    by_p = {e["provider"]: e for e in scored}
    assert by_p["beta"]["weighted_score"] == pytest.approx(1.0 * 0.5)  # latency norm 0.5
    # providers WITH the value keep the shared effective weights
    assert per_axis(scored, "alpha")["freshness"]["weight"] == pytest.approx(0.5)


def test_provider_with_no_scored_axes_is_listed_unscored_last():
    doc = make_doc()
    doc["ranking"].append(_rank_entry("delta", None, None, status="unranked"))
    doc["metrics"]["delta"] = {}
    scored = weighted_scores(doc, {"latency": 1.0})
    assert scored[-1]["provider"] == "delta"
    assert scored[-1]["weighted_score"] is None
    assert scored[-1]["rank"] is None


# ---- §8 honesty: absent axis -> dropped + renormalized --------------------------------------

def test_absent_axis_dropped_and_renormalized():
    doc = make_doc()  # no provider has cost
    eff = effective_axis_weights(doc, {"cost": 0.5, "latency": 0.25, "freshness": 0.25})
    assert "cost" not in eff
    assert eff["latency"] == pytest.approx(0.5)
    assert eff["freshness"] == pytest.approx(0.5)
    scored = weighted_scores(doc, {"cost": 0.5, "latency": 0.25, "freshness": 0.25})
    assert "cost" not in per_axis(scored, "alpha")  # per_axis covers effective axes only


def test_unranked_provider_unscored_on_arena_axis_only():
    doc = make_doc()
    doc["ranking"].append(_rank_entry("delta", None, 0.9, status="unranked"))
    doc["metrics"]["delta"] = {"latency": {"p50": 200.0}}
    scored = weighted_scores(doc, {"arena": 0.5, "latency": 0.5})
    delta = per_axis(scored, "delta")
    assert delta["arena"]["value"] is None  # unranked: never scored on the arena axis
    assert delta["latency"]["weight"] == pytest.approx(1.0)  # renormalized to its data
    assert next(e for e in scored if e["provider"] == "delta")["weighted_score"] is not None


# ---- validation ------------------------------------------------------------------------------

def test_unknown_axis_rejected_listing_known_axes():
    with pytest.raises(ValueError) as exc:
        weighted_scores(make_doc(), {"vibes": 1.0})
    msg = str(exc.value)
    assert "vibes" in msg
    for axis in AXES:
        assert axis in msg


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        weighted_scores(make_doc(), {"latency": -0.5, "freshness": 1.5})


# ---- determinism -----------------------------------------------------------------------------

def test_deterministic_and_pure():
    doc = make_doc()
    snapshot = copy.deepcopy(doc)
    a = weighted_scores(doc, {"latency": 0.5, "freshness": 0.5})
    b = weighted_scores(doc, {"latency": 0.5, "freshness": 0.5})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert doc == snapshot  # never mutates the input document


def test_ties_break_by_provider_name():
    doc = make_doc()
    for p in doc["metrics"]:
        doc["metrics"][p] = {"accuracy": {"rate": 1.0}}  # everyone identical
    scored = weighted_scores(doc, {"accuracy": 1.0})
    assert [e["provider"] for e in scored] == ["alpha", "beta", "gamma"]


# ---- CLI: python -m arena.rerank -------------------------------------------------------------

def _write_doc(tmp_path, doc):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_cli_prints_unweighted_weighted_and_effective_weights(tmp_path, capsys):
    doc = make_doc()
    path = _write_doc(tmp_path, doc)
    rc = main([path, "--weights", "latency=0.4,freshness=0.4,cost=0.2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "UNWEIGHTED RANKING" in out                       # (a) original ranking, always
    assert "#1 alpha" in out and "win-rate 0.80" in out
    assert "WEIGHTED (your priorities)" in out               # (b) weighted view
    assert "latency 0.50 · freshness 0.50" in out            # (c) renormalized weights
    assert "dropped (no data this run" in out and "cost" in out
    assert "×" in out                                        # per-axis contribution breakdown


def test_cli_never_mutates_input_file(tmp_path):
    path = _write_doc(tmp_path, make_doc())
    before = open(path).read()
    assert main([path, "--weights", "latency=1"]) == 0
    assert open(path).read() == before


def test_cli_falls_back_to_config_weights(tmp_path, capsys):
    doc = make_doc()
    doc["config"] = {"weights": {"latency": 1.0}}
    rc = main([_write_doc(tmp_path, doc)])
    assert rc == 0
    assert "latency 1.00" in capsys.readouterr().out


def test_cli_clear_message_when_no_weights_anywhere(tmp_path, capsys):
    rc = main([_write_doc(tmp_path, make_doc())])
    out = capsys.readouterr().out
    assert rc == 2
    assert "--weights" in out and "arena" in out  # names the known axes


def test_cli_rejects_malformed_and_unknown_weights(tmp_path, capsys):
    path = _write_doc(tmp_path, make_doc())
    assert main([path, "--weights", "latency=fast"]) == 2
    assert main([path, "--weights", "vibes=1.0"]) == 2
    assert "vibes" in capsys.readouterr().err


def test_cli_on_example_run_results(capsys):
    rc = main([EXAMPLE_RESULTS, "--weights", "latency=0.5,accuracy=0.3,cost=0.2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WEIGHTED (your priorities)" in out
    assert "dropped (no data this run" in out and "cost" in out  # example run has no cost data


# ---- dashboard (render_cli_summary) integration ----------------------------------------------

def _example_doc():
    with open(EXAMPLE_RESULTS) as f:
        return json.load(f)


def test_dashboard_has_no_weighted_section_without_weights():
    doc = _example_doc()
    assert doc.get("weights_effective") == {}
    assert "WEIGHTED (your priorities)" not in render_cli_summary(doc)


def test_dashboard_weighted_section_with_real_data():
    doc = _example_doc()
    doc["weights_effective"] = {"latency": 0.5, "freshness": 0.3, "accuracy": 0.2}
    out = render_cli_summary(doc)
    assert "RANKING" in out                                   # unweighted table stays primary
    assert "WEIGHTED (your priorities)" in out
    assert "weights: accuracy 0.20 · latency 0.50 · freshness 0.30" in out
    # re-ranked order matches the engine on the same doc
    expected = [e["provider"] for e in weighted_scores(doc, doc["weights_effective"])
                if e["weighted_score"] is not None]
    weighted_part = out.split("WEIGHTED (your priorities)")[1].split("BY CATEGORY")[0]
    rendered = [line.split()[1] for line in weighted_part.splitlines()
                if line.strip().startswith("#")]
    assert rendered == expected
    # exa has the best latency + freshness in the example run: it should lead the weighted view
    assert rendered[0] == "exa"


def test_dashboard_weighted_section_survives_bad_weights():
    doc = _example_doc()
    doc["weights_effective"] = {"vibes": 1.0}
    out = render_cli_summary(doc)
    assert "weighted view unavailable" in out


def test_cli_bad_file_and_bad_json_exit_cleanly(tmp_path, capsys):
    from arena.rerank import main
    assert main([str(tmp_path / "missing.json"), "--weights", "accuracy=1"]) == 2
    assert "cannot read" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main([str(bad), "--weights", "accuracy=1"]) == 2
    assert "cannot read" in capsys.readouterr().err
