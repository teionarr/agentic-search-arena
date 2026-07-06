"""Drift report: diff two results.json documents (pure fixtures, no network)."""

import json

from arena.drift import diff_runs, render_drift


def _doc(timestamp, model_id, qhash, ranking):
    return {"timestamp": timestamp, "model_id": model_id, "query_set_hash": qhash,
            "ranking": ranking}


def _score(provider, rank, wr, lo, hi, status="ranked"):
    return {"provider": provider, "rank": rank, "win_rate": wr, "ci_low": lo, "ci_high": hi,
            "status": status, "n_comparisons": 10, "tie_group": 0}


BEFORE = _doc("2026-06-01T00:00:00", "judge-m", "H1", [
    _score("tavily", 1, 0.70, 0.60, 0.80),
    _score("brave", 2, 0.30, 0.20, 0.40),
])
AFTER_SWAPPED = _doc("2026-07-01T00:00:00", "judge-m", "H1", [
    _score("brave", 1, 0.65, 0.55, 0.75),
    _score("tavily", 2, 0.35, 0.25, 0.45),
])


def test_rank_change_and_deltas():
    d = diff_runs(BEFORE, AFTER_SWAPPED)
    t = d["providers"]["tavily"]
    assert t["rank_before"] == 1 and t["rank_after"] == 2 and t["rank_delta"] == -1
    assert t["win_rate_delta"] == -0.35
    assert set(d["rank_changes"]) == {"tavily", "brave"}


def test_shift_beyond_ci_flagged_only_when_cis_disjoint():
    d = diff_runs(BEFORE, AFTER_SWAPPED)
    assert d["providers"]["tavily"]["shifted_beyond_ci"] is True  # [.60,.80] vs [.25,.45]
    # Overlapping CIs -> not significant.
    after_noise = _doc("t", "judge-m", "H1", [
        _score("tavily", 1, 0.65, 0.55, 0.75), _score("brave", 2, 0.35, 0.25, 0.45)])
    d2 = diff_runs(BEFORE, after_noise)
    assert d2["providers"]["tavily"]["shifted_beyond_ci"] is False
    assert d2["rank_changes"] == []


def test_added_and_removed_providers():
    after = _doc("t", "judge-m", "H1", [
        _score("tavily", 1, 0.7, 0.6, 0.8), _score("youcom", 2, 0.3, 0.2, 0.4)])
    d = diff_runs(BEFORE, after)
    assert d["added"] == ["youcom"] and d["removed"] == ["brave"]
    assert "brave" not in d["providers"]  # only common providers get deltas


def test_unranked_side_has_no_delta_or_flag():
    after = _doc("t", "judge-m", "H1", [
        _score("tavily", 1, 0.7, 0.6, 0.8),
        {"provider": "brave", "rank": None, "win_rate": None, "ci_low": None, "ci_high": None,
         "status": "unranked", "n_comparisons": 1, "tie_group": None}])
    d = diff_runs(BEFORE, after)
    b = d["providers"]["brave"]
    assert b["rank_delta"] is None and b["win_rate_delta"] is None
    assert b["shifted_beyond_ci"] is None
    assert b["status_after"] == "unranked"


def test_comparability_warnings_in_render():
    after = _doc("t", "other-judge", "H2", [_score("tavily", 1, 0.7, 0.6, 0.8)])
    text = render_drift(diff_runs(BEFORE, after))
    assert "different query sets" in text
    assert "different judge models" in text


def test_render_happy_path():
    text = render_drift(diff_runs(BEFORE, AFTER_SWAPPED))
    assert "SEARCH ARENA DRIFT" in text
    assert "beyond CI overlap" in text
    assert "different query sets" not in text


def test_cli_main_writes_json(tmp_path, monkeypatch, capsys):
    before_p, after_p, out_p = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "d.json"
    before_p.write_text(json.dumps(BEFORE))
    after_p.write_text(json.dumps(AFTER_SWAPPED))
    import compare_runs
    monkeypatch.setattr("sys.argv", ["compare_runs.py", str(before_p), str(after_p),
                                     "--json", str(out_p)])
    assert compare_runs.main() == 0
    assert "SEARCH ARENA DRIFT" in capsys.readouterr().out
    assert json.loads(out_p.read_text())["rank_changes"]
