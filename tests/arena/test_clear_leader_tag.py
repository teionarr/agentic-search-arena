"""'clear leader' tag: only for a singleton tie group at rank 1 (a real run labeled a
last-place provider 'clear leader' because it was alone in its group)."""

from arena.report import render_cli_summary


def _doc():
    return {
        "timestamp": "t", "model_id": "m", "n_queries": 5, "cost_usd": None,
        "degenerate_run": False, "n_decided_comparisons": 10, "n_excluded_comparisons": 0,
        "judge": {}, "scope": {}, "stage_status": {},
        "ranking": [
            {"provider": "winner", "win_rate": 0.9, "ci_low": 0.85, "ci_high": 0.95,
             "n_comparisons": 10, "status": "ranked", "rank": 1, "tie_group": 0},
            {"provider": "mid_a", "win_rate": 0.5, "ci_low": 0.4, "ci_high": 0.6,
             "n_comparisons": 10, "status": "ranked", "rank": 2, "tie_group": 1},
            {"provider": "mid_b", "win_rate": 0.48, "ci_low": 0.38, "ci_high": 0.58,
             "n_comparisons": 10, "status": "ranked", "rank": 3, "tie_group": 1},
            {"provider": "loser", "win_rate": 0.05, "ci_low": 0.01, "ci_high": 0.09,
             "n_comparisons": 10, "status": "ranked", "rank": 4, "tie_group": 2},
        ],
        "metrics": {p: {"coverage": {}} for p in ("winner", "mid_a", "mid_b", "loser")},
    }


def test_only_rank_one_singleton_gets_clear_leader():
    lines = render_cli_summary(_doc()).splitlines()
    winner_line = next(ln for ln in lines if "winner" in ln)
    loser_line = next(ln for ln in lines if "loser" in ln)
    assert "clear leader" in winner_line
    assert "clear leader" not in loser_line       # separated at the BOTTOM is not a leader
    assert "tied" in next(ln for ln in lines if "mid_a" in ln)


def test_no_clear_leader_when_rank_one_is_tied():
    doc = _doc()
    # Put winner into the same tie group as the mids -> no singleton at rank 1.
    doc["ranking"][0]["tie_group"] = 1
    assert "clear leader" not in render_cli_summary(doc)
