"""Tier B runner: pure evaluation logic + skip behavior (§14). No AI, no keys."""

from arena.tier_b import THRESHOLDS, evaluate, render, skip_report


def _result(cal=0.9, n_decidable=12, swap=0.95, kappa=None,
            ranking=True, scope=True, rationale=True):
    return {
        "calibration": {"agreement": cal, "n_decidable": n_decidable, "n_abstained": 0},
        "judge": {"swap_consistency": swap, "swap_total": 40, "inter_judge_kappa": kappa},
        "ranking": [{"provider": "p1"}] if ranking else [],
        "scope": {"p1": {"status": "included"}} if scope else {},
        "rationale_log": [{"query": "q"}] if rationale else [],
    }


def _by_name(checks):
    return {c["check"]: c for c in checks}


def test_all_green_passes():
    checks = evaluate(_result(), n_providers=3, secondary_configured=False)
    by = _by_name(checks)
    assert by["judge-vs-gold calibration"]["status"] == "pass"
    assert by["judge swap-consistency"]["status"] == "pass"
    assert by["inter-judge agreement κ"]["status"] == "skip"   # no secondary -> skip, not fail
    assert by["e2e live smoke"]["status"] == "pass"


def test_calibration_below_bar_fails():
    checks = evaluate(_result(cal=0.79), n_providers=3, secondary_configured=False)
    c = _by_name(checks)["judge-vs-gold calibration"]
    assert c["status"] == "fail" and c["bar"] == THRESHOLDS["calibration"] == 0.80


def test_boundary_is_inclusive():
    checks = evaluate(_result(cal=0.80, swap=0.85), n_providers=2, secondary_configured=False)
    by = _by_name(checks)
    assert by["judge-vs-gold calibration"]["status"] == "pass"
    assert by["judge swap-consistency"]["status"] == "pass"


def test_missing_signal_is_fail_not_silent_pass():
    checks = evaluate(_result(cal=None), n_providers=3, secondary_configured=False)
    c = _by_name(checks)["judge-vs-gold calibration"]
    assert c["status"] == "fail" and c["value"] is None


def test_kappa_checked_only_with_secondary():
    ok = evaluate(_result(kappa=0.7), n_providers=3, secondary_configured=True)
    assert _by_name(ok)["inter-judge agreement κ"]["status"] == "pass"
    low = evaluate(_result(kappa=0.5), n_providers=3, secondary_configured=True)
    assert _by_name(low)["inter-judge agreement κ"]["status"] == "fail"
    none = evaluate(_result(kappa=None), n_providers=3, secondary_configured=True)
    assert _by_name(none)["inter-judge agreement κ"]["status"] == "fail"  # configured but silent


def test_smoke_fails_on_missing_artifacts_or_providers():
    one_prov = evaluate(_result(), n_providers=1, secondary_configured=False)
    assert _by_name(one_prov)["e2e live smoke"]["status"] == "fail"
    no_rationale = evaluate(_result(rationale=False), n_providers=3, secondary_configured=False)
    assert _by_name(no_rationale)["e2e live smoke"]["status"] == "fail"


def test_render_names_metric_and_verdict():
    text = render(evaluate(_result(cal=0.5), n_providers=3, secondary_configured=False),
                  n_queries=30)
    assert "TIER B" in text and "Tier B: FAIL" in text
    assert "judge-vs-gold calibration" in text and "0.50" in text and "≥0.80" in text
    green = render(evaluate(_result(), n_providers=3, secondary_configured=False))
    assert "Tier B: PASS" in green


def test_cli_skips_cleanly_without_anthropic_key(monkeypatch, capsys):
    import sys
    from arena import tier_b
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("arena.secrets.load_secrets", lambda: None)
    monkeypatch.setattr(sys, "argv", ["tier_b"])
    assert tier_b.main() == 0                       # skip is exit 0 — gates a release, not a commit
    assert "SKIPPED" in capsys.readouterr().out


def test_skip_report_mentions_reason():
    assert "no keys at all" in skip_report("no keys at all")
