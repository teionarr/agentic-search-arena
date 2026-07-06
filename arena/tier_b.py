"""Tier B runner (§14): the thresholded, AI-involved eval gate.

One live arena run over a gold sample (SimpleQA) powers all four §14 Tier-B checks:

  1. judge-vs-gold calibration    >= 0.80   (the headline credibility number, §6.5)
  2. judge swap-consistency       >= 0.85   (position-bias noise)
  3. inter-judge agreement kappa  >= 0.60   (only when a secondary judge is configured)
  4. e2e live smoke: well-formed ranking + scope report + rationale log over >=2 providers

Red/green, never vibes: each check prints value vs bar; any FAIL exits non-zero and the
artifact (``tier_b.json``) records thresholds, values, and sample size so a miss is a
flagged issue naming the metric and the sample. Auto-SKIPs (exit 0, saying so) when keys
are absent — Tier B gates a release, not a commit (§14).

    python -m arena.tier_b --n 30
"""

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# §14 starting thresholds — tunable, but every run records the bars it was judged against.
THRESHOLDS = {"calibration": 0.80, "swap_consistency": 0.85, "kappa": 0.60}


def _check(name: str, value, bar: float, note: str = "") -> dict:
    """One thresholded check. A missing signal (None) is a FAIL with a reason — a gate that
    silently passes on absent data is not a gate."""
    if value is None:
        return {"check": name, "status": "fail", "value": None, "bar": bar,
                "note": note or "no signal produced"}
    return {"check": name, "status": "pass" if value >= bar else "fail",
            "value": round(float(value), 4), "bar": bar, "note": note}


def evaluate(result: dict, n_providers: int, secondary_configured: bool,
             thresholds: dict = THRESHOLDS) -> list:
    """Pure §14 Tier-B evaluation over a finished arena result. No AI — unit-testable."""
    checks = []

    cal = result.get("calibration") or {}
    checks.append(_check("judge-vs-gold calibration", cal.get("agreement"),
                         thresholds["calibration"],
                         note=f"{cal.get('n_decidable', 0)} decidable pairs"))

    judge = result.get("judge") or {}
    checks.append(_check("judge swap-consistency", judge.get("swap_consistency"),
                         thresholds["swap_consistency"],
                         note=f"{judge.get('swap_total', 0)} double-judged pairs"))

    if secondary_configured:
        checks.append(_check("inter-judge agreement κ", judge.get("inter_judge_kappa"),
                             thresholds["kappa"]))
    else:
        checks.append({"check": "inter-judge agreement κ", "status": "skip",
                       "value": None, "bar": thresholds["kappa"],
                       "note": "no secondary judge configured (reported when one is)"})

    smoke_ok = (n_providers >= 2 and bool(result.get("ranking"))
                and bool(result.get("scope")) and bool(result.get("rationale_log")))
    checks.append({"check": "e2e live smoke", "status": "pass" if smoke_ok else "fail",
                   "value": None, "bar": None,
                   "note": f"{n_providers} providers; ranking+scope+rationale "
                           f"{'present' if smoke_ok else 'INCOMPLETE'}"})
    return checks


def render(checks: list, n_queries: int = None) -> str:
    W = 64
    out = ["", "═" * W, "  TIER B — thresholded evals (§14)", "═" * W]
    if n_queries is not None:
        out.append(f"  sample: {n_queries} gold queries (SimpleQA)")
    for c in checks:
        mark = {"pass": "✅", "fail": "❌", "skip": "·"}[c["status"]]
        val = f"{c['value']:.2f}" if isinstance(c["value"], float) else "—"
        bar = f" (bar ≥{c['bar']:.2f})" if c["bar"] is not None and c["status"] != "skip" else ""
        note = f"  {c['note']}" if c.get("note") else ""
        out.append(f"  {mark} {c['check']:<28} {val}{bar}{note}")
    verdict = "FAIL" if any(c["status"] == "fail" for c in checks) else "PASS"
    out.append("─" * W)
    out.append(f"  Tier B: {verdict}")
    out.append("═" * W + "\n")
    return "\n".join(out)


def skip_report(reason: str) -> str:
    return (f"\nTIER B: SKIPPED — {reason}\n"
            "Tier B needs real keys (ANTHROPIC_API_KEY + ≥2 provider keys); it gates a "
            "release, not a commit, so this skip exits 0 (§14).\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the §14 Tier-B thresholded evals")
    parser.add_argument("--n", type=int, default=30, help="Gold (SimpleQA) sample size")
    parser.add_argument("--model", default=None, help="Judge model override")
    parser.add_argument("--config", default=None, help="Optional arena config")
    parser.add_argument("--output", default="results/tier_b.json",
                        help="Where to write the check artifact")
    args = parser.parse_args()

    from arena import secrets
    secrets.load_secrets()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(skip_report("ANTHROPIC_API_KEY absent (judge/reader unavailable)"))
        return 0

    from arena.adapters.registry import REGISTRY
    from arena.config import load_config, resolve_config_path
    from arena.llm import DEFAULT_MODEL, LLMClient
    from arena.pipeline import run_arena
    from arena.scope import resolve_scope
    from arena.spike import load_simpleqa

    config = load_config(resolve_config_path(args.config))
    scope = resolve_scope(config.providers)
    if len(scope.included) < 2:
        print(skip_report(f"only {len(scope.included)} provider key(s) present (need ≥2)"))
        return 0

    adapters = []
    for name in scope.included:
        spec = REGISTRY[name]
        override = (config.providers.get(name, {}) or {}).get("config", {})
        try:
            adapters.append(spec.build(name, {**spec.default_config, **(override or {})}))
        except Exception as e:
            logger.error(f"[{name}] failed to initialize: {e}")
    if len(adapters) < 2:
        print(skip_report("fewer than 2 providers initialized"))
        return 0

    model_id = args.model or DEFAULT_MODEL
    judge_llm = LLMClient(model=model_id)
    secondary = LLMClient(model=config.judge_secondary) if config.judge_secondary else None
    queries = load_simpleqa(args.n)
    logger.info(f"Tier B: {len(queries)} gold queries × {len(adapters)} providers, judge={model_id}")

    result = run_arena(config, queries, adapters, scope, judge_llm, judge_llm,
                       grader_llm=judge_llm, secondary_judge_llm=secondary)

    checks = evaluate(result, n_providers=len(adapters),
                      secondary_configured=secondary is not None)
    print(render(checks, n_queries=len(queries)))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"thresholds": THRESHOLDS, "n_queries": len(queries),
                   "n_providers": len(adapters), "model_id": model_id,
                   "checks": checks}, f, indent=2)
    logger.info(f"Wrote {args.output}")

    return 1 if any(c["status"] == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
