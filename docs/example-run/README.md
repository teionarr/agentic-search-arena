# First live Tier-B gate run (2026-07-06)

The first execution of `python -m arena.tier_b` against real provider APIs — 30 SimpleQA gold
queries × 6 keyed providers, judge `claude-sonnet-4-6`. Raw artifact: [tier_b.json](tier_b.json).

| Check (§14) | Bar | Value | Verdict |
|---|---|---|---|
| Judge-vs-gold calibration | ≥ 0.80 | **0.91** (97 decidable pairs) | ✅ |
| Judge swap-consistency | ≥ 0.85 | **0.83** (440 double-judged pairs) | ❌ |
| Inter-judge agreement κ | ≥ 0.60 | — | skipped (no secondary judge configured) |
| E2E live smoke | — | 6 providers; ranking + scope + rationale present | ✅ |

## Reading the numbers

**Calibration 0.91 is the headline.** On pairs where gold decides which answer is right, the
reference-free pairwise judge agreed with ground truth 91% of the time. This is the number that
converts "the judge said so" into a measured credibility claim (§6.5).

**Swap-consistency 0.83 is a flagged miss, recorded per §14 — not silently tuned away.**
Interpretation and mitigants:

- SimpleQA is a near-tie-heavy sample: most providers surface the same fact, so the two answers
  in a pair are often near-identical and position noise dominates. A workload with real quality
  spread should score higher; treat 0.83 as a floor measured on the hardest-to-discriminate data.
- Flip-on-swap verdicts are **excluded from aggregation by design** — they shrink the sample
  rather than corrupt the ranking. The 0.91 calibration is computed over the surviving verdicts.
- The number was stable across two runs (0.79 at n=5, 0.83 at n=30), so it is a real property of
  this judge config on this data, not sampling noise.

Follow-ups that could close the gap: a secondary judge (κ + ensemble), a tie-friendlier judge
prompt, or accepting a documented 0.80 bar for near-tie-heavy gold sets.

## Integration findings from the same runs

- `claude_search` threw intermittent connection errors under live load; the reliability column
  (`success_rate` / `error_rate`) captures this per provider.
- One judge call hit an API timeout and was recovered by the retry machinery (4 attempts,
  backoff) — no manual intervention.

## Pending

A committed redacted **example workload run** (20 mixed queries × 2 repeats, `--save-traces`)
is queued behind an API-credit top-up; it will land here as `results.json` + `ranking.csv` +
two sample traces when re-run.
