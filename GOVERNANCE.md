# Governance & Independence

A comparison tool is dead the moment it looks captured. This document states who is behind the
arena, how providers get in, how results get published, and what stops any provider — including
the one whose repo this forks — from being favored.

## Affiliations & funding (full disclosure)

- **This repository is a fork of [`tavily-ai/tavily-search-evals`](https://github.com/tavily-ai/tavily-search-evals)**,
  a repo created and maintained by **Tavily — one of the providers being ranked**. The arena
  additions (everything under `arena/`, `run_arena.py`, `compare_runs.py`) are independent,
  additive work; the base handlers and benchmark tables in the upstream README are Tavily's.
- No provider pays for placement, ranking position, default-enablement, or favorable
  configuration. There is no sponsorship, referral, or affiliate relationship behind any
  ranking output.
- The arena's judge/reader run on the Anthropic API by default (configurable). Claude-family
  providers judged by a Claude judge are handled by the self-preference mitigation (§5 of the
  requirements; see README "Native-answer path & self-preference").

## Neutrality is enforced, not promised

The scoring path cannot see provider identity. This is a tested invariant, not a policy:

- **Symmetry test** (`tests/arena/test_pipeline_e2e.py::test_symmetry_identical_evidence_ties`):
  two providers fed byte-identical evidence must receive identical scores and a tie. Any
  difference on identical input is a failing build.
- **Identity isolation**: payloads sent to the reader and judge carry no provider name; the
  judge is blinded and every pair is order-swapped, with flip-on-swap verdicts excluded.
- **No provider-specific scoring branches**: the only place a provider name appears is the
  adapter registry. Reader, judge, and aggregation are provider-agnostic.
- **Documented production configs**: every provider runs its vendor-documented settings with
  `max_results` held constant; the full config is snapshotted into each run's output so anyone
  can audit that nothing was tuned to advantage one provider.
- **Public pricing only**: the cost column prices from `configs/pricing.yaml`, a dated,
  user-editable file of public list prices, with its `as_of` date printed beside every cost.

## How providers are added

Any provider can be added by anyone — one adapter + one registry line (see CONTRIBUTING.md).
Acceptance criteria are mechanical, identical for every provider, and reviewable in the PR:

1. The adapter normalizes to the common `{answer, results[], latency_ms, cost_units, raw}` shape.
2. A fixture test maps a canned raw payload to that shape (no live calls in CI).
3. The provider's default config is its vendor-documented production setting, cited in the PR.
4. No changes to the reader, judge, or aggregation modules.

Removal happens only for a dead API or an unmaintained adapter, recorded in the changelog —
never for ranking poorly.

## How results are published

- The tool publishes **method, not verdicts**: every ranking is produced by whoever runs it, on
  their queries, with their keys. There is no hosted leaderboard to capture.
- Any published example run must include its `results.json` (config snapshot, query-set hash,
  run manifest, per-decision rationale log) so it can be independently re-run and disputed.
- Benchmark-suite re-runs shown next to vendor-published numbers report the **delta and the
  trace, without accusation** — a gap can be legitimate (different reader/judge/config/date).
- Results are dated. Providers change weekly; `compare_runs.py` exists precisely so that stale
  numbers are treated as drift to measure, not truth to defend.

## Disputing a result

Open an issue with the run's `results.json` (and traces if saved). Because every verdict logs
its rationale and every run snapshots its config, a dispute is a concrete re-run, not an
argument. If a dispute exposes a harness bug or a biased prompt, the fix lands as a normal PR
with a regression test.
