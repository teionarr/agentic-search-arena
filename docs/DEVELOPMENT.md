# Development guide

Single source for contributors. The most valuable contribution is a **provider adapter** (one
file + one registry line) or a **query set / test case**. The bar for a first PR is deliberately
low — no plugin system, no entry points, no scaffolding.

## Setup

```sh
git clone <your fork>
cd agentic-search-arena
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # full set — covers arena, legacy handlers, and tests
python -m pytest tests/arena -q      # Tier A: fast, no network, no keys — must be 100% green
```

## Architecture map

One straight pipeline, provider-agnostic after the first stage:

```
adapters ──▶ reader ──▶ judge ──▶ aggregate ──▶ report
```

- **Adapters** (`arena/adapters/`) — one handler + one normalizer per provider, wired in
  `arena/adapters/registry.py`. This registry is the **only extension point**. Everything
  downstream sees one unified shape:
  `{answer, results: [EvidenceDoc(url, title, content, published_date?)], latency_ms, cost_units, raw}`.
- **Reader** (`arena/reader.py`) — a fixed model synthesizes an answer from each provider's
  evidence. Provider identity is stripped before this point and never reappears.
- **Judge** (`arena/judge.py`) — blinded, order-swapped pairwise comparison; verdicts that flip
  on swap are excluded. An optional secondary judge (`judge.secondary`, including cross-family
  `openai:<model>`) enables ensembling, inter-judge κ, and self-preference routing for
  native-answer providers.
- **Aggregate** (`arena/aggregate.py`) — Bradley–Terry (default) or win-rate over the
  swap-survived comparisons; CIs, tie groups, per-category slices.
- **Report** (`arena/report.py`) — CLI dashboard + `results.json` + `ranking.csv`, with a config
  snapshot, query-set hash, and environment manifest in every run.

Supporting modules follow the same additive pattern: `anchors.py` (free consensus +
machine-verify anchors), `arbitrate.py` (human adjudication of pivotal ties), `downstream.py`
(your command as the metric), `benchmark.py` (public-set loaders), `tier_b.py` (live release
gate), `drift.py`/`compare_runs.py` (run-to-run diffs), `tracing.py` (optional Langfuse).

## Adding a provider (the only extension point)

1. **Handler** — a small class with `async search(query)` that calls the provider's API
   (pattern: `arena/adapters/firecrawl_handler.py`, ~50 lines). If the base repo already has a
   handler for your provider under `handlers/`, reuse it via a factory instead.

2. **Normalizer** — a function in `arena/adapters/normalize.py` mapping the raw response to
   `EvidenceDoc(url, title, content, published_date?)` items, plus its entry in the
   `NORMALIZERS` dict.

3. **Registry entry** — one `ProviderSpec` in `arena/adapters/registry.py` (required env
   key(s), default config, factory). Use the provider's **vendor-documented production
   settings** as the default config and cite the doc page in your PR.

4. **Fixture test** — add a canned raw payload under `tests/arena/fixtures/` and a test that
   your normalizer maps it to the unified shape (see `test_normalize_adapter.py` for the
   pattern). No live calls in CI.

That's it. Do **not** touch `arena/reader.py`, `arena/judge.py`, or `arena/aggregate.py` in a
provider PR — provider identity must never reach the scoring path (see
[GOVERNANCE.md](../GOVERNANCE.md); the symmetry test will fail your PR if it does).

## The two test tiers

- **Tier A** — deterministic, no AI, no keys, no network:

  ```sh
  python -m pytest tests/arena -q
  ```

  Runs in CI on every push/PR (`.github/workflows/tests.yml`); **100% green is the merge gate.**

- **Tier B** — thresholded, AI-involved, costs money; gates a **release**, not a commit:

  ```sh
  python -m arena.tier_b --n 30    # needs ANTHROPIC_API_KEY + ≥2 provider keys, else SKIPs
  ```

  One live run over a SimpleQA gold sample checks four bars:

  | check | threshold |
  |-------|-----------|
  | judge-vs-gold calibration | ≥ 0.80 |
  | judge swap-consistency | ≥ 0.85 |
  | inter-judge agreement κ (when a secondary judge is configured) | ≥ 0.60 |
  | e2e live smoke | ranking + scope + rationale present |

  Red/green with exit codes; every run writes `tier_b.json` recording the thresholds, values,
  and sample size. A missing signal is a FAIL, never a silent pass. The first real gate run
  (with a flagged miss, kept red) is committed at
  [example-run/tier_b.json](example-run/tier_b.json) with commentary in
  [example-run/README.md](example-run/README.md).

## Datasets

Vendored benchmark data files (SimpleQA, FRAMES, FreshQA) each record source, license,
retrieval date, row count, and transformations in
[../datasets/DATASETS.md](../datasets/DATASETS.md) — check a dataset's license before vendoring
new data files. Public sets connect through thin loaders (`arena/benchmark.py`), not vendored
services. Domain query sets (CSV/JSONL with `query`, optional
`expected_answer`/`category`/`freshness_need`) are welcome as examples under `datasets/`.

## CI & workflows

- **`tests.yml`** — Tier A on every push and PR (Python 3.12). Merge gate.
- **`drift.yml`** — manual-dispatch (`workflow_dispatch`) drift measurement: runs the arena on
  `datasets/example_queries.csv`, diffs against the previous run's `results.json` artifact with
  `compare_runs.py`, and uploads `arena-results` + `drift-report` artifacts. **Every dispatch
  spends real API credits (~$8–15).** The weekly cron is a commented-out block — uncomment
  `schedule:` knowing each run costs money. Secrets go in
  **Settings → Secrets and variables → Actions**: `ANTHROPIC_API_KEY` (required) plus any
  provider keys; providers without a secret are skipped and shown in the scope report. The
  optional `reader_model` input defaults to a Haiku model to keep cost down.
- **`claude-bot.yml`** — posts PR/issue comments as the repo's GitHub App identity.

## Release hygiene

A release (tag) requires a fresh **Tier B** pass — run `python -m arena.tier_b`, commit the
resulting `tier_b.json` alongside the release notes, and treat any red bar as a stop-ship
unless it is explicitly documented the way the committed example run documents its
swap-consistency miss.

## Ground rules (enforced in review)

- **Tier A green to merge** — deterministic, no AI, no keys.
- **Never fabricate a metric** — a missing date, price, unit count, or gold answer produces a
  blank (plus a coverage note), never an estimate. Absent metrics drop their weight and the
  rest renormalize.
- **No provider-specific scoring branches**, bonuses, or special cases — anywhere. The only
  place a provider name appears is the adapter registry.
- **Additive changes preferred** — avoid modifying or reformatting the inherited base files
  (`handlers/`, `run_evaluation.py`, `utils/`) without need; new adapters, evaluators, and
  output files keep diffs reviewable.
- New behavior ships with its own Tier-A tests, matching the style in `tests/arena/`.

## Reporting issues with a ranking

Attach the run's `results.json` (and `traces/` if you ran with `--save-traces`). Every verdict
carries its rationale and every run snapshots its config + query-set hash + manifest, so a
disputed result is reproducible by anyone. See [GOVERNANCE.md](../GOVERNANCE.md) for the
dispute process.
