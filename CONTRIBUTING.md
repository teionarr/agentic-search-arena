# Contributing

The most valuable contribution is a **provider adapter** (one file + one registry line) or a
**query set / test case**. The bar for a first PR is deliberately low — no plugin system, no
entry points, no scaffolding.

## Setup

```sh
git clone <your fork>
cd tavily-search-evals
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/arena -q        # Tier A: fast, no network, no keys — must be 100% green
```

## Adding a provider (the only extension point)

1. **Handler** — a small class with `async search(query)` that calls the provider's API
   (pattern: `arena/adapters/firecrawl_handler.py`, ~50 lines). If the base repo already has a
   handler for your provider under `handlers/`, reuse it via a factory instead.

2. **Normalizer** — a function in `arena/adapters/normalize.py` mapping the raw response to
   `EvidenceDoc(url, title, content, published_date?)` items, plus its entry in the
   `NORMALIZERS` dict. Everything downstream sees only the unified shape:

   ```
   {answer: str|None, results: [EvidenceDoc...], latency_ms: float,
    cost_units: float|None, raw: <native payload>}
   ```

3. **Registry entry** — one `ProviderSpec` in `arena/adapters/registry.py` (required env
   key(s), default config, factory). Use the provider's **vendor-documented production
   settings** as the default config and cite the doc page in your PR.

4. **Fixture test** — add a canned raw payload under `tests/arena/fixtures/` and a test that
   your normalizer maps it to the unified shape (see `test_normalize_adapter.py` for the
   pattern). No live calls in CI.

That's it. Do **not** touch `arena/reader.py`, `arena/judge.py`, or `arena/aggregate.py` in a
provider PR — provider identity must never reach the scoring path (see GOVERNANCE.md; the
symmetry test will fail your PR if it does).

## Ground rules (from the requirements, enforced in review)

- **Tier A green to merge**: `python -m pytest tests/arena -q` — deterministic, no AI, no keys.
- **Additive by default**: this repo is a standalone fork (no upstream PRs are planned).
  Still, avoid modifying or reformatting the inherited base files (`handlers/`,
  `run_evaluation.py`, `utils/`) without need — additive changes (new adapters, evaluators,
  output files) keep diffs reviewable and history clean.
- **Never fabricate a metric**: a missing date, price, unit count, or gold answer produces a
  blank (plus a coverage note), never an estimate. Absent metrics drop their weight and the
  rest renormalize.
- **No provider-specific scoring branches**, bonuses, or special cases — anywhere.
- New behavior ships with its own Tier-A tests, matching the style in `tests/arena/`.

## Contributing queries / datasets

Public benchmark sets connect through thin loaders (`arena/benchmark.py`), not vendored
services — check the dataset's license before vendoring data files. Domain query sets
(CSV/JSONL with `query`, optional `expected_answer`/`category`/`freshness_need`) are welcome
as examples under `datasets/`.

## Reporting issues with a ranking

Attach the run's `results.json` (and `traces/` if you ran with `--save-traces`). Every verdict
carries its rationale and every run snapshots its config + query-set hash + manifest, so a
disputed result is reproducible by anyone. See GOVERNANCE.md for the dispute process.
