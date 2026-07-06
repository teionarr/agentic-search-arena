# Search Arena

**Which agentic search API actually wins on *your* workload?**

Different solutions for different tasks — some faster, some more accurate, some cheaper. Vendor
benchmarks won't tell you which one wins on *your* queries. This tool will: bring your own
queries and whichever API keys you have, and get back a win-rate ranking with confidence
intervals, accuracy, latency, cost, and freshness — with zero golden answers required.

- **Fully auditable.** This tool doesn't favor anyone — ask your AI to verify. Neutrality is
  test-enforced, not promised: a symmetry test fails the build if two providers given
  byte-identical evidence score differently, the judge is blinded and order-swapped, and every
  decision writes its rationale to the log. See [GOVERNANCE.md](GOVERNANCE.md).
- **Fully customizable.** Your data, your decisions, any judge you like: bring your own queries
  CSV, set your own axis weights, and swap in a cross-family judge (`openai:<model>`).
- **Fully traceable.** Optional per-query [Langfuse](https://langfuse.com) tracing — every
  search, synthesis, and judgment as a span in your own project.

Three minutes to install, about thirty minutes to run on your data — and you have an answer.
Companies burn hundreds of thousands of dollars living with a bad choice made on marketing
benchmarks.

## See it in 30 seconds

```sh
python run_arena.py --demo
```

No API keys, no spend: the demo renders a real committed run
([docs/example-run/](docs/example-run/) — a 20-query mixed workload × 2 repeats across 6
providers; actual spend $15.47). What it shows, straight from that run:

| # | provider | win-rate | 95% CI | accuracy | reliability |
|---|----------|---------:|--------|---------:|------------|
| 1 | perplexity_search | 0.84 | [0.79, 0.88] | 16/16 | ok |
| 2 | exa | 0.81 | [0.76, 0.85] | 16/16 | ok |
| 3 | brave | 0.62 | [0.55, 0.70] | 16/16 | ok |
| 4 | tavily | 0.57 | [0.50, 0.64] | 16/16 | ok |
| 5 | serper | 0.44 | [0.37, 0.52] | 16/16 | ok |
| 6 | claude_search | 0.03 | [0.01, 0.05] | 4/4 | ok 65% |

perplexity_search (0.84) and exa (0.81) lead, but the CI overlap chain groups #1–#5 as
statistically tied at n=40; only claude_search separates — and its `ok 65%` shows *why*: an
availability problem, visibly distinct from a quality problem. Per-category slices flip the
story: exa takes #1 on sports and tech while perplexity_search leads finance and research.

## Quickstart

```sh
git clone https://github.com/teionarr/agentic-search-arena
cd agentic-search-arena
pip install -r requirements-arena.txt      # Python ≥ 3.11
cp .env.example .env                       # then fill in your keys
```

`ANTHROPIC_API_KEY` is required (judge + reader). Every provider key is optional — providers
without a key are skipped per provider and reported in the scope report (you need at least one
provider key for a real run; with none, the tool tells you exactly what's missing and exits
cleanly). One install note: five
providers (`tavily`, `exa`, `brave`, `serper`, `perplexity_search`) reuse the handlers inherited
from the upstream project, which need the full legacy dependency set — if you want those in your
ranking (you probably do), `pip install -r requirements.txt` instead (a superset; the legacy
harness in [docs/legacy-benchmarks.md](docs/legacy-benchmarks.md) also uses it).

First run, on the bundled example workload with a cheap reader (~$1–2):

```sh
python run_arena.py --queries datasets/example_queries.csv --reader-model claude-haiku-4-5-20251001
```

Then the real thing — your own queries (~$8–15 for a typical workload; the committed example run
measured $15.47 for 20 queries × 2 repeats × 6 providers):

```sh
python run_arena.py --queries my_queries.csv --repeats 2 --save-traces
```

**Queries file** (CSV or JSONL; `query` required, everything else optional):

| column | meaning |
|--------|---------|
| `query` | the search query (required) |
| `expected_answer` | gold answer — unlocks the judge-free accuracy column + judge calibration |
| `category` | tag for per-category rankings (e.g. `finance`, `tech`) |
| `freshness_need` | mark queries where recency matters — unlocks the freshness column |

Aim for ≥3 providers and ~30–50 queries for a statistically meaningful ranking; with less, the
tool honestly reports `tied` / `unranked` rather than inventing precision.

## What you get

A CLI dashboard plus `results.json` + `ranking.csv` under `results/arena/<timestamp>/`:

- **Win-rate ranking with 95% CIs and honest ties** — providers whose CIs overlap are grouped as
  a statistical tie, never falsely ordered; too few comparisons → `unranked`.
- **Accuracy vs gold** — judge-free, wherever rows carry `expected_answer` (plus free
  machine-verified anchors for mechanically checkable answers — no extra key needed).
- **Latency** — p50/p95 per provider.
- **Cost** — $/query from a dated public pricing map, and **$ per correct answer** where
  accuracy exists: a cheap API that needs three tries isn't cheap.
- **Freshness + coverage honesty** — freshness scores carry a date-coverage figure and a
  low-confidence flag instead of pretending sparse data is comparable.
- **Reliability** — a provider that errors is flagged separately from one that finds nothing.
- **Per-category slices** — "best" is workload-dependent; the slices show it.
- **Repeats spread** — `--repeats N` re-searches every query N times and reports the per-repeat
  win-rate spread as the noise floor.

Priorities change; don't re-spend to re-rank. Re-weight any finished run offline:

```sh
python -m arena.rerank results.json --weights accuracy=0.5,latency=0.3,cost=0.2
```

## When the ranking is too close to call

Reference-free pairwise judging is a proxy, not a truth oracle: it cannot rigorously separate
genuinely near-equal providers at small sample sizes — there the tool correctly reports a tie.
For higher-stakes decisions, escalate up the ladder (all three tiers are implemented):

1. **Free anchors (automatic).** Consensus silver labels where ≥3 independent providers converge
   on the same answer, plus deterministic machine-verification of checkable answers. No AI, no
   extra cost — always on.
2. **Human adjudication of pivotal ties.** `python -m arena.arbitrate <run_dir>` finds the
   10–30 pairwise calls that would actually change the ranking and serves them to you blinded;
   your verdicts re-aggregate with the rest.
3. **Downstream task success.** Set `downstream.command` in your config to your own agent loop
   or eval script; the arena runs it per provider and the exit codes become the metric. Your
   *actual task* is the final word.

## Trust, verified

- **Judge validity is measured, not assumed:** 0.91 judge-vs-gold calibration on gold-decidable
  pairs (bar ≥ 0.80), recorded in the committed
  [docs/example-run/tier_b.json](docs/example-run/tier_b.json).
- **Swap-consistency is reported and allowed to fail red** — the same artifact records a 0.83
  against a 0.85 bar, flagged rather than tuned away; verdicts that flip on order-swap are
  excluded from aggregation, not averaged in.
- **Blinded, order-swapped pairwise judging** — provider identity is stripped before the reader
  and judge see anything, and every pair is judged in both orders.
- **Two test tiers:** Tier A — 326 deterministic offline tests, run in CI on every push
  (100% green is the merge gate); Tier B — a thresholded, AI-involved live gate
  (`python -m arena.tier_b`) that gates releases.
- **CodeQL scanning and Dependabot** run on the repository.

Who's behind it, how providers get in, and how to dispute a ranking:
[GOVERNANCE.md](GOVERNANCE.md). A fully worked real run with commentary:
[docs/example-run/](docs/example-run/).

## Going further

- **Benchmark-suite mode** — `python run_arena.py --queries q.csv --benchmark-suite` re-runs
  public sets (SimpleQA, and FRAMES/FreshQA when vendored) across every enabled provider under
  one identical policy, and emits a **marketing-claims ledger**: each vendor's published number
  next to your neutral re-run, with the delta and the trace — real vendor claims ship in
  [configs/published_claims.yaml](configs/published_claims.yaml). No accusations: every claim
  carries an `as_of` date and source, every re-run a timestamp.
- **Drift over time** — providers ship changes weekly, so any ranking is dated. Re-run and diff
  with `python compare_runs.py old/results.json new/results.json` (rank moves flagged only
  beyond CI overlap); `.github/workflows/drift.yml` automates the loop as a manual-dispatch
  workflow (each run spends real credits, roughly $8–15).
- **Langfuse tracing** — set `langfuse.enabled: true` in your config plus Langfuse keys in your
  secrets; each query becomes one trace with `provider.search`, `reader.synthesize`, and
  `judge.compare` spans, in your own Langfuse project.
- **Config reference** — copy [configs/arena.example.yaml](configs/arena.example.yaml) to
  `configs/arena.yaml` (git-ignored) to disable providers, set evidence budgets, pick models,
  weights, and judges. Zero-config works: no file means every keyed provider runs on strong
  defaults.
- **Provider roster (12):** `tavily`, `exa`, `brave`, `serper`, `perplexity_search`,
  `perplexity` (Sonar), `firecrawl`, `linkup`, `claude_search`, `youcom`, `parallel`, `gemini`.
  Adding one is a single adapter + one registry line — see
  [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

> **Data note:** `results.json` and the rationale log contain your full query text and the web
> content each provider returned — treat `results/` as sensitive. It is git-ignored by default.

## Credits

Based on [tavily-ai/tavily-search-evals](https://github.com/tavily-ai/tavily-search-evals) —
the inherited SimpleQA / Document Relevance harness lives on unchanged in
[docs/legacy-benchmarks.md](docs/legacy-benchmarks.md). The arena also draws on
[`youdotcom-oss/web-search-api-evals`](https://github.com/youdotcom-oss/web-search-api-evals)
(sampler / synthesize / grade design), reference-free LLM-as-judge and Chatbot-Arena-style
pairwise aggregation, and the SimpleQA / FRAMES / FreshQA benchmark datasets (provenance in
[datasets/DATASETS.md](datasets/DATASETS.md)).

Licensed under the [MIT License](LICENSE).
