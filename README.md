# **Evaluation Framework for Web Search APIs**

## **Overview**
This repository provides evaluation frameworks for benchmarking web search APIs, combining static benchmarks and dynamic datasets to measure accuracy, relevance, and retrieval performance across different providers.
### Benchmarks:
1. [SimpleQA](https://openai.com/index/introducing-simpleqa/) Benchmark
    -  Runs the full SimpleQA dataset against each provider.
    - Retrieved documents are reformatted for an LLM (we used `gpt-4.1`) to extract a predicted answer.
    - The predicted answer is graded using the official SimpleQA classifier.
    - For providers that return direct answers, it is possible to bypass the LLM step and compare the returned answer directly with the SimpleQA ground truth (though our evaluations used the classifier route).
2. Document Relevance Benchmark 
    - Uses [QuotientAI](https://docs.quotientai.co/data-collection/logs) to assess the relevance of retrieved documents against a given query.
    - Involves generating a dynamic dataset using the open-source [Dynamic Eval Datasets Generator](https://github.com/Eyalbenba/tavily-web-eval-generator).
    - You can use the provided dataset (`datasets/document_relevance_dynamic_test_set.json`) or easily create new datasets on topics of your choice with the above generator.
    - This flexibility allows evaluation on domain-specific or real-time topics, making the benchmark more reflective of production-like tasks than static datasets.

### **Features**
- Comparative evaluation of multiple search providers
- Out-of-the-box support for Tavily, Exa, Brave, Google (SERP via Serper), Perplexity Search, Perplexity, and GPTR
- Easy integration of additional providers (see [this section](#adding-a-new-search-provider-to-the-evaluation))
- Customizable configuration for each provider
- Parallelized, independent evaluation pipelines
- Automatic resume from the last checkpoint in case of errors

---

## **Evaluation Results**

The table below presents evaluation results across various search providers and LLMs on the SimpleQA benchmark. 

| Provider | Accuracy |
|----------|-------|
| Tavily   | 93.3%   |
| Perplexity Search | 85.92% |
| Google (SERP) using SERPER | 82.15% |
| Brave Search | 76.05% |
| Exa Search | 71.24%   |

The table below presents evaluation results across various search providers and LLMs on the Document Relevance benchmark. 

| Provider | Accuracy |
|----------|-------|
| Tavily   | 83.02%   |
| Perplexity Search | 71.2% |
| Google (SERP) using SERPER | 58.11% |
| Brave Search | 56.2% |
| Exa Search | 51.33%   |

NOTE: The `config.json` file contains the search parameters we used to evaluate each provider above. 

---

## **Running Locally**

1. **Clone the repository**:
    ```sh
    git clone https://github.com/tavily-ai/tavily-search-evals
    cd tavily-search-evals
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:  
    Create a `.env` file in the root directory and add the following:
    ```env
    TAVILY_API_KEY=XXX
    OPENAI_API_KEY=XXX
    EXA_API_KEY=XXX
    PERPLEXITY_API_KEY=XXX
    SERPER_API_KEY=XXX
    BRAVE_API_KEY=XXX
    ```

4. **Run**:
```sh
python run_evaluation.py
```

### **Command Line Options**

- `--evaluation_type`: Type of evaluation to run (simpleqa or document_relevance, default: simpleqa)
- `--config`: Path to JSON config file with provider parameters (default: configs/config.json)
- `--start_index`: Starting index for examples (inclusive, default: 0)
- `--end_index`: Ending index for examples (exclusive, default: all examples)
- `--random_sample`: Number of random samples to select (overrides start/end index)
- `--post_process_model`: Model for post-processing for SimpleQA (default: gpt-4.1)
- `--output_dir`: Directory to save results (default: results)
- `--sequential`: Run providers sequentially instead of in parallel
- `--rerun`: Continue evaluation on existing results directory, output_dir must exist
- `--token_model`: Model for token consumption calculation (default: gpt-4.1)
- `--evaluator_model`: Model for correctness evaluation for SimpleQA (default: gpt-4.1)

### **Output**

Evaluation results are saved in the `results/` directory with the following structure:

```
results/
└── {evaluation_type}/                      # Evaluation type folder (simpleqa or document_relevance)
    └── YYYY-MM-DD_HH-MM-SS/               
        ├── summary.csv                     # Overall evaluation summary
        ├── config.json                     # Configuration used for this evaluation
        ├── {provider}_{evaluation_type}_results.csv   # Individual provider results
        └── ...                             # Additional provider result files
```

#### **Example Output:**
```bash
results/
├── simpleqa/
│   └── 2025-01-15_14-30-25/
│       ├── summary.csv
│       ├── config.json
│       ├── tavily_simpleqa_results.csv
│       ├── exa_simpleqa_results.csv
│       ├── serper_simpleqa_results.csv
│       ├── brave_simpleqa_results.csv
│       └── perplexity_search_simpleqa_results.csv
└── document_relevance/
    └── 2025-01-15_15-45-12/
        ├── summary.csv
        ├── config.json
        ├── tavily_document_relevance_results.csv
        ├── exa_document_relevance_results.csv
        └── ...
```

### **Config Example**

Configuration file `config.json` might look like:
```json
{
  "tavily": {
    "search_depth": "advanced",
    "include_raw_content": false,
    "max_results": 10,
  },
  "perplexity_search": {
    "max_results": 10,
    "max_tokens_per_page": 512
  }
}
```
### **Resume Evaluation**

If your evaluation is interrupted, you can continue from where it stopped using the `--rerun` flag (`output_dir` folder must exist with the previous run's partial results):

```sh
python run_evaluation.py --output_dir results/my_evaluation --rerun
```

This will:
1. Load existing results from the specified output directory
2. Skip questions that have already been evaluated
3. Continue with the remaining questions in the dataset
4. Update the summary statistics with all results when complete

---

## **Adding a New Search Provider to the Evaluation**
### Supported Search Providers
The current supported search providers are:
- `tavily`
- `perplexity`
- `perplexity_search`
- `gptr`
- `exa`
- `serper`
- `brave`


You can extend the system to evaluate additional search providers by following these steps:

1. Create a new handler file in the `handlers` directory (e.g., `handlers/new_provider_handler.py`).

2. Add your provider to the handler registry:
- Update `handlers/__init__.py` to import and expose your new handler.
- Update the `get_search_handlers` function in `app.py` and `run_benchmark.py` to include your new provider.

3. Update environment variables, add your provider's API key to the `.env` file:
```
NEW_PROVIDER_API_KEY=your_api_key_here
```

4. Use your provider in evaluation config:
```json
{
  "new_provider": {
    "custom_param1": "value1",
    "custom_param2": "value2"
  }
}
```

Remember to implement appropriate error handling and respect any rate limits or API constraints for your new provider.

---

## **Arena Mode (reference-free provider ranking)**

Arena mode answers a different question than the benchmarks above: **"which search API wins on
*my* workload?"** — with **zero golden answers required.** You bring your own queries and
whichever provider keys you have; the arena runs each query through every enabled provider, a
**fixed reader** synthesizes an answer from each provider's returned evidence, and a **blind,
order-swapped pairwise judge** decides which answer is better supported. Verdicts that flip on
order-swap are excluded as low-confidence; the survivors aggregate into a **win-rate ranking
with confidence intervals**. Judge reliability is **measured, not assumed**, and every decision
is logged.

It is **additive** to this repo — a separate `run_arena.py` entrypoint that reuses the existing
provider handlers. It does not change `run_evaluation.py`, the handlers, or any existing behavior.

### Quickstart

```sh
pip install -r requirements.txt          # adds anthropic + pyyaml (needs Python ≥3.11)

# Put keys in a .env file (or Doppler — auto-detected). All provider keys are optional;
# ANTHROPIC_API_KEY is required (judge + reader). OPENAI_API_KEY is optional (accuracy anchor).
#   ANTHROPIC_API_KEY=...        TAVILY_API_KEY=...   EXA_API_KEY=...   BRAVE_API_KEY=...
#   SERPER_API_KEY=...           PERPLEXITY_API_KEY=...

python run_arena.py --queries my_queries.csv     # CSV or JSONL; required column: query
```

Zero-config: with no config file, the arena runs across every provider whose key is present.
Aim for **≥3 providers and ~30–50 queries** for a statistically meaningful ranking (fewer → the
tool honestly reports `tied` / `unranked` rather than inventing precision).

**Queries file** (`query` required; the rest optional):

```csv
query,expected_answer,category
who won the 2022 world cup final?,Argentina,sports
what is the latest stable python version?,,tech
```

`expected_answer` turns on a judge-free **accuracy** column (and judge-vs-gold **calibration**)
for those rows.

### Providers (roster)

Document-returning providers, each a one-line registry entry: `tavily`, `exa`, `brave`,
`serper`, `perplexity_search`, `firecrawl`, `linkup`. Plus `claude_search` (Anthropic web
search), a **native-answer** provider that also returns its own synthesized answer (see the
native-answer path below). Every key is optional — a provider with no key (or disabled in
config) is skipped and reported, never an error.

### Reading the output

A run prints a CLI dashboard and writes `results.json` + `ranking.csv` under
`results/arena/<timestamp>/`:

- **Ranking** — win-rate bar + 95% CI per provider. Providers whose CIs overlap are shown as a
  **statistical tie** (grouped, not falsely ordered); too few comparisons → `unranked`.
- **acc** — judge-free accuracy vs `expected_answer` (blank where no gold). A sharper signal
  that can separate providers the pairwise judge calls tied.
- **judge-vs-gold agreement** — on gold-decidable pairs, how often the judge picked the
  provably-correct answer. This is the judge's **validity** number (bar: ≥0.80).
- **judge reliability (swap-consistency)** — fraction of pairs where the verdict survived the
  A/B order-swap. Measures position-bias **noise** (bar: ≥0.85); low here does not mean the
  judge is *wrong* (see calibration).
- **cov** — avg tokens/result, surfaced next to rank so the evidence-granularity difference
  between providers is visible, not hidden.
- **scope** + **stage status** — exactly what ran/was skipped and why, and a green/red health
  line per pipeline stage.

- **BY CATEGORY** — when the queries file tags rows with `category`, each slice gets its own
  ranking (same judge, same aggregation). "Best" is workload-dependent; the slices show it.
- **ok N%** — reliability: a provider that *errors* is flagged separately from one that merely
  finds nothing (only deviations from 100% are shown).
- **cost/q (… /correct)** — $/query from the dated pricing map, and where accuracy anchors
  exist, **$ per correct answer** ($/query ÷ accuracy) — a cheap API that needs three tries
  isn't cheap.

Other commands and flags:

```sh
python run_arena.py --queries q.csv --repeats 3      # re-search every query 3× — providers are
                                                     # non-deterministic; reports the per-repeat
                                                     # win-rate spread as the noise floor
python run_arena.py --queries q.csv --save-traces    # persist per-query audit traces (redacted
                                                     # raw payloads + exact reader inputs)
python compare_runs.py old/results.json new/results.json   # drift report between two runs:
                                                     # rank moves flagged only beyond CI overlap
python -m arena.spike --n 30       # quick run on vendored SimpleQA (gold) — prints calibration too
python -m arena.calibrate --n 50   # judge-vs-gold calibration on a larger gold sample
```

Providers ship changes weekly, so treat any ranking as dated. Re-run on a schedule (cron / CI)
and diff with `compare_runs.py`; the report warns when query sets or judge models differ, so
you never compare apples to oranges silently.

### Benchmark-suite mode (§7)

A second mode re-runs **public sets** (SimpleQA, and FRAMES / FreshQA when you vendor the data
file) across every enabled provider under **one identical policy** — one config, one reader, one
judge, one grader. Same machinery as the arena; it just swaps your queries for a public set. Two
payoffs from one run:

- **Calibration report** (§6.5) — judge-vs-gold agreement %: on gold-decidable pairs, how often
  the reference-free pairwise judge agrees with ground truth. This is the judge's headline
  credibility number (bar ≥0.80).
- **Marketing-claims ledger** (§7) — each vendor's *published* benchmark number shown next to the
  neutral re-run under identical conditions, with the **delta and the trace**. It does **not**
  accuse: a gap can be legitimate (different reader/judge/config/run-date; the web drifts), so
  every claim carries an `as_of` date + `source` and every re-run is timestamped. Supply the
  numbers in a user-editable file (see `configs/published_claims.example.yaml`); with none, the
  neutral re-run is still produced.

When both the arena (your queries) and a benchmark ran, an **arena-vs-benchmark cross-signal**
shows your-workload rank next to the public rank — disagreement is the insight the blogs can't
sell, and you believe the public half because you re-ran it neutrally yourself.

Defaults to a **sample** (a few hundred per set); full runs are opt-in via `sample_size`.

```sh
python run_arena.py --queries my_queries.csv --benchmark-suite   # or modes.benchmark_suite in config
```

Writes `benchmark_suite.json` alongside `results.json` in the run dir.

### Definition of done: two test tiers (§14)

- **Tier A** — deterministic, no AI, no keys (`python -m pytest tests/arena -q`). Runs in CI on
  every push/PR (`.github/workflows/tests.yml`); 100% green is the merge gate.
- **Tier B** — thresholded, AI-involved, costs money; gates a **release**, not a commit:

  ```sh
  python -m arena.tier_b --n 30    # needs ANTHROPIC_API_KEY + ≥2 provider keys, else SKIPs
  ```

  One live run over a SimpleQA gold sample checks all four bars — judge-vs-gold calibration
  ≥ 0.80, swap-consistency ≥ 0.85, inter-judge κ ≥ 0.60 (when a secondary judge is
  configured), and the e2e live smoke. Red/green with exit codes; every run writes
  `tier_b.json` recording the thresholds, values, and sample size. A missing signal is a
  FAIL, never a silent pass.

### Config (optional)

Copy `configs/arena.example.yaml` to `configs/arena.yaml` (git-ignored) to override defaults —
disable providers, set the evidence budget, pick models. It is honored automatically by every
command. Use `--reader-model <cheap-model>` to run the reader/grader on a cheaper model than the
judge and cut cost.

### Honest limits (by design)

Reference-free pairwise judging is a **proxy**, not a truth oracle. It is strong where evidence
quality visibly differs and rank + accuracy + calibration triangulate; it **cannot rigorously
separate genuinely near-equal providers** at small sample sizes — there the tool correctly
reports a tie. The escalation ladder (consensus anchors → human adjudication of pivotal ties →
downstream task success) is the path for higher-stakes decisions and is future work (see below).

**Native-answer path & self-preference (§5).** Most providers are ranked on the *web layer*: the
fixed reader synthesizes an answer from each provider's evidence, so provider identity and style
are invisible to the judge. **Claude web search** additionally returns its own synthesized answer
(the native-answer path). Because the default judge is Claude, a Claude-family native answer could
be favored by style — so the judge is always **blinded + order-swapped**, and in native-answer mode
pairs involving a Claude-family provider are routed to a configured secondary judge or, if none is
set, flagged `possible-self-preference`. The secondary judge can be a genuinely different model
family (§5): set `judge.secondary: "openai:<model>"` to route those pairs to an OpenAI judge
(requires `OPENAI_API_KEY`); a bare model id keeps a second Claude judge. The caveat is surfaced
in the run summary and rationale log whenever native mode runs. Reader-synthesized (primary-path) pairs are never flagged.

> **Data note:** `results.json` and the rationale log contain your full query text and the web
> content each provider returned — treat `results/` as sensitive. It is git-ignored by default.

### Optional Langfuse tracing (§11)

Off by default. Set `langfuse.enabled: true` in your config **and** provide Langfuse keys in your
secrets (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`) to trace each query as
one Langfuse trace with `provider.search`, `reader.synthesize`, and `judge.compare` spans. Missing
keys silently disable tracing (no error). Trace data goes to **your** Langfuse project and passes
through the same `redact()` boundary as `results.json`, so no secret value is ever exported.

### Neutrality & governance

Neutrality is an **enforced invariant, not an intention**: provider identity is stripped before
the reader and judge see anything, no provider-specific branch exists in the scoring path, and
the symmetry test fails the build if two providers given byte-identical evidence score
differently. Every run snapshots its config, query-set hash, and environment manifest (harness
commit + package versions) so a third party can re-run it. Funding, affiliations (including
that this fork builds on a repo maintained by Tavily, a ranked provider), the provider
inclusion process, and the dispute process are documented in [GOVERNANCE.md](GOVERNANCE.md).

### Extending

Adding a provider is **one adapter + one registry line** (the only documented extension point):
a normalizer in `arena/adapters/normalize.py` mapping the raw response to
`{url, title, content}`, and an entry in `arena/adapters/registry.py`. No auto-discovery, no
plugin system. Full walkthrough in [CONTRIBUTING.md](CONTRIBUTING.md).

### Roadmap (deferred)

More native-answer providers (Perplexity Sonar, GPT-Researcher); Tier-1 consensus anchors;
Tier-2 human adjudication; Tier-3 downstream success + Langfuse tracing.
(Claude web-search native-answer path + self-preference handling, cost-per-query, freshness
scoring, and benchmark-suite mode are now implemented — see the sections above.)

**Prior art / design credit:** Arena mode extends this repo's provider-handler architecture, and
draws on [`youdotcom-oss/web-search-api-evals`](https://github.com/youdotcom-oss/web-search-api-evals)
(sampler / synthesize / grade design), reference-free LLM-as-judge and Chatbot-Arena-style
pairwise aggregation, and the SimpleQA / FRAMES / FreshQA benchmark datasets.

---

## **License**

This project is made available under the [MIT License](https://github.com/tavily-ai/tavily-mcp/blob/main/LICENCE).
