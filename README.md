# 🔍 Agentic Search Picker

> **Stop picking your AI agent's search API from vendors' blog posts. Rank them on *your* data.**

[![Tier A tests](https://github.com/teionarr/agentic-search-arena/actions/workflows/tests.yml/badge.svg)](https://github.com/teionarr/agentic-search-arena/actions/workflows/tests.yml)
[![CodeQL](https://github.com/teionarr/agentic-search-arena/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/teionarr/agentic-search-arena/security/code-scanning)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg)](docs/DEVELOPMENT.md)

Tavily or Exa? Brave or Perplexity? Every vendor's benchmark says *"me."* 🙄

This tool asks **your queries** instead — and hands you a ranked, priced, statistically honest
answer in **~30 min** for **~$10**. A bad pick costs teams 6 figures. A good one costs a coffee. ☕

- ⚡ **12 providers · 1 command · 0 golden answers needed**
- 🏆 Win-rate ranking with **95% confidence intervals** — honest ties, never fake precision
- 🎯 Accuracy vs gold · 💰 $/query **and $/correct answer** · ⏱️ p50/p95 latency · 📰 freshness · 🔌 reliability
- 🕵️ **Fully auditable** — blinded judge, every verdict logged with its reasoning, neutrality enforced by tests (ask your AI to verify)
- 🎛️ **Fully customizable** — your data, your metric weights, any judge you like (`openai:` / `claude:`)
- 🔭 **Fully traceable** — optional [Langfuse](https://langfuse.com) span for every search, synthesis, and verdict

---

## 🏁 The answer you get

One command on your queries CSV → this lands in your terminal:

```text
════════════════════════════════════════════════════════
  SEARCH ARENA                                   your-workload ranking
════════════════════════════════════════════════════════
  40 queries × 2 repeats · judge claude-sonnet-4-6 · cost $11.20
  412/480 comparisons used · judge reliability 0.88 · bradley_terry

  #1 tavily            ████████████████┃██░░░░ 0.81 [0.75–0.86]  acc 95%  cost $0.0160/q ($0.0168/correct)  ← clear leader
  #2 exa               ██████████████┃░░░░░░░░ 0.68 [0.61–0.74]  acc 92%  cost n/a                          · tied
  #3 perplexity_search █████████████┃░░░░░░░░░ 0.64 [0.57–0.71]  acc 90%  cost $0.0050/q ($0.0056/correct)  · tied
  #4 brave             ██████████┃░░░░░░░░░░░░ 0.49 [0.42–0.56]  acc 88%  cost $0.0050/q ($0.0057/correct)
  #5 serper            ███████░░░┃░░░░░░░░░░░░ 0.33 [0.27–0.40]  acc 84%  cost $0.0003/q ($0.0004/correct)  ok 92%

  BY CATEGORY   finance: #1 tavily 0.90 · tech: #1 exa 0.86 · news: #1 perplexity_search 0.88
════════════════════════════════════════════════════════
```

*Illustrative output — your numbers **will** differ. That's the entire point.* 😉
(Want to see a real, committed run first? `python run_arena.py --demo` — 0 keys, 0 spend.)

How to read it in 10 seconds:

- **`· tied`** = the CIs overlap; the tool refuses to invent an order it can't defend
- **`$/correct answer`** = a cheap API that needs 3 tries isn't cheap 💸
- **`ok 92%`** = that provider *errored* on 8% of calls — availability, not quality
- **BY CATEGORY** = "best" flips per task; pick per workload, not per logo

Priorities changed? Re-rank any finished run **offline, $0**:

```sh
python -m arena.rerank results.json --weights accuracy=0.5,latency=0.3,cost=0.2
```

---

## 🚀 Get *your* answer (3 steps)

```sh
git clone https://github.com/teionarr/agentic-search-arena && cd agentic-search-arena
pip install -r requirements.txt           # Python ≥ 3.11
cp .env.example .env                      # add ANTHROPIC_API_KEY + whichever provider keys you have
```

**1️⃣ Warm-up** — bundled queries, cheap reader, **~$1–2**:

```sh
python run_arena.py --queries datasets/example_queries.csv --reader-model claude-haiku-4-5-20251001
```

**2️⃣ The real thing** — your queries, **~$8–15** (a committed reference run measured $15.47 for 20 queries × 2 repeats × 6 providers):

```sh
python run_arena.py --queries my_queries.csv --repeats 2 --save-traces
```

**3️⃣ Decide** — read the dashboard, re-weight offline, escalate if it's close (see below).

**Your queries file** (CSV or JSONL — 1 required column, 3 optional superpowers):

| column | what it unlocks |
|--------|----------------|
| `query` | required — the search query |
| `expected_answer` | 🎯 judge-free accuracy column + judge calibration |
| `category` | 📊 per-category rankings (`finance`, `tech`, …) |
| `freshness_need` | 📰 the freshness column for recency-critical rows |

Aim for **≥3 providers and 30–50 queries**; with less, you'll honestly get `tied`/`unranked`
instead of made-up precision.

---

## 🔌 Providers (12 — every key optional)

| provider | env key | | provider | env key |
|----------|---------|-|----------|---------|
| Tavily | `TAVILY_API_KEY` | | Firecrawl | `FIRECRAWL_API_KEY` |
| Exa | `EXA_API_KEY` | | Linkup | `LINKUP_API_KEY` |
| Brave | `BRAVE_API_KEY` | | You.com | `YOU_API_KEY` |
| Serper (Google) | `SERPER_API_KEY` | | Parallel | `PARALLEL_API_KEY` |
| Perplexity Search | `PERPLEXITY_API_KEY` | | Gemini grounding | `GEMINI_API_KEY` |
| Perplexity Sonar | `PERPLEXITY_API_KEY` | | Claude web search | `ANTHROPIC_API_KEY` |

- `ANTHROPIC_API_KEY` is the only **required** key (it powers the judge + reader).
- No key → that provider is skipped and reported in the scope report. No crashes, no surprises.
- Secrets load from `.env` or [Doppler](https://www.doppler.com/) (auto-detected) — never from CLI args.
- Missing your provider? It's **1 adapter + 1 registry line** → [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

---

## ⚖️ Too close to call? Climb the ladder

Reference-free judging is a proxy, not an oracle — near-equal providers at small n are a
genuine tie, and the tool says so. When the stakes justify more:

1. 🆓 **Free anchors (always on):** consensus labels where ≥3 providers agree + deterministic
   machine-checks. 0 extra cost.
2. 🧑‍⚖️ **You, for 15 minutes:** `python -m arena.arbitrate <run_dir>` serves you *only* the
   10–30 blinded calls that could actually flip the ranking.
3. 🤖 **Your actual task:** point `downstream.command` at your own agent loop — its exit codes
   become the metric. The final word is your workload, not our judge.

---

## 🛡️ Trust, verified (not promised)

- **Judge validity is measured:** 0.91 judge-vs-gold calibration (bar ≥ 0.80), committed in
  [docs/example-run/tier_b.json](docs/example-run/tier_b.json).
- **Allowed to fail red:** the same artifact records swap-consistency 0.83 vs a 0.85 bar —
  flagged, not tuned away. Verdicts that flip on order-swap are *excluded*, never averaged in.
- **Blinded + order-swapped judging:** provider identity is stripped before the judge sees
  anything; a symmetry test fails CI if 2 providers with byte-identical evidence score
  differently.
- **370 offline tests** gate every commit (CI); a thresholded live gate (`python -m arena.tier_b`)
  gates releases. CodeQL + Dependabot watch the repo.

Who runs this, how providers get in, how to dispute a ranking → [GOVERNANCE.md](GOVERNANCE.md).
A fully worked real run with commentary → [docs/example-run/](docs/example-run/).

---

## 🧰 Going further

- 📜 **Marketing-claims ledger** — `--benchmark-suite` re-runs public sets (SimpleQA, FRAMES,
  FreshQA) neutrally and prints each vendor's *published* number next to your re-run, with
  sources and dates ([configs/published_claims.yaml](configs/published_claims.yaml)).
- 📈 **Drift over time** — providers ship weekly; re-run and
  `python compare_runs.py old.json new.json` flags rank moves only beyond CI overlap.
  [`drift.yml`](.github/workflows/drift.yml) automates it (manual dispatch; each run ≈ $8–15).
- 🔭 **Langfuse tracing** — `langfuse.enabled: true` + keys → every query is 1 trace with
  search/synthesize/judge spans in *your* project.
- ⚙️ **Config** — copy [configs/arena.example.yaml](configs/arena.example.yaml) to
  `configs/arena.yaml` for provider toggles, models, weights, judges. 0-config works too.

> 🔒 `results/` contains your query text and fetched web content — treat it as sensitive.
> It's git-ignored by default.

---

## 🙏 Credits

Based on [tavily-ai/tavily-search-evals](https://github.com/tavily-ai/tavily-search-evals)
(the inherited harness lives on in [docs/legacy-benchmarks.md](docs/legacy-benchmarks.md)).
Design draws on [youdotcom-oss/web-search-api-evals](https://github.com/youdotcom-oss/web-search-api-evals),
Chatbot-Arena-style pairwise aggregation, and the SimpleQA / FRAMES / FreshQA datasets
(provenance: [datasets/DATASETS.md](datasets/DATASETS.md)).

[MIT License](LICENSE).
