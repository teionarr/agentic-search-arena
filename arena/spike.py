"""Step-0 vertical spike — settle the two empirical bets before building out further.

Runs the real in-scope providers -> reader -> order-swapped judge -> win-rate on a small
SimpleQA sample and prints the kill-criteria signals:
  (1) rank vs evidence-coverage correlation (is verbosity still dominant after the cap?)
  (2) do >=3 providers separate (non-overlapping CIs)?
  (3) comparison-graph balance (validates win-rate over Bradley-Terry)
It also dumps one real raw payload per provider to tests/arena/fixtures/ so the adapter
tests can be frozen against real shapes ("capture-real-then-freeze").

Requires provider keys + ANTHROPIC_API_KEY in .env, and the base repo's full requirements
installed (importing handlers pulls the base deps). Usage:

    python -m arena.spike --n 30
"""

import argparse
import csv
import json
import logging
import os

from arena import secrets
from arena.adapters.registry import REGISTRY
from arena.config import Query
from arena.llm import DEFAULT_MODEL, LLMClient
from arena.pipeline import run_arena
from arena.report import build_document, render_cli_summary
from arena.scope import resolve_scope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FIXTURE_DIR = "tests/arena/fixtures"


def load_simpleqa(n: int) -> list:
    """Map the base's SimpleQA csv (metadata,problem,answer) into arena queries."""
    rows = []
    with open("datasets/simple_qa_test_set.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(Query(query=row["problem"], expected_answer=row.get("answer")))
            if len(rows) >= n:
                break
    return rows


def spearman(rank_by_provider: dict, coverage_by_provider: dict) -> float:
    """Spearman rho between arena rank and coverage (both as rankings). numpy-only."""
    import numpy as np
    provs = [p for p in rank_by_provider if coverage_by_provider.get(p) is not None]
    if len(provs) < 2:
        return float("nan")
    ranks = np.array([rank_by_provider[p] for p in provs], dtype=float)
    covs = np.array([coverage_by_provider[p] for p in provs], dtype=float)
    cov_rank = covs.argsort().argsort().astype(float)
    a = ranks - ranks.mean()
    b = cov_rank - cov_rank.mean()
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
    return float((a * b).sum() / denom) if denom else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="search-arena Step-0 spike")
    parser.add_argument("--n", type=int, default=30, help="Number of SimpleQA queries")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None, help="Optional arena config (honors provider enabled:false)")
    args = parser.parse_args()

    secrets.load_secrets()
    from arena.config import load_config, resolve_config_path
    config = load_config(resolve_config_path(args.config))
    scope = resolve_scope(config.providers)
    included = scope.included
    logger.info(f"Included providers: {included}")
    if len(included) < 2:
        logger.error("Spike needs >=2 providers with keys present.")
        return 1

    adapters = []
    for name in included:
        spec = REGISTRY[name]
        override = (config.providers.get(name, {}) or {}).get("config", {})
        try:
            adapters.append(spec.build(name, {**spec.default_config, **(override or {})}))
        except Exception as e:
            logger.error(f"[{name}] failed to initialize: {e}")
    if not adapters:
        logger.error("All included providers failed to initialize.")
        return 1

    model_id = args.model or DEFAULT_MODEL
    llm = LLMClient(model=model_id)
    queries = load_simpleqa(args.n)

    result = run_arena(config, queries, adapters, scope, llm, llm, grader_llm=llm)
    doc = build_document(result, [q.query for q in queries],
                         config_snapshot={"model_id": model_id}, model_id=model_id)
    print(render_cli_summary(doc))

    # Kill-criteria signals.
    ranked = [s for s in doc["ranking"] if s["status"] == "ranked"]
    rank_by = {s["provider"]: s["rank"] for s in ranked}
    cov_by = {p: doc["metrics"][p]["coverage"]["avg_tokens_per_result"] for p in rank_by}
    rho = spearman(rank_by, cov_by)
    # Real separation = the ranked providers fall into more than one tie group.
    separated = len(doc["tie_groups"]) > 1
    balance = {p: doc["metrics"][p]["cells_succeeded"] for p in result["providers"]}

    print("\n===== KILL-CRITERIA SIGNALS =====")
    print(f"(1) Spearman(rank, coverage) = {rho:.3f}  "
          f"(|rho| high after the cap => verbosity still dominates)")
    print(f"(2) >=3 providers ranked: {len(ranked) >= 3}; some pair separates (non-overlap CI): {separated}")
    print(f"(3) comparison-graph balance (successful cells/provider): {balance}")
    print(f"    swap-consistency: {result['judge']['swap_consistency']}")
    print("=================================\n")

    _dump_fixtures(adapters, queries[0].query)
    return 0


def _dump_fixtures(adapters, query: str) -> None:
    """Freeze one real raw payload per provider for the adapter tests."""
    import asyncio
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    for a in adapters:
        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(a.search(query))
        finally:
            loop.close()
        path = os.path.join(FIXTURE_DIR, f"{a.name}_raw.json")
        with open(path, "w") as f:
            json.dump(res.raw, f, indent=2, default=str)
        logger.info(f"Froze real payload: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
