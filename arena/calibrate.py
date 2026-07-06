"""Calibration against gold (§6.5) — the headline credibility number.

Answers the question a skeptic asks: "on data where you know the truth, how often does the
reference-free pairwise judge agree with ground truth?" On gold-bearing queries we grade each
provider's synthesized answer as correct/incorrect, then — for every pair where gold makes one
answer right and the other wrong — check whether the judge picked the correct one.

    python -m arena.calibrate --n 50

Requires ANTHROPIC_API_KEY (judge/reader) + provider keys; OPENAI_API_KEY optional but
recommended (independent grader). Falls back to a Claude grader with a disclosed caveat.
"""

import argparse
import csv
import logging
from typing import Callable, List, Optional

from arena import secrets
from arena.adapters.registry import REGISTRY
from arena.config import Query, load_config, resolve_config_path
from arena.grade import grader_kind
from arena.llm import DEFAULT_MODEL, LLMClient
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry, resolve_scope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CALIBRATION_BAR = 0.80  # §6.5 / Tier B threshold


def load_simpleqa_gold(n: int) -> List[Query]:
    rows = []
    with open("datasets/simple_qa_test_set.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(Query(query=row["problem"], expected_answer=row.get("answer")))
            if len(rows) >= n:
                break
    return rows


def run_calibration(queries, adapters, reader_llm, judge_llm, grader_llm, config,
                    search_gatherer: Optional[Callable] = None) -> dict:
    """Judge-vs-gold agreement, computed as a byproduct of a normal (parallel) arena run.

    Delegates to ``run_arena`` so it shares the concurrent reader/grade/judge phases — no
    separate sequential loop. ``search_gatherer`` can be injected in tests."""
    scope = Scope(entries=[ScopeEntry(a.name, INCLUDED) for a in adapters])
    result = run_arena(config, queries, adapters, scope, reader_llm, judge_llm,
                       grader_llm=grader_llm, search_gatherer=search_gatherer)
    cal = result["calibration"]
    graded = sum((m.get("accuracy", {}) or {}).get("total", 0) for m in result["metrics"].values())
    return {
        "agreement": cal["agreement"],
        "n_decidable_pairs": cal["n_decidable"],
        "n_judge_abstained": cal["n_abstained"],
        "n_graded_answers": graded,
        "grader": grader_kind(),
        # Surfaced for benchmark-suite mode (§7): the neutral-rerun accuracy column and the
        # public-benchmark ranking. Additive — existing callers ignore these keys.
        "metrics": result["metrics"],
        "ranking": result["ranking"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="search-arena calibration vs gold (SimpleQA)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    secrets.load_secrets()
    config = load_config(resolve_config_path(args.config))  # honors configs/arena.yaml by default
    scope = resolve_scope(config.providers)
    if len(scope.included) < 2:
        logger.error("Calibration needs >=2 providers enabled with keys present.")
        return 1

    adapters = []
    for n in scope.included:
        spec = REGISTRY[n]
        override = (config.providers.get(n, {}) or {}).get("config", {})
        try:
            adapters.append(spec.build(n, {**spec.default_config, **(override or {})}))
        except Exception as e:
            logger.error(f"[{n}] failed to initialize: {e}")
    if not adapters:
        logger.error("All included providers failed to initialize.")
        return 1
    model_id = args.model or DEFAULT_MODEL
    judge_llm = LLMClient(model=model_id)
    reader_llm = LLMClient(model=model_id)
    grader_llm = LLMClient(model=model_id)  # used only if OPENAI_API_KEY is absent
    queries = load_simpleqa_gold(args.n)

    kind = grader_kind()
    logger.info(f"Calibrating judge={model_id} on {len(queries)} SimpleQA queries "
                f"across {scope.included}; grader={kind}")
    r = run_calibration(queries, adapters, reader_llm, judge_llm, grader_llm, config)

    print("\n===== JUDGE CALIBRATION vs GOLD (SimpleQA) =====")
    if r["agreement"] is None:
        print("  Not enough gold-decidable pairs to estimate agreement.")
    else:
        status = "PASS" if r["agreement"] >= CALIBRATION_BAR else "BELOW BAR"
        print(f"  Judge-vs-gold agreement: {r['agreement']:.2%}  (bar {CALIBRATION_BAR:.0%} -> {status})")
    print(f"  Decidable pairs: {r['n_decidable_pairs']}  ·  graded answers: {r['n_graded_answers']}")
    print(f"  Grader: {r['grader']}" + ("  (independent, different model family)" if r["grader"] == "openai"
                                        else "  (Claude fallback — same family as judge; less independent)"))
    cost = judge_llm.cost_usd() + reader_llm.cost_usd() + (grader_llm.cost_usd() if kind == "claude" else 0)
    print(f"  LLM cost: ${cost:.2f}")
    print("================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
