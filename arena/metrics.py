"""Latency, evidence-coverage, and the optional accuracy anchor.

Cost and freshness are M1. Latency uses only successful timings (None/missing excluded, not
coerced to 0). Coverage = avg tokens/result per provider (reused base token utils). Accuracy
is populated only where a queries row has ``expected_answer`` AND ``OPENAI_API_KEY`` is
present (the reused SimpleQA grader); blank otherwise, never fabricated.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def latency_percentiles(latencies_ms: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """p50/p95 over successful calls only. None/missing are excluded, not zeroed."""
    vals = [x for x in latencies_ms if x is not None]
    if not vals:
        return {"p50": None, "p95": None, "n": 0}
    arr = np.array(vals, dtype=float)
    return {"p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "n": len(vals)}


def evidence_coverage(token_counts: List[int]) -> Dict[str, Optional[float]]:
    """Average tokens/result across a provider's returned documents."""
    if not token_counts:
        return {"avg_tokens_per_result": None, "n_results": 0}
    return {"avg_tokens_per_result": float(np.mean(token_counts)),
            "n_results": len(token_counts)}


def renormalize_weights(weights: Dict[str, float], present_metrics: List[str]) -> Dict[str, float]:
    """Drop absent metrics and renormalize the rest so they sum to 1 (§8)."""
    kept = {k: v for k, v in weights.items() if k in present_metrics and v is not None}
    total = sum(kept.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in kept.items()}


def grade_accuracy(query: str, predicted: str, expected: str) -> Optional[bool]:
    """Optional accuracy anchor via the reused SimpleQA grader.

    Returns True/False, or None if OpenAI is unavailable / the grader can't run. Requires
    ``OPENAI_API_KEY`` and the base's ``langchain_openai`` dependency; imported lazily so the
    core path never needs OpenAI.
    """
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        import asyncio
        from evaluators.correctness_evaluator import CorrectnessEvaluator, CorrectnessConfig
        evaluator = CorrectnessEvaluator(CorrectnessConfig())
        result = asyncio.get_event_loop().run_until_complete(
            evaluator.evaluate({"question": query}, {"answer": predicted}, {"answer": expected})
        )
        return result["score"] == 1.0
    except Exception as e:
        logger.warning(f"Accuracy grading unavailable: {e.__class__.__name__}")
        return None
