"""Correctness grading against a gold answer, for calibration (§6.5).

Prefers the reused SimpleQA grader (OpenAI gpt-4.1) because grading with a *different* model
family than the Claude judge keeps calibration independent. Falls back to a Claude grader when
no OpenAI key is present (less independent — the caller discloses this).
"""

import logging
import os
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _GradeResult(BaseModel):
    correct: bool


_GRADER_SYSTEM = (
    "You grade whether a predicted answer correctly answers a question, given the gold answer. "
    "Correct = the prediction contains the gold answer's key fact with no contradiction. Ignore "
    "phrasing, extra detail, and hedging as long as the gold fact is present and not contradicted."
)


def grader_kind() -> str:
    """Which grader will be used: 'openai' (independent) or 'claude' (fallback)."""
    return "openai" if os.environ.get("OPENAI_API_KEY") else "claude"


def grade_answer(question: str, answer: Optional[str], gold: Optional[str], llm=None) -> Optional[bool]:
    """True/False if the answer matches gold, or None if it can't be graded."""
    if not answer or not gold:
        return None
    if os.environ.get("OPENAI_API_KEY"):
        r = _grade_openai(question, answer, gold)
        if r is not None:
            return r
    if llm is not None:
        user = (f"Question: {question}\nGold answer: {gold}\nPredicted answer: {answer}\n"
                "Is the predicted answer correct?")
        v = llm.structured(_GRADER_SYSTEM, user, _GradeResult, max_tokens=100)
        return v.correct if v else None
    return None


def _grade_openai(question: str, answer: str, gold: str) -> Optional[bool]:
    """Reuse the base repo's SimpleQA grader (async; run in a fresh loop)."""
    try:
        import asyncio
        from evaluators.correctness_evaluator import CorrectnessConfig, CorrectnessEvaluator
        evaluator = CorrectnessEvaluator(CorrectnessConfig())
        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(
                evaluator.evaluate({"question": question}, {"answer": answer}, {"answer": gold}))
        finally:
            loop.close()
        return res["score"] == 1.0
    except Exception as e:
        logger.warning(f"OpenAI grading unavailable: {e.__class__.__name__}")
        return None


def pair_agreement(x: str, y: str, correct: dict, winner: Optional[str]) -> Optional[bool]:
    """Whether the judge's verdict agrees with gold for one pair.

    Returns None when the pair is not decidable against gold (both same correctness, missing
    grade, or the judge abstained/tied); otherwise True if the judge picked the correct one.
    """
    if x not in correct or y not in correct or correct[x] == correct[y]:
        return None
    if winner not in (x, y):  # tie or excluded -> judge abstained
        return None
    correct_provider = x if correct[x] else y
    return winner == correct_provider
