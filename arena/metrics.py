"""Latency, evidence-coverage, and the optional accuracy anchor.

Cost and freshness are M1. Latency uses only successful timings (None/missing excluded, not
coerced to 0). Coverage = avg tokens/result per provider (reused base token utils). Accuracy
is populated only where a queries row has ``expected_answer`` AND ``OPENAI_API_KEY`` is
present (the reused SimpleQA grader); blank otherwise, never fabricated.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default freshness window (days) when a query's freshness_need gives no explicit span (§8.3).
DEFAULT_FRESHNESS_WINDOW_DAYS = 30

# freshness_need shorthands -> window in days. A bare integer in freshness_need is read as days.
_FRESHNESS_WINDOW_ALIASES = {
    "day": 1, "daily": 1, "24h": 1,
    "week": 7, "weekly": 7,
    "month": 30, "monthly": 30,
    "quarter": 90,
    "year": 365, "yearly": 365, "annual": 365,
}

# Date formats we accept as reliable, tried in order. We never guess: an unparseable value
# leaves the result undated (excluded from the score, counted only in coverage — §8.3).
_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%B %d, %Y",
    "%b %d, %Y",
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


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


# ---- Freshness (§8.3): dated-in-window share + disclosed date coverage --------------------
#
# On time-sensitive queries (a row's ``freshness_need`` is set), a provider's freshness score is
# the share of its returned results that fall inside the query's freshness window — counting ONLY
# results that carry a reliable date (provider ``published_date``, else a date parsed from the
# content). A missing date is NEVER estimated: undated results are excluded from the score and
# counted only in coverage. Low coverage flags the score low-confidence.

# Below this date-coverage fraction the freshness score rests on too few dated results to trust.
FRESHNESS_LOW_CONFIDENCE_COVERAGE = 0.5


def parse_freshness_window_days(freshness_need: Optional[str],
                                default_days: int = DEFAULT_FRESHNESS_WINDOW_DAYS) -> int:
    """Read a ``freshness_need`` cell into a window length in days.

    Accepts a bare integer (days), a ``<n>d`` / ``<n>day(s)`` span, or a named alias
    (day/week/month/quarter/year). Anything unrecognized (including an empty marker) falls back
    to ``default_days`` — the row is still time-sensitive, we just use the default window.
    """
    if freshness_need is None:
        return default_days
    s = str(freshness_need).strip().lower()
    if not s:
        return default_days
    if s in _FRESHNESS_WINDOW_ALIASES:
        return _FRESHNESS_WINDOW_ALIASES[s]
    m = re.fullmatch(r"(\d+)\s*(d|day|days)?", s)
    if m:
        n = int(m.group(1))
        return n if n > 0 else default_days
    return default_days


def parse_reliable_date(doc, now: Optional[datetime] = None) -> Optional[datetime]:
    """Return a reliable UTC datetime for one result, or None (never guessed).

    Prefers the provider ``published_date``; falls back to an ISO date found in the content.
    A future date (later than ``now``) is rejected as unreliable rather than counted as fresh.
    """
    now = now or datetime.now(timezone.utc)
    for raw in (getattr(doc, "published_date", None), getattr(doc, "content", None)):
        dt = _parse_date_str(raw)
        if dt is not None and dt <= now:
            return dt
    return None


def _parse_date_str(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    m = _ISO_DATE_RE.search(s)  # last resort: an ISO date embedded in free-text content
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def freshness_score(docs, window_days: int, now: Optional[datetime] = None) -> Dict[str, Optional[float]]:
    """Freshness for one provider on one time-sensitive query's returned ``docs`` (§8.3).

    ``score`` = dated-in-window / dated (None when no result carries a reliable date).
    ``coverage`` = dated / total. ``low_confidence`` is True when coverage is below the bar
    (the score rests on too few dated results). Undated results only affect coverage.
    """
    now = now or datetime.now(timezone.utc)
    total = len(docs)
    dated = 0
    in_window = 0
    for d in docs:
        dt = parse_reliable_date(d, now=now)
        if dt is None:
            continue
        dated += 1
        if (now - dt).total_seconds() <= window_days * 86400:
            in_window += 1
    coverage = (dated / total) if total else None
    score = (in_window / dated) if dated else None
    low_confidence = coverage is None or coverage < FRESHNESS_LOW_CONFIDENCE_COVERAGE
    return {"score": score, "coverage": coverage, "dated": dated, "in_window": in_window,
            "n_results": total, "low_confidence": low_confidence}


def aggregate_freshness(per_query: List[Dict[str, int]]) -> Optional[Dict[str, Optional[float]]]:
    """Combine a provider's per-query freshness tallies across the time-sensitive queries.

    Each item is a ``freshness_score`` result. Returns None when the provider had no
    time-sensitive results at all (so freshness is absent for it and its weight is dropped, §8).
    Score/coverage are pooled over raw counts (result-weighted), not averaged over queries.
    """
    tallies = [q for q in per_query if q and q.get("n_results")]
    if not tallies:
        return None
    total = sum(q["n_results"] for q in tallies)
    dated = sum(q["dated"] for q in tallies)
    in_window = sum(q["in_window"] for q in tallies)
    coverage = (dated / total) if total else None
    score = (in_window / dated) if dated else None
    low_confidence = coverage is None or coverage < FRESHNESS_LOW_CONFIDENCE_COVERAGE
    return {"score": score, "coverage": coverage, "dated": dated, "in_window": in_window,
            "n_results": total, "n_queries": len(tallies), "low_confidence": low_confidence}
