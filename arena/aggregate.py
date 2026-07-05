"""Aggregation: round-robin win-rate + bootstrap confidence intervals.

Deviation from requirements §6.3 (Bradley-Terry), documented in the plan: on M0's complete,
balanced, single-judge comparison graph win-rate and BT rank equivalently, and win-rate
removes the MLE/separability/convergence/prior-tuning risk surface. BT returns at M1 only if
the comparison graph goes unbalanced.

Deterministic: a pinned bootstrap seed + fixed resampling procedure => byte-reproducible.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_COMPARISONS = 2  # below this a provider is 'unranked — insufficient valid comparisons'


@dataclass
class ProviderScore:
    provider: str
    win_rate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_comparisons: int
    status: str          # "ranked" | "unranked"
    rank: Optional[int] = None
    tie_group: Optional[int] = None


@dataclass
class Aggregation:
    scores: List[ProviderScore]
    n_decided: int
    n_excluded: int
    tie_groups: List[List[str]] = field(default_factory=list)


def _winrate_from(games: Dict[str, List[float]], providers: List[str]) -> Dict[str, float]:
    return {p: (float(np.mean(games[p])) if games[p] else float("nan")) for p in providers}


def aggregate(comparisons: List[dict], providers: List[str], seed: int = 0,
              n_boot: int = 2000, ci: float = 0.95, min_comparisons: int = MIN_COMPARISONS) -> Aggregation:
    """Aggregate pairwise comparisons into a ranking.

    ``comparisons``: list of ``{"a": name, "b": name, "winner": name|"tie"|None}``.
    ``winner is None`` means excluded (swap-flip / skipped) and is ignored.
    """
    decided = [c for c in comparisons if c.get("winner") is not None]
    n_excluded = len(comparisons) - len(decided)

    # Per-provider game scores (win=1, loss=0, tie=0.5).
    games: Dict[str, List[float]] = {p: [] for p in providers}
    for c in decided:
        a, b, w = c["a"], c["b"], c["winner"]
        if w == "tie":
            games[a].append(0.5); games[b].append(0.5)
        elif w == a:
            games[a].append(1.0); games[b].append(0.0)
        elif w == b:
            games[a].append(0.0); games[b].append(1.0)

    point = _winrate_from(games, providers)

    # Bootstrap over the decided comparisons (seeded, fixed procedure).
    rng = np.random.RandomState(seed)
    boot: Dict[str, List[float]] = {p: [] for p in providers}
    if decided:
        idx_all = np.arange(len(decided))
        for _ in range(n_boot):
            sample = rng.choice(idx_all, size=len(decided), replace=True)
            g: Dict[str, List[float]] = {p: [] for p in providers}
            for i in sample:
                c = decided[i]
                a, b, w = c["a"], c["b"], c["winner"]
                if w == "tie":
                    g[a].append(0.5); g[b].append(0.5)
                elif w == a:
                    g[a].append(1.0); g[b].append(0.0)
                elif w == b:
                    g[a].append(0.0); g[b].append(1.0)
            for p in providers:
                if g[p]:
                    boot[p].append(float(np.mean(g[p])))

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100

    scores: List[ProviderScore] = []
    for p in providers:
        n = len(games[p])
        if n < min_comparisons or np.isnan(point[p]):
            scores.append(ProviderScore(p, None, None, None, n, "unranked"))
            continue
        arr = np.array(boot[p]) if boot[p] else np.array([point[p]])
        scores.append(ProviderScore(
            provider=p, win_rate=point[p],
            ci_low=float(np.percentile(arr, lo_pct)),
            ci_high=float(np.percentile(arr, hi_pct)),
            n_comparisons=n, status="ranked",
        ))

    ranked = sorted([s for s in scores if s.status == "ranked"],
                    key=lambda s: s.win_rate, reverse=True)
    unranked = [s for s in scores if s.status == "unranked"]

    # Contiguous tie-grouping over sorted order: join current group if CIs overlap.
    tie_groups: List[List[str]] = []
    for s in ranked:
        if tie_groups and _overlaps(s, _by_name(ranked, tie_groups[-1][-1])):
            tie_groups[-1].append(s.provider)
        else:
            tie_groups.append([s.provider])
    for gi, group in enumerate(tie_groups):
        for name in group:
            sc = _by_name(ranked, name)
            sc.tie_group = gi
    for i, s in enumerate(ranked):
        s.rank = i + 1

    return Aggregation(scores=ranked + unranked, n_decided=len(decided),
                       n_excluded=n_excluded, tie_groups=tie_groups)


def _overlaps(a: ProviderScore, b: ProviderScore) -> bool:
    return not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)


def _by_name(scores: List[ProviderScore], name: str) -> ProviderScore:
    for s in scores:
        if s.provider == name:
            return s
    raise KeyError(name)
