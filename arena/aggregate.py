"""Aggregation over the swap-survived pairwise comparisons.

Two selectable estimators (config ``aggregation.method``):

- ``bradley_terry`` (DEFAULT, §6.3): a Bradley-Terry MLE strength per provider, fit numpy-only
  with the standard MM (minorization-maximization) iteration — no scipy. Strengths are mapped to
  a 0–1 "win-rate-scale" expected-score so the CI/tie/report path is unchanged. Confidence
  intervals come from the SAME seeded bootstrap the win-rate path uses, so output stays
  byte-reproducible.
- ``winrate``: the round-robin win-rate + bootstrap CI estimator (the documented M0 deviation),
  kept intact for continuity.

Both estimators share the seeded bootstrap, overlapping-CI tie grouping, and the
``unranked — insufficient valid comparisons`` floor.

**Judge-reliability weighting (§6.3)** is per-judge and engages ONLY when a per-judge signal
exists (a secondary judge's agreement, or gold calibration). It is passed in as an optional
per-comparison ``weight``; the default single-judge no-gold run supplies none, so aggregation is
plain unweighted Bradley-Terry over the swap-survived set.

Deterministic: a pinned bootstrap seed + fixed resampling + fixed BT iteration => byte-reproducible.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_COMPARISONS = 2  # below this a provider is 'unranked — insufficient valid comparisons'
BT_MAX_ITERS = 1000
BT_TOL = 1e-9


@dataclass
class ProviderScore:
    provider: str
    win_rate: Optional[float]        # BT: expected-score on the 0–1 scale; winrate: raw win-rate
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
    method: str = "bradley_terry"
    tie_groups: List[List[str]] = field(default_factory=list)


def _weight_of(c: dict) -> float:
    """Per-comparison weight (judge-reliability weighting, §6.3). Absent => 1.0 (unweighted).

    Validated at the boundary: nan/inf/zero/negative weights would corrupt the weighted
    wins/games and BT CIs (and could mark a provider as played on a zero weight), so reject them.
    """
    w = c.get("weight")
    if w is None:
        return 1.0
    wt = float(w)
    if not np.isfinite(wt) or wt <= 0:
        raise ValueError("comparison weight must be a positive finite number")
    return wt


def _winrate_point(decided: List[dict], providers: List[str]) -> Dict[str, float]:
    """Weighted win-rate (win=1, loss=0, tie=0.5) per provider; NaN if it played nothing."""
    num: Dict[str, float] = {p: 0.0 for p in providers}
    den: Dict[str, float] = {p: 0.0 for p in providers}
    for c in decided:
        a, b, w = c["a"], c["b"], c["winner"]
        wt = _weight_of(c)
        if w == "tie":
            num[a] += 0.5 * wt; num[b] += 0.5 * wt
        elif w == a:
            num[a] += 1.0 * wt
        elif w == b:
            num[b] += 1.0 * wt
        else:
            continue
        den[a] += wt; den[b] += wt
    return {p: (num[p] / den[p] if den[p] > 0 else float("nan")) for p in providers}


def _components(games: np.ndarray, played: np.ndarray) -> List[List[int]]:
    """Connected components of the comparison graph (adjacency ``games`` > 0), over played nodes.

    BT strengths are only identifiable WITHIN a connected component — the relative scale between
    two providers that never share a comparison path is undefined. So we fit and normalize each
    component separately; cross-component providers then land at incomparable scales, which the
    CI/tie logic honestly reflects as unresolved rather than as a spurious ordering.
    """
    n = games.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for start in range(n):
        if seen[start] or not played[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not seen[v] and games[u, v] > 0:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _bt_fit(decided: List[dict], providers: List[str]) -> Dict[str, float]:
    """Bradley-Terry MLE strengths via the MM iteration (Hunter 2004), numpy-only.

    Returns per-provider strength ``pi`` (positive). Providers with no games get ``nan``. Ties
    count as half a win to each side. Weights scale each game's contribution. Each CONNECTED
    COMPONENT of the comparison graph is fit and geometric-mean-normalized independently — strength
    is only identifiable within a component, so disconnected groups are not forced onto one scale.
    """
    idx = {p: i for i, p in enumerate(providers)}
    n = len(providers)
    # wins[i]: weighted wins credited to i; games[i][j]: weighted games between i and j.
    wins = np.zeros(n)
    games = np.zeros((n, n))
    played = np.zeros(n, dtype=bool)
    for c in decided:
        a, b, w = c["a"], c["b"], c["winner"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        wt = _weight_of(c)
        games[i, j] += wt; games[j, i] += wt
        played[i] = played[j] = True
        if w == "tie":
            wins[i] += 0.5 * wt; wins[j] += 0.5 * wt
        elif w == a:
            wins[i] += wt
        elif w == b:
            wins[j] += wt

    pi = np.ones(n)
    active = played.copy()
    if not active.any():
        return {p: float("nan") for p in providers}

    for comp in _components(games, played):
        members = np.array(comp)
        cpi = np.ones(len(members))
        for _ in range(BT_MAX_ITERS):
            prev = cpi.copy()
            for ci, i in enumerate(members):
                denom = 0.0
                for cj, j in enumerate(members):
                    if j == i or games[i, j] == 0:
                        continue
                    denom += games[i, j] / (cpi[ci] + cpi[cj])
                if denom > 0 and wins[i] > 0:
                    cpi[ci] = wins[i] / denom
                # wins[i] == 0 (no credited wins): leave tiny but positive.
            gm = np.exp(np.mean(np.log(np.clip(cpi, 1e-12, None))))
            cpi = cpi / gm
            prev_gm = np.exp(np.mean(np.log(np.clip(prev, 1e-12, None))))
            if np.max(np.abs(cpi - prev / prev_gm)) < BT_TOL:
                break
        for ci, i in enumerate(members):
            pi[i] = cpi[ci]

    return {p: (float(pi[idx[p]]) if active[idx[p]] else float("nan")) for p in providers}


def _bt_expected_scores(decided: List[dict], providers: List[str]) -> Dict[str, float]:
    """Map BT strengths to a 0–1 expected-score against the field's mean opponent.

    Uses a fixed reference opponent (the geometric mean strength => 1 after normalization), so
    ``score = pi / (pi + 1)``. Monotone in strength, on the same 0–1 scale as win-rate, so it
    flows through the existing CI / tie-grouping / report path unchanged.
    """
    strengths = _bt_fit(decided, providers)
    out: Dict[str, float] = {}
    for p in providers:
        s = strengths[p]
        out[p] = float("nan") if (s is None or np.isnan(s)) else s / (s + 1.0)
    return out


def _point_estimate(method: str, decided: List[dict], providers: List[str]) -> Dict[str, float]:
    if method == "winrate":
        return _winrate_point(decided, providers)
    if method == "bradley_terry":
        return _bt_expected_scores(decided, providers)
    raise ValueError(f"Unknown aggregation method: {method!r}. Known: 'bradley_terry', 'winrate'")


def aggregate(comparisons: List[dict], providers: List[str], seed: int = 0,
              n_boot: int = 2000, ci: float = 0.95, min_comparisons: int = MIN_COMPARISONS,
              method: str = "bradley_terry") -> Aggregation:
    """Aggregate pairwise comparisons into a ranking.

    ``comparisons``: list of ``{"a": name, "b": name, "winner": name|"tie"|None, "weight"?: float}``.
    ``winner is None`` means excluded (swap-flip / skipped) and is ignored. ``weight`` (optional,
    §6.3) scales a comparison's contribution for judge-reliability weighting; absent => 1.0.
    ``method``: ``"bradley_terry"`` (default) or ``"winrate"``.
    """
    decided = [c for c in comparisons if c.get("winner") is not None]
    n_excluded = len(comparisons) - len(decided)

    point = _point_estimate(method, decided, providers)

    # Games-played count per provider (drives the min-comparisons floor). Weights don't count
    # here — the floor is about how many valid comparisons exist, not their reliability.
    n_games: Dict[str, int] = {p: 0 for p in providers}
    for c in decided:
        n_games[c["a"]] += 1
        n_games[c["b"]] += 1

    # Bootstrap over the decided comparisons (seeded, fixed procedure) — shared by both methods.
    rng = np.random.RandomState(seed)
    boot: Dict[str, List[float]] = {p: [] for p in providers}
    if decided:
        idx_all = np.arange(len(decided))
        for _ in range(n_boot):
            sample = rng.choice(idx_all, size=len(decided), replace=True)
            resampled = [decided[i] for i in sample]
            est = _point_estimate(method, resampled, providers)
            for p in providers:
                if not np.isnan(est[p]):
                    boot[p].append(est[p])

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100

    scores: List[ProviderScore] = []
    for p in providers:
        n = n_games[p]
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
                       n_excluded=n_excluded, method=method, tie_groups=tie_groups)


def _overlaps(a: ProviderScore, b: ProviderScore) -> bool:
    return not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)


def _by_name(scores: List[ProviderScore], name: str) -> ProviderScore:
    for s in scores:
        if s.provider == name:
            return s
    raise KeyError(name)
