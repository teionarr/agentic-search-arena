"""Judge-reliability signals (§6.4) and judge-reliability weighting (§6.3).

Two things live here, both pure and AI-free:

- **Inter-judge agreement (Cohen's κ, §6.4).** When a secondary judge is configured every pair
  is judged by BOTH judges; κ measures agreement above chance over their paired verdicts. It is
  reported regardless of value (Tier B bar κ ≥ 0.6).

- **Judge-reliability weighting (§6.3).** Per-judge and engages ONLY when a per-judge signal
  exists — a secondary judge's cross-agreement here, or gold calibration elsewhere. In the
  default single-judge no-gold run there is no per-judge signal, so ``judge_weights`` returns an
  empty mapping and aggregation stays PLAIN unweighted Bradley-Terry. With a secondary judge, a
  judge that agrees less with the ensemble is down-weighted in the expected direction.

Weighting is expressed as a per-JUDGE scalar; the pipeline folds it into each comparison's
``weight`` before calling ``aggregate`` (a comparison contributed by a down-weighted judge
carries proportionally less).
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    """Cohen's κ between two raters over paired categorical labels.

    ``labels_a[i]`` and ``labels_b[i]`` are the two judges' verdicts on the same item (e.g.
    "x" / "y" / "tie"). Returns ``None`` if there are no paired items. When both raters are
    perfectly constant on the same label (no chance-correctable variance) κ is defined as 1.0.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must be the same length")
    n = len(labels_a)
    if n == 0:
        return None

    cats = sorted(set(labels_a) | set(labels_b))
    ci = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    conf = np.zeros((k, k))
    for a, b in zip(labels_a, labels_b):
        conf[ci[a], ci[b]] += 1

    po = np.trace(conf) / n
    row = conf.sum(axis=1) / n
    col = conf.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if pe >= 1.0:  # both raters degenerate on one label => perfect (definitional) agreement
        return 1.0
    return float((po - pe) / (1.0 - pe))


def judge_weights(agreements: Dict[str, float], mode: str = "auto") -> Dict[str, float]:
    """Per-judge reliability weights from each judge's agreement-with-ensemble signal.

    ``agreements``: ``judge_id -> agreement rate in [0, 1]`` (share of items where the judge
    matched the ensemble/consensus). Engages ONLY when the signal actually DISCRIMINATES between
    judges (§6.3):

    - fewer than 2 judges, or ``mode == "off"``  => ``{}`` (no weighting; plain unweighted BT).
    - if every judge has the same agreement (no per-judge signal to weight on — e.g. a pure
      two-judge run, where cross-agreement is symmetric and cannot say WHICH judge is right)
      => ``{}`` (report κ, stay unweighted). This is the spec-correct default: down-weighting
      needs a genuine reliability signal (a 3+ judge consensus or gold calibration), not an
      arbitrary tie-break.
    - otherwise each judge's weight is its agreement rate (clamped to a small floor so a fully
      unreliable judge is discounted but never fully silenced), normalized so the mean weight is
      1.0. A judge that agrees less with the ensemble gets a strictly smaller weight — the
      expected direction.
    """
    if mode == "off" or len(agreements) < 2:
        return {}
    vals = list(agreements.values())
    if max(vals) - min(vals) < 1e-12:  # no discriminating signal -> unweighted (report κ only)
        return {}
    floor = 0.05
    raw = {j: max(floor, float(a)) for j, a in agreements.items()}
    mean = float(np.mean(list(raw.values())))
    if mean <= 0:
        return {}
    return {j: w / mean for j, w in raw.items()}


def consensus_agreements(per_judge_labels: Dict[str, List[str]]) -> Dict[str, float]:
    """Each judge's agreement with the consensus (majority) label, item by item.

    ``per_judge_labels``: ``judge_id -> [label per item]`` (all lists the same length, aligned by
    item). Used to feed ``judge_weights`` when 3+ judges are configured.

    With exactly TWO judges there is no majority — every disagreement is a 1-1 tie. Picking a
    "winner" by sorted label would make a judge look more reliable purely for choosing e.g. ``"x"``
    over ``"y"`` (i.e. by provider-name order), biasing the weights. So for two judges we return
    the SYMMETRIC pairwise agreement rate for BOTH (they agree the same amount by definition);
    ``judge_weights`` then sees equal agreements and stays unweighted. For 3+ judges the modal
    label is a real majority; a rare exact modal tie is broken deterministically by sorted label
    only for reproducibility (it no longer drives asymmetric two-judge weighting).
    """
    ids = list(per_judge_labels)
    if len(ids) < 2:
        return {}
    n = len(per_judge_labels[ids[0]])
    if any(len(per_judge_labels[j]) != n for j in ids):
        raise ValueError("per_judge_labels must contain aligned (equal-length) label lists")

    if len(ids) == 2:  # symmetric pairwise agreement — no arbitrary tie-break (§6.3)
        a, b = ids
        agree = (sum(1 for la, lb in zip(per_judge_labels[a], per_judge_labels[b]) if la == lb) / n
                 if n else 0.0)
        return {a: agree, b: agree}

    hits: Dict[str, int] = {j: 0 for j in ids}
    for i in range(n):
        col = [per_judge_labels[j][i] for j in ids]
        # Modal label across 3+ judges; exact ties broken by sorted label for reproducibility.
        counts: Dict[str, int] = {}
        for c in col:
            counts[c] = counts.get(c, 0) + 1
        top = max(sorted(counts), key=lambda c: counts[c])
        for j in ids:
            if per_judge_labels[j][i] == top:
                hits[j] += 1
    return {j: (hits[j] / n if n else 0.0) for j in ids}
