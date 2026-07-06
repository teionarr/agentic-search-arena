"""Post-hoc weighted re-ranking over a canonical results.json (§8/§10).

The user weights the axes to their priorities and the tool re-ranks — WITHOUT re-running
(or re-paying for) anything: this operates purely on a results.json a previous run already
wrote, and never mutates it. The unweighted arena ranking is always shown too and stays
primary (§10).

Scoring: per axis, provider values are min-max normalized to [0, 1] across the providers
that have a value (inverted for lower-is-better axes: latency, cost); when every holder has
the same value the axis cannot discriminate and every holder scores 1.0. A provider's score
is the weighted sum of its normalized axis values.

Honesty rules (§8 — never score a missing metric as zero):
- An axis NO provider has data for is dropped and the remaining weights renormalized, via
  the existing :func:`arena.metrics.renormalize_weights` (the same primitive
  :func:`arena.cost.effective_weights` uses; that oracle reads the ``metrics`` dict and does
  not know the ``arena``/``reliability`` axes, so presence here is computed from the same
  field paths on the extracted values instead — identical semantics, one renormalizer).
- A provider missing a value on a PRESENT axis scores ``None`` there, and its remaining
  weights are renormalized FOR THAT PROVIDER (again via ``renormalize_weights``): each
  provider is scored only over the axes it actually has, at proportionally scaled weights.
- Unknown weight keys are rejected loudly, listing the known axes.

Neutrality: no provider-specific branches — every provider flows through the identical
extraction/normalization/summation path (§15).
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

from arena.metrics import renormalize_weights

# Axis -> (lower_is_better, where its value lives in the doc). Order is the canonical
# iteration order everywhere below, which is what makes output deterministic.
AXES: Dict[str, bool] = {
    "arena": False,        # ranking[].win_rate
    "accuracy": False,     # metrics[p].accuracy.rate
    "latency": True,       # metrics[p].latency.p50 (lower is better)
    "cost": True,          # metrics[p].cost.usd_per_query (lower is better)
    "freshness": False,    # metrics[p].freshness.score
    "coverage": False,     # metrics[p].coverage.avg_tokens_per_result
    "reliability": False,  # metrics[p].reliability.success_rate
    "downstream": False,   # metrics[p].downstream.success_rate
}

_METRIC_PATHS = {
    "accuracy": ("accuracy", "rate"),
    "latency": ("latency", "p50"),
    "cost": ("cost", "usd_per_query"),
    "freshness": ("freshness", "score"),
    "coverage": ("coverage", "avg_tokens_per_result"),
    "reliability": ("reliability", "success_rate"),
    "downstream": ("downstream", "success_rate"),
}


def _validate_weights(weights: Dict[str, float]) -> None:
    unknown = [k for k in weights if k not in AXES]
    if unknown:
        raise ValueError(f"Unknown weight axis(es): {', '.join(sorted(unknown))}. "
                         f"Known axes: {', '.join(AXES)}")
    negative = [k for k, v in weights.items() if v is not None and v < 0]
    if negative:
        raise ValueError(f"Negative weight(s) not allowed: {', '.join(sorted(negative))}")


def _providers(doc: dict) -> List[str]:
    return [s["provider"] for s in doc.get("ranking", [])]


def _axis_values(doc: dict) -> Dict[str, Dict[str, Optional[float]]]:
    """Raw value per axis per provider (None where absent). Same field paths the table renders.

    A provider whose arena status is ``unranked`` is listed but has no arena value — it is
    unscored on that axis (its weight renormalizes to its other axes), never zero-scored."""
    metrics = doc.get("metrics") or {}
    values: Dict[str, Dict[str, Optional[float]]] = {a: {} for a in AXES}
    for s in doc.get("ranking", []):
        p = s["provider"]
        values["arena"][p] = s.get("win_rate") if s.get("status") == "ranked" else None
        m = metrics.get(p) or {}
        for axis, (block, field) in _METRIC_PATHS.items():
            values[axis][p] = (m.get(block) or {}).get(field)
    return values


def _normalize(vals: Dict[str, Optional[float]], lower_is_better: bool) -> Dict[str, Optional[float]]:
    """Min-max normalize one axis across the providers that have a value; invert if lower-is-better.

    Degenerate axis (all holders equal, incl. a single holder): 1.0 for every holder — the
    axis cannot discriminate, so no holder is penalized for it."""
    have = {p: v for p, v in vals.items() if v is not None}
    if not have:
        return {p: None for p in vals}
    lo, hi = min(have.values()), max(have.values())
    out: Dict[str, Optional[float]] = {}
    for p, v in vals.items():
        if v is None:
            out[p] = None
        elif hi == lo:
            out[p] = 1.0
        else:
            out[p] = (hi - v) / (hi - lo) if lower_is_better else (v - lo) / (hi - lo)
    return out


def effective_axis_weights(doc: dict, weights: Dict[str, float]) -> Dict[str, float]:
    """User weights with absent axes (no provider has data) dropped and renormalized (§8).

    Rejects unknown axes. Returned in canonical ``AXES`` order (deterministic)."""
    _validate_weights(weights)
    values = _axis_values(doc)
    present = [a for a in weights if any(v is not None for v in values[a].values())]
    eff = renormalize_weights(weights, present)
    return {a: eff[a] for a in AXES if a in eff}


def weighted_scores(doc: dict, weights: Dict[str, float]) -> List[dict]:
    """Re-rank a results.json document by user priority weights. Pure; never mutates ``doc``.

    Returns ``[{provider, weighted_score, rank, per_axis: {axis: {value, normalized, weight}}}]``
    sorted by score descending (ties broken by provider name — deterministic). ``per_axis``
    covers exactly the effective (present) axes; ``weight`` is the PER-PROVIDER renormalized
    weight actually applied (None where the provider lacks that axis). A provider with no
    scored axes at all gets ``weighted_score`` and ``rank`` of None, listed last."""
    eff = effective_axis_weights(doc, weights)
    values = _axis_values(doc)
    normalized = {a: _normalize(values[a], AXES[a]) for a in eff}

    entries = []
    for p in _providers(doc):
        per_axis = {a: {"value": values[a][p], "normalized": normalized[a][p], "weight": None}
                    for a in eff}
        available = [a for a in eff if per_axis[a]["normalized"] is not None]
        w_p = renormalize_weights(eff, available)  # per-provider renormalization (§8)
        score = None
        if w_p:
            score = sum(w_p[a] * per_axis[a]["normalized"] for a in eff if a in w_p)
            for a, w in w_p.items():
                per_axis[a]["weight"] = w
        entries.append({"provider": p, "weighted_score": score, "rank": None,
                        "per_axis": per_axis})

    entries.sort(key=lambda e: (e["weighted_score"] is None,
                                -(e["weighted_score"] or 0.0), e["provider"]))
    for i, e in enumerate(entries):
        if e["weighted_score"] is not None:
            e["rank"] = i + 1
    return entries


# ---- CLI: python -m arena.rerank <results.json> --weights accuracy=0.5,latency=0.3 --------


def _parse_weights_arg(s: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition("=")
        if not sep:
            raise ValueError(f"Malformed weight '{part}' (expected axis=number)")
        try:
            weights[key.strip()] = float(val)
        except ValueError:
            raise ValueError(f"Malformed weight '{part}' (expected axis=number)")
    if not weights:
        raise ValueError("Empty --weights")
    return weights


def render_rerank(doc: dict, weights: Dict[str, float]) -> str:
    """The full re-rank report: unweighted ranking (always, primary) + weighted view."""
    eff = effective_axis_weights(doc, weights)
    scored = weighted_scores(doc, weights)
    dropped = sorted(a for a, v in weights.items() if v is not None and a not in eff)

    out = ["", "  UNWEIGHTED RANKING (arena — always primary)"]
    for s in doc.get("ranking", []):
        if s.get("status") == "ranked":
            out.append(f"   #{s['rank']} {s['provider']:<18} win-rate {s['win_rate']:.2f} "
                       f"[{s['ci_low']:.2f}–{s['ci_high']:.2f}]")
        else:
            out.append(f"   --  {s['provider']:<18} unranked — insufficient valid comparisons")

    out.append("")
    out.append("  WEIGHTED (your priorities)")
    if eff:
        out.append("   effective weights: " + " · ".join(f"{a} {w:.2f}" for a, w in eff.items()))
    if dropped:
        out.append(f"   dropped (no data this run, remaining weights renormalized): "
                   f"{', '.join(dropped)}")
    if not eff:
        out.append("   no weighted axis has data this run — nothing to re-rank")
        out.append("")
        return "\n".join(out)
    for e in scored:
        if e["weighted_score"] is None:
            out.append(f"   --  {e['provider']:<18} no scored axes")
            continue
        parts = []
        for a in eff:
            cell = e["per_axis"][a]
            if cell["normalized"] is None:
                parts.append(f"{a} n/a")  # missing here: its weight was renormalized away
            else:
                parts.append(f"{a} {cell['normalized']:.2f}×{cell['weight']:.2f}"
                             f"={cell['normalized'] * cell['weight']:.2f}")
        out.append(f"   #{e['rank']} {e['provider']:<18} score {e['weighted_score']:.2f}   "
                   + " · ".join(parts))
    out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m arena.rerank",
        description="Re-rank an existing results.json by your priority weights — post-hoc, "
                    "no APIs re-run, input file never modified.")
    parser.add_argument("results", help="path to a results.json from a previous run")
    parser.add_argument("--weights",
                        help="axis=weight[,axis=weight...] e.g. accuracy=0.5,latency=0.3,cost=0.2"
                             f" (known axes: {', '.join(AXES)})")
    args = parser.parse_args(argv)

    with open(args.results, "r") as f:
        doc = json.load(f)

    try:
        if args.weights:
            weights = _parse_weights_arg(args.weights)
        else:
            weights = ((doc.get("config") or {}).get("weights")) or {}
            if not weights:
                print("No --weights given and this run's config carries no weights.\n"
                      f"Pass --weights axis=w,... (known axes: {', '.join(AXES)})")
                return 2
        print(render_rerank(doc, weights))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
