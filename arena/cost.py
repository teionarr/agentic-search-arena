"""Cost-per-query column from the dated pricing map (§8.2).

Cost is ``usd_per_unit × units_consumed``, normalized to **$/query** per provider. The pricing
file (``configs/pricing.yaml``) ships current-ish defaults with an explicit ``as_of`` date; that
date is surfaced with the cost column so staleness is visible, never silently wrong.

Honesty rule: a provider that reports **no units** (``cost_units`` absent — the adapter left it
``None``) gets a **blank** cost. We never fabricate a cost from an assumed unit count. When cost is
blank/partial for the run, its weight is dropped and the remaining weights renormalize (§8) via the
existing :func:`arena.metrics.renormalize_weights`.
"""

import logging
import os
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = "configs/pricing.yaml"


_EMPTY_PRICING = {"as_of": None, "providers": {}}


def load_pricing(path: Optional[str] = None) -> dict:
    """Load the pricing map. Returns ``{as_of, providers: {name: {unit, usd_per_unit}}}``.

    Pricing is OPTIONAL (§8.2): a missing file, invalid YAML, a non-mapping root, or a
    non-mapping ``providers`` all leave cost blank rather than aborting the run — so this
    ALWAYS returns a dict with ``as_of`` + ``providers`` (empty mapping by default)."""
    path = path or DEFAULT_PRICING_PATH
    if not os.path.isfile(path):
        logger.info(f"No pricing file at {path}; cost column will be blank")
        return dict(_EMPTY_PRICING)
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("pricing root must be a mapping")
        providers = raw.get("providers", {}) or {}
        if not isinstance(providers, dict):
            raise ValueError("pricing 'providers' must be a mapping")
        return {"as_of": raw.get("as_of"), "providers": providers}
    except Exception as e:
        logger.warning(f"Ignoring pricing file {path} ({e.__class__.__name__}: {e}); "
                       "cost column will be blank")
        return dict(_EMPTY_PRICING)


def cost_per_query(pricing: dict, provider: str, units_consumed: Optional[float]) -> Optional[float]:
    """``usd_per_unit × units_consumed`` for one provider, or ``None`` when it can't be computed.

    Returns ``None`` (blank) when the provider is unpriced or reports no units — never fabricated."""
    if units_consumed is None:
        return None
    spec = (pricing.get("providers") or {}).get(provider)
    if not spec:
        return None
    unit_price = spec.get("usd_per_unit")
    if unit_price is None:
        return None
    return float(unit_price) * float(units_consumed)


def cost_block(pricing: dict, provider: str, units_consumed: Optional[float]) -> dict:
    """The per-provider cost cell for the metrics dict: ``$/query`` + the pricing ``as_of`` date.

    ``usd_per_query`` is ``None`` (blank) when uncomputable; ``as_of`` is always surfaced so the
    reader sees how fresh the prices are even when a given provider has no cost."""
    return {
        "usd_per_query": cost_per_query(pricing, provider, units_consumed),
        "unit": ((pricing.get("providers") or {}).get(provider) or {}).get("unit"),
        "units_consumed": units_consumed,
        "as_of": pricing.get("as_of"),
    }


def attach_cost(metrics: Dict[str, dict], pricing: dict,
                units_by_provider: Dict[str, Optional[float]]) -> Dict[str, dict]:
    """Add a ``cost`` block to each provider's metrics in-place, and return ``metrics``.

    ``units_by_provider`` maps provider -> units consumed for the run (``None`` when the adapter
    reported no units). Additive: existing metric cells are untouched."""
    for provider, m in metrics.items():
        m["cost"] = cost_block(pricing, provider, units_by_provider.get(provider))
    return metrics


def attach_cost_per_success(metrics: Dict[str, dict]) -> Dict[str, dict]:
    """Cost per *successful outcome*: ``usd_per_query ÷ accuracy_rate`` (§8 cost normalization —
    a cheap API that needs three tries isn't cheap).

    Derived only where an accuracy anchor exists AND the provider has a cost; blank (None) when
    unanchored, unpriced, or the provider never answered correctly — never fabricated. Assumes
    uniform per-query cost across anchored and unanchored queries (the only per-query cost we
    track), which the field name makes auditable rather than hidden."""
    for m in metrics.values():
        cost = m.get("cost")
        if cost is None:
            continue
        upq = cost.get("usd_per_query")
        rate = (m.get("accuracy") or {}).get("rate")
        cost["usd_per_correct"] = (upq / rate) if (upq is not None and rate) else None
    return metrics


def present_cost_providers(metrics: Dict[str, dict]) -> List[str]:
    """Providers that have a non-blank cost — used to decide whether the cost weight survives."""
    return [p for p, m in metrics.items()
            if (m.get("cost") or {}).get("usd_per_query") is not None]


# How to tell whether a weighted metric axis is actually populated for the run. An axis with no
# data has its weight dropped and the rest renormalized (§8) — never scored as zero. Axes not
# listed here (unknown/custom) pass through as present. This is the single presence oracle shared
# by cost, freshness, accuracy, latency and coverage so they all renormalize together.
_AXIS_PRESENCE = {
    "cost": lambda m: (m.get("cost") or {}).get("usd_per_query") is not None,
    "freshness": lambda m: (m.get("freshness") or {}).get("score") is not None,
    "accuracy": lambda m: (m.get("accuracy") or {}).get("rate") is not None,
    "latency": lambda m: (m.get("latency") or {}).get("p50") is not None,
    "coverage": lambda m: (m.get("coverage") or {}).get("avg_tokens_per_result") is not None,
}


def _axis_present(axis: str, metrics: Dict[str, dict]) -> bool:
    """True if any provider has data for ``axis``. Unknown axes are treated as present."""
    check = _AXIS_PRESENCE.get(axis)
    if check is None:
        return True
    return any(check(m) for m in metrics.values())


def effective_weights(weights: Dict[str, float], metrics: Dict[str, dict]) -> Dict[str, float]:
    """Renormalized metric weights with every absent/partial axis dropped (§8).

    The single sanctioned renormalization path for all weighted metrics: cost, freshness,
    accuracy, latency and coverage each survive only if some provider has data for them, else
    the weight is dropped and the remainder renormalized via the existing
    :func:`arena.metrics.renormalize_weights`. No competing per-axis renormalization exists."""
    from arena.metrics import renormalize_weights
    present = [k for k in weights if _axis_present(k, metrics)]
    return renormalize_weights(weights, present)
