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


def load_pricing(path: Optional[str] = None) -> dict:
    """Load the pricing map. Returns ``{as_of, providers: {name: {unit, usd_per_unit}}}``.

    Missing file -> empty map (cost simply stays blank; the run never errors on it)."""
    path = path or DEFAULT_PRICING_PATH
    if not os.path.isfile(path):
        logger.info(f"No pricing file at {path}; cost column will be blank")
        return {"as_of": None, "providers": {}}
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pricing root must be a mapping")
    providers = raw.get("providers", {}) or {}
    return {"as_of": raw.get("as_of"), "providers": providers}


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


def present_cost_providers(metrics: Dict[str, dict]) -> List[str]:
    """Providers that have a non-blank cost — used to decide whether the cost weight survives."""
    return [p for p, m in metrics.items()
            if (m.get("cost") or {}).get("usd_per_query") is not None]
