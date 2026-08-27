"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def carbon_aware_region_rank(regions: list | None = None, carbon_weight: float = 0.5) -> list:
    """"Your Turn" #5 — rank regions by a blended, min-max normalized cost+carbon score.

    Lower score is better. Equal-weighted by default; raise carbon_weight toward 1.0 to
    bias a scheduler for shiftable/interruptible jobs toward decarbonization over pure
    electricity price. Returns [(region, score), ...] sorted best-first.
    """
    names = regions if regions is not None else list(REGION_CARBON.keys())
    carbons = [REGION_CARBON[n] for n in names]
    prices = [REGION_PRICE_KWH[n] for n in names]
    c_lo, c_hi = min(carbons), max(carbons)
    p_lo, p_hi = min(prices), max(prices)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    scored = []
    for n in names:
        c_n = norm(REGION_CARBON[n], c_lo, c_hi)
        p_n = norm(REGION_PRICE_KWH[n], p_lo, p_hi)
        scored.append((n, carbon_weight * c_n + (1 - carbon_weight) * p_n))
    return sorted(scored, key=lambda x: x[1])


def region_shift_savings(kwh: float, from_region: str, to_region: str) -> dict:
    """"Your Turn" #5 — $ and gCO2e saved moving `kwh` of shiftable compute between regions."""
    from_cost = kwh * REGION_PRICE_KWH.get(from_region, 0.12)
    to_cost = kwh * REGION_PRICE_KWH.get(to_region, 0.12)
    from_carbon = kwh * REGION_CARBON.get(from_region, 400)
    to_carbon = kwh * REGION_CARBON.get(to_region, 400)
    return {
        "usd_saved": round(from_cost - to_cost, 4),
        "gco2_saved": round(from_carbon - to_carbon, 2),
    }
