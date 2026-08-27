"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    interrupt_rate: float = 0.05,
    spot_to_on_demand_ratio: float = 0.60,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
    reserved_1yr_discount: float = 0.20,
    reserved_3yr_discount: float = 0.45,
    high_duty_for_3yr: float = 0.85,
) -> dict:
    """"Your Turn" #1 — recommend_tier() with interruption rate + 1yr-vs-3yr reserved.

    recommend_tier() treats every interruptible job as an automatic spot win and every
    reserved commitment as the same length. Neither holds up: a high interrupt_rate can
    erase spot's discount once checkpoint rework is priced in, and a 3yr lock-in needs a
    steadier duty cycle than a 1yr one to be safe. This policy:

      1. Only recommends spot if the interruption-adjusted effective cost (checkpoint
         overhead + expected rework hours) still beats on-demand.
      2. Otherwise checks duty cycle against the 1yr and 3yr break-evens separately,
         requiring `high_duty_for_3yr` before taking the longer, deeper discount.

    Returns {'tier': ..., 'reason': ...} instead of a bare string so the rationale is
    inspectable (useful when comparing against recommend_tier()'s pick).
    """
    duty = max(0.0, hours_per_day) / 24.0

    if interruptible and hours_per_day < 24:
        effective_hours_frac = (1.0 + ckpt_overhead_frac) + interrupt_rate * rework_hours_per_interrupt
        effective_spot_frac = effective_hours_frac * spot_to_on_demand_ratio
        if effective_spot_frac < 1.0:
            return {"tier": "spot", "reason": f"interrupt-adjusted cost {effective_spot_frac:.2f}x on-demand"}

    be_3yr = break_even_utilization(reserved_3yr_discount)
    be_1yr = break_even_utilization(reserved_1yr_discount)
    if duty >= max(be_3yr, high_duty_for_3yr):
        return {"tier": "reserved_3yr",
                "reason": f"duty {duty:.0%} clears the {high_duty_for_3yr:.0%} bar for a 3yr commit"}
    if duty >= be_1yr:
        return {"tier": "reserved_1yr",
                "reason": f"duty {duty:.0%} clears 1yr break-even ({be_1yr:.0%}) but not the 3yr bar"}
    return {"tier": "on_demand", "reason": f"duty {duty:.0%} too low/spiky for any commitment"}


def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """"Your Turn" #2 helper — $/hr per GB of VRAM (lower is better for memory-bound jobs)."""
    if hbm_gb <= 0:
        return float("inf")
    return on_demand_hr / hbm_gb


def cache_breakeven_hit_frac(cache_write_overhead_frac: float = 0.25, cache_discount: float = 0.10) -> float:
    """"Your Turn" #3 — minimum cache-hit fraction before prompt caching pays for itself.

    A cache write costs (1 + cache_write_overhead_frac)x a normal read once; each hit
    after that costs cache_discount x. Break-even is the write overhead amortized over
    the discount a hit earns you: write_overhead / (1 - discount).
    """
    return min(1.0, max(0.0, cache_write_overhead_frac / (1.0 - cache_discount)))


def cache_is_worth_it(
    cache_hit_frac: float,
    cache_write_overhead_frac: float = 0.25,
    cache_discount: float = 0.10,
) -> bool:
    """"Your Turn" #3 — True once cache_hit_frac clears the write-overhead break-even.

    Below break-even, the one-time write premium isn't amortized by enough discounted
    reads and caching *adds* cost versus never caching at all.
    """
    return cache_hit_frac >= cache_breakeven_hit_frac(cache_write_overhead_frac, cache_discount)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
