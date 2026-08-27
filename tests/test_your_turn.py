import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics, sustainability
from missions import m6_your_turn


def test_recommend_tier_v2_interruption_rate():
    # low interrupt rate -> spot still wins
    assert pricing.recommend_tier_v2(10, True, interrupt_rate=0.02)["tier"] == "spot"
    # extreme interrupt rate erases the spot discount -> falls through to reserved/on_demand
    r = pricing.recommend_tier_v2(10, True, interrupt_rate=3.0)
    assert r["tier"] != "spot"


def test_recommend_tier_v2_1yr_vs_3yr():
    assert pricing.recommend_tier_v2(24, False)["tier"] == "reserved_3yr"          # 100% duty
    assert pricing.recommend_tier_v2(19.5, False)["tier"] == "reserved_1yr"        # 81% duty
    assert pricing.recommend_tier_v2(2, False)["tier"] == "on_demand"              # spiky


def test_dollars_per_gb_vram():
    assert abs(pricing.dollars_per_gb_vram(2.0, 80) - 0.025) < 1e-9
    assert pricing.dollars_per_gb_vram(2.0, 0) == float("inf")


def test_cache_breakeven_and_worth_it():
    be = pricing.cache_breakeven_hit_frac(cache_write_overhead_frac=0.25, cache_discount=0.10)
    assert abs(be - 0.25 / 0.9) < 1e-9
    assert pricing.cache_is_worth_it(be + 0.01) is True
    assert pricing.cache_is_worth_it(be - 0.01) is False


def test_right_size_for_memory_bound():
    catalog = {
        "SMALL": {"hbm_gb": "40", "peak_bw_tbs": "1.0", "on_demand_hr": "1.0"},
        "BIG": {"hbm_gb": "200", "peak_bw_tbs": "5.0", "on_demand_hr": "2.0"},
    }
    assert metrics.right_size_for_memory_bound(catalog, min_hbm_gb=100, min_bw_tbs=2.0) == "BIG"
    assert metrics.right_size_for_memory_bound(catalog, min_hbm_gb=1000, min_bw_tbs=2.0) is None


def test_carbon_aware_region_rank_and_shift():
    ranked = sustainability.carbon_aware_region_rank()
    names = [r[0] for r in ranked]
    assert "europe-central2" == names[-1]                 # dirtiest + pricier -> worst score
    s = sustainability.region_shift_savings(1000, "europe-central2", "europe-north1")
    assert s["usd_saved"] > 0 and s["gco2_saved"] > 0


def test_m6_extensions_run_and_measure_something():
    result = m6_your_turn.run(verbose=False)
    assert result["tier_policy_v2"]["changed"] >= 1
    assert result["cache_economics"]["flagged_cost_share"] > 0
    assert result["reasoning_budget"]["kwh_saved_per_month"] > 0
    assert result["carbon_aware_scheduling"]["usd_saved"] > 0
