"""M6 — "Your Turn" extensions (README, section "Phần mở rộng Your Turn").

The lab passes without this file; it exists to *measure* the five instructor
extension points on top of M1-M5, using new helpers added to finops/ for the
purpose (pricing.recommend_tier_v2/dollars_per_gb_vram/cache_is_worth_it,
metrics.right_size_for_memory_bound, sustainability.carbon_aware_region_rank).
Doesn't touch the graded M1-M5 flow, verify.py, or run_all.py.

Run: python missions/m6_your_turn.py   ->  outputs/your_turn.md
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type, ROOT
from missions.m2_inference_levers import MODEL_PRICES
from finops import pricing, metrics, sustainability

DAYS = 30


# 1. Tier policy: interruption rate + 1yr-vs-3yr reserved -----------------------------

def tier_policy_v2() -> dict:
    jobs = load_csv("workloads.csv")
    rows, changed = [], 0
    for j in jobs:
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        old = pricing.recommend_tier(hpd, interruptible)
        new = pricing.recommend_tier_v2(hpd, interruptible)
        equiv = old == new["tier"] or (old == "reserved" and new["tier"] in ("reserved_1yr", "reserved_3yr"))
        if not equiv:
            changed += 1
        rows.append({"job_id": j["job_id"], "v1_tier": old, "v2_tier": new["tier"],
                     "changed": not equiv, "reason": new["reason"]})
    return {"rows": rows, "changed": changed}


# 2. Right-size memory-bound inference GPUs by $/GB-VRAM -----------------------------

def right_size_memory_bound() -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()
    agg = defaultdict(lambda: {"type": None, "workload": None, "tflops": [], "bw": [], "mem": []})
    for r in tel:
        a = agg[r["gpu_id"]]
        a["type"] = r["gpu_type"]
        a["workload"] = r["workload"]
        a["tflops"].append(num(r["achieved_tflops"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))

    by_type = defaultdict(list)
    for gid, a in agg.items():
        if a["workload"] == "train":
            continue  # training MFU is M1's audit; this is for inference/embed fleets
        avg_tflops = sum(a["tflops"]) / len(a["tflops"])
        avg_bw = sum(a["bw"]) / len(a["bw"])
        intensity = metrics.arithmetic_intensity(avg_tflops, avg_bw) if avg_bw else 0.0
        if metrics.roofline_regime(intensity, ridge_point=295) == "memory-bound":
            by_type[a["type"]].append({"gpu_id": gid, "peak_mem": max(a["mem"]), "avg_bw": avg_bw})

    consolidations = []
    for gtype, members in by_type.items():
        total_mem = sum(m["peak_mem"] for m in members) * 1.15  # headroom
        need_bw = max(m["avg_bw"] for m in members)
        fit = metrics.right_size_for_memory_bound(cat, min_hbm_gb=total_mem, min_bw_tbs=need_bw)
        if not fit or fit == gtype:
            continue
        n_fit = max(1, -(-int(total_mem) // int(num(cat[fit]["hbm_gb"]))))  # ceil div
        cur_monthly = len(members) * num(cat[gtype]["on_demand_hr"]) * 24 * DAYS
        fit_monthly = n_fit * num(cat[fit]["on_demand_hr"]) * 24 * DAYS
        if fit_monthly < cur_monthly:
            consolidations.append({
                "from_type": gtype, "from_count": len(members), "gpu_ids": [m["gpu_id"] for m in members],
                "to_type": fit, "to_count": n_fit,
                "monthly_before": round(cur_monthly, 2), "monthly_after": round(fit_monthly, 2),
                "monthly_savings": round(cur_monthly - fit_monthly, 2),
            })
    return {"consolidations": consolidations,
            "monthly_savings": round(sum(c["monthly_savings"] for c in consolidations), 2)}


# 3. Cache economics: is caching worth it per traffic segment? -----------------------

def cache_economics() -> dict:
    rows = load_csv("token_usage.csv")
    breakeven = pricing.cache_breakeven_hit_frac()
    groups = defaultdict(lambda: {"input": 0, "cached": 0, "cost": 0.0, "n": 0})
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=bool(int(num(r["is_batch"]))))
        g = groups[(r["team"], r["project"])]
        g["input"] += inp
        g["cached"] += cached
        g["cost"] += cost
        g["n"] += 1

    total_cost = sum(g["cost"] for g in groups.values())
    segments = []
    for (team, project), g in groups.items():
        reuse = g["cached"] / g["input"] if g["input"] else 0.0
        worth_it = pricing.cache_is_worth_it(reuse)
        segments.append({"team": team, "project": project or "(untagged)", "requests": g["n"],
                         "reuse_rate": round(reuse, 3), "worth_it": worth_it,
                         "cost_per_day": round(g["cost"], 2),
                         "cost_share": round(g["cost"] / total_cost, 3) if total_cost else 0.0})

    not_worth_it = [s for s in segments if not s["worth_it"]]
    return {"breakeven_hit_frac": round(breakeven, 3), "segments": segments,
            "flagged_segments": not_worth_it,
            "flagged_cost_share": round(sum(s["cost_share"] for s in not_worth_it), 3)}


# 4. Reasoning traffic budget: $ + Wh, propose a routing cap -------------------------

def reasoning_budget(cap_frac: float = 0.05) -> dict:
    rows = load_csv("token_usage.csv")
    total_n = len(rows)
    reasoning_n = sum(1 for r in rows if int(num(r["is_reasoning"])) == 1)

    reasoning_cost = reasoning_wh = other_cost = other_wh = 0.0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=bool(int(num(r["is_batch"]))))
        is_r = int(num(r["is_reasoning"])) == 1
        wh = sustainability.wh_per_query(inp + out, is_reasoning=is_r)
        if is_r:
            reasoning_cost += cost; reasoning_wh += wh
        else:
            other_cost += cost; other_wh += wh

    total_cost, total_wh = reasoning_cost + other_cost, reasoning_wh + other_wh
    non_reasoning_n = total_n - reasoning_n
    avg_reasoning_cost = reasoning_cost / reasoning_n if reasoning_n else 0.0
    avg_reasoning_wh = reasoning_wh / reasoning_n if reasoning_n else 0.0
    avg_other_cost = other_cost / non_reasoning_n if non_reasoning_n else 0.0
    avg_other_wh = other_wh / non_reasoning_n if non_reasoning_n else 0.0

    # Routing rule: cap reasoning traffic at cap_frac of daily requests; route the
    # excess to a non-reasoning response on the same tier instead (chain-of-thought
    # inflates output tokens ~6x in this fleet, on top of the 80x energy multiplier).
    excess_n = max(0, reasoning_n - int(total_n * cap_frac))
    usd_saved = excess_n * max(0.0, avg_reasoning_cost - avg_other_cost)
    wh_saved = excess_n * max(0.0, avg_reasoning_wh - avg_other_wh)

    return {
        "reasoning_share_of_requests": round(reasoning_n / total_n, 3) if total_n else 0.0,
        "reasoning_share_of_energy": round(reasoning_wh / total_wh, 3) if total_wh else 0.0,
        "reasoning_cost_per_day": round(reasoning_cost, 2),
        "reasoning_wh_per_day": round(reasoning_wh, 1),
        "cap_frac": cap_frac, "requests_over_cap": excess_n,
        "usd_saved_per_day": round(usd_saved, 3), "usd_saved_per_month": round(usd_saved * DAYS, 2),
        "kwh_saved_per_month": round(wh_saved * DAYS / 1000.0, 2),
    }


# 5. Carbon-aware scheduling for shiftable/interruptible jobs -------------------------

def carbon_aware_scheduling(current_region: str = "us-east-1") -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    best_region = sustainability.carbon_aware_region_rank()[0][0]

    shifts = []
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        gtype, ngpu = j["gpu_type"], int(num(j["num_gpus"]))
        hpd, days = num(j["hours_per_day"]), num(j["days"])
        kwh = (num(cat[gtype]["watts"]) / 1000.0) * hpd * days * ngpu
        shift = sustainability.region_shift_savings(kwh, current_region, best_region)
        shifts.append({"job_id": j["job_id"], "kwh": round(kwh, 1), **shift})

    return {"current_region": current_region, "best_region": best_region, "shifts": shifts,
            "usd_saved": round(sum(s["usd_saved"] for s in shifts), 2),
            "kg_co2_saved": round(sum(s["gco2_saved"] for s in shifts) / 1000.0, 2)}


def build_markdown(t1, t2, t3, t4, t5) -> str:
    lines = ["# NimbusAI — \"Your Turn\" Extensions (M6, optional)", ""]

    lines += ["## 1. Tier policy v2 — interruption rate + 1yr vs 3yr reserved", "",
              f"{t1['changed']}/{len(t1['rows'])} job recommendations change vs. `recommend_tier()`:", "",
              "| job | v1 | v2 | reason |", "|---|---|---|---|"]
    for r in t1["rows"]:
        mark = " **(changed)**" if r["changed"] else ""
        lines.append(f"| {r['job_id']}{mark} | {r['v1_tier']} | {r['v2_tier']} | {r['reason']} |")

    lines += ["", "## 2. Right-sizing memory-bound inference GPUs by $/GB-VRAM", ""]
    if t2["consolidations"]:
        lines.append(f"Consolidation opportunity found: **${t2['monthly_savings']:,.0f}/month**")
        lines += ["", "| from | to | before | after | savings |", "|---|---|---|---|---|"]
        for c in t2["consolidations"]:
            lines.append(f"| {c['from_count']}x {c['from_type']} ({', '.join(c['gpu_ids'])}) | "
                        f"{c['to_count']}x {c['to_type']} | ${c['monthly_before']:,.0f} | "
                        f"${c['monthly_after']:,.0f} | ${c['monthly_savings']:,.0f} |")
    else:
        lines.append("No consolidation opportunity beats the current fleet on $/GB-VRAM.")

    lines += ["", "## 3. Cache economics — is caching worth it per segment?", "",
              f"Break-even reuse rate (write overhead amortized): **{t3['breakeven_hit_frac']:.1%}**", "",
              "| team | project | requests | reuse rate | worth it? | cost/day | cost share |",
              "|---|---|---|---|---|---|---|"]
    for s in sorted(t3["segments"], key=lambda x: -x["cost_share"]):
        lines.append(f"| {s['team']} | {s['project']} | {s['requests']} | {s['reuse_rate']:.1%} | "
                    f"{'yes' if s['worth_it'] else 'NO'} | ${s['cost_per_day']:.2f} | {s['cost_share']:.1%} |")
    lines.append(f"\n**{t3['flagged_cost_share']:.0%}** of daily inference cost sits in segments below "
                f"break-even reuse — caching isn't earning back its write overhead there; disable it "
                f"or investigate why reuse is low before keeping it on.")

    lines += ["", "## 4. Reasoning budget — $ and Wh, capped at "
              f"{t4['cap_frac']:.0%} of requests", "",
              f"- Reasoning is **{t4['reasoning_share_of_requests']:.1%}** of requests but "
              f"**{t4['reasoning_share_of_energy']:.1%}** of energy (Wh) — the 80x multiplier dominates.",
              f"- Reasoning costs **${t4['reasoning_cost_per_day']:.2f}/day**, "
              f"**{t4['reasoning_wh_per_day']:,.0f} Wh/day**.",
              f"- Routing rule: cap reasoning at {t4['cap_frac']:.0%} of requests, reroute the "
              f"{t4['requests_over_cap']} excess/day to a non-reasoning response on the same tier "
              f"→ **${t4['usd_saved_per_month']:,.2f}/month** + **{t4['kwh_saved_per_month']:,.1f} kWh/month** saved."]

    lines += ["", "## 5. Carbon-aware scheduling for interruptible training jobs", "",
              f"Best blended cost+carbon region: **{t5['best_region']}** (vs. baseline `{t5['current_region']}`)",
              "", "| job | kWh | $ saved | gCO2e saved |", "|---|---|---|---|"]
    for s in t5["shifts"]:
        lines.append(f"| {s['job_id']} | {s['kwh']:,.0f} | ${s['usd_saved']:,.2f} | {s['gco2_saved']:,.0f} |")
    lines.append(f"\n**Total: ${t5['usd_saved']:,.2f} + {t5['kg_co2_saved']:,.1f} kg CO2e** saved by "
                f"scheduling these jobs into {t5['best_region']} instead of {t5['current_region']}.")

    lines += ["", "_All figures are June-2026 as-of snapshots on synthetic data; re-baseline before acting._"]
    return "\n".join(lines)


def run(verbose: bool = True) -> dict:
    t1, t2, t3, t4, t5 = (tier_policy_v2(), right_size_memory_bound(), cache_economics(),
                          reasoning_budget(), carbon_aware_scheduling())
    md = build_markdown(t1, t2, t3, t4, t5)

    out_md = os.path.join(ROOT, "outputs", "your_turn.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    if verbose:
        print("== M6 Your Turn Extensions ==")
        print(md)
        print("\nWritten: outputs/your_turn.md")

    return {"tier_policy_v2": t1, "right_size_memory_bound": t2, "cache_economics": t3,
            "reasoning_budget": t4, "carbon_aware_scheduling": t5}


if __name__ == "__main__":
    run()
