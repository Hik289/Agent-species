"""Evaluate a sequence of models that retry tasks after a failure.

Each model attempts the task in turn until the environment reports success
or all models have been tried. Total cost includes every attempt. The runner
compares these sequences with single-model agents on the held-out tasks."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_interactive
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

CASCADE_TIERS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-5"]
HARNESS = "react"  # fixed harness across tiers (isolate the cascade mechanism)

def _task_key(t):
    return t[1] if (isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], str)) else str(t)

def main():
    N_HELDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sel5 = get_tasks("alfworld", "valid_seen", 5, seed=0, max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld", "valid_seen", 5 + N_HELDOUT + 8, seed=0, max_steps=30)
    heldout = [t for t in big if _task_key(t) not in sel_files][:N_HELDOUT]
    assert not (set(_task_key(t) for t in heldout) & sel_files), "held-out leaks selection!"
    print(f"[setup] cascade on {len(heldout)} held-out ALFWorld tasks; tiers={CASCADE_TIERS}", flush=True)

    proto = SameXProtocol(max_turns=30, token_budget=100000)
    enf = SameXEnforcer(proto)
    client = LLMClient(budget_usd_hard_cap=250.0,
                       tracker_path="./results/cascade_tracker.json")

    results = {"config": {"mechanism": "test-time confidence cascade (escalate on non-success)",
                          "tiers": CASCADE_TIERS, "harness": HARNESS, "n_heldout": len(heldout),
                          "note": "gate uses env success signal only, no gold peeking; cost=sum all tiers"},
               "per_task": []}
    casc_pass = 0; casc_cost = 0.0; casc_pt = []; casc_pt_cost = []
    tier_stops = {m: 0 for m in CASCADE_TIERS}  # where the cascade stopped (success tier or last)
    for j, spec in enumerate(heldout):
        task_cost = 0.0; solved = 0; stop_tier = None
        for tier in CASCADE_TIERS:
            task = make_task(spec)  # fresh env per attempt
            meter = enf.new_task_meter()  # same-X token/turn budget, fresh per attempt
            try:
                r = run_interactive(task, {"model": tier, "harness": HARNESS, "params": {}},
                                    client, proto, meter,
                                    purpose=f"casc_{j}_{tier.split('-')[1]}")
                task_cost += r["cost_usd"]
                if r["success"]:
                    solved = 1; stop_tier = tier
                    break
            except Exception as ex:
                print(f"  task{j} {tier} ERR {str(ex)[:40]}", flush=True)
        if stop_tier is None:
            stop_tier = CASCADE_TIERS[-1]
        tier_stops[stop_tier] += 1
        casc_pass += solved; casc_cost += task_cost
        casc_pt.append(solved); casc_pt_cost.append(round(task_cost, 5))
        results["per_task"].append({"task": j, "solved": solved, "cost": round(task_cost, 5),
                                    "stop_tier": stop_tier})
        print(f"  task{j}: solved={solved} cost=${task_cost:.3f} stop@{stop_tier.split('-')[1]}", flush=True)
        json.dump(results, open("./results/cascade_heldout.json", "w"), indent=2)
    n = len(heldout)
    sr = casc_pass / n
    se = math.sqrt(sr * (1 - sr) / n) if n else 0
    cop = casc_cost / casc_pass if casc_pass else None
    results["summary"] = {"SR": round(sr, 4), "SE": round(se, 4), "total_cost": round(casc_cost, 4),
                          "cost_per_episode": round(casc_cost / n, 5),
                          "CoP": round(cop, 4) if cop else None,
                          "tier_stops": tier_stops, "per_task_cost": casc_pt_cost, "per_task": casc_pt}
    json.dump(results, open("./results/cascade_heldout.json", "w"), indent=2)
    print(f"\nCASCADE: SR={sr:.3f}±{se:.3f} cost/ep=${casc_cost/n:.3f} CoP=${cop if cop else 0:.3f}", flush=True)
    print(f"  tier stops: {tier_stops}", flush=True)
    print("CASCADE DONE", flush=True)

if __name__ == "__main__":
    main()
