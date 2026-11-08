"""Compare single GPT and Claude agents on retail tasks.

The bridge uses the setup's primary model and execution method. These runs
measure single-agent success and cost on the retail benchmark."""
import sys, json, math, random
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def single(model):
    return {"roles": {"solver": {"model": model, "harness": "react", "params": {}}},
            "topology": "single", "topology_params": {}, "_lineage_id": "tau3cv"}

GENOTYPES = {"single-GPT": single("gpt-5.5"),
             "single-opus": single("claude-opus-5"),
             "single-sonnet": single("claude-sonnet-5")}

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    DOMAIN = "retail"
    specs = get_tasks("taubench", DOMAIN, N, seed=0)  # tau3 retail, n tasks
    print(f"[setup] tau3 single cross-vendor premise check: {len(specs)} tasks, domain={DOMAIN}", flush=True)
    proto = SameXProtocol(max_turns=30, token_budget=100000)
    enf = SameXEnforcer(proto)
    client = LLMClient(budget_usd_hard_cap=200.0,
                       tracker_path="./results/tau3cv_tracker.json")
    RES = "./results/tau3cv_heldout.json"
    results = {"config": {"benchmark": "tau3-retail", "n": len(specs),
                          "note": "SINGLE models only (tau3 bridge = primary role, no true pipeline); tests GPT cheap-capable premise generalization"},
               "legs": {}}
    for name, g in GENOTYPES.items():
        attempt = 0
        while True:
            attempt += 1
            passes = 0; cost = 0.0; pt = []; ptc = []
            for j, spec in enumerate(specs):
                task = make_task(spec)
                try:
                    r = run_genotype_on_task(g, task, client, protocol=proto, enforcer=enf,
                                             purpose=f"t3cv_{name}_{j}")
                    passes += r["success"]; cost += r["cost_usd"]
                    pt.append(1 if r["success"] else 0); ptc.append(round(r["cost_usd"], 5))
                except Exception as ex:
                    print(f"  {name} t{j} ERR {str(ex)[:45]}", flush=True)
                    pt.append(0); ptc.append(0.0)
            n = len(specs); zero = sum(1 for x in ptc if x == 0)
            all_zero = (cost == 0) or (zero == n and n > 0)
            if not all_zero or attempt > 2:
                break
            print(f"  [RETRY {attempt}] {name}: all-$0 (infra artifact), re-running", flush=True)
        sr = passes/n; se = math.sqrt(sr*(1-sr)/n) if n else 0; cop = cost/passes if passes else None
        invalid = (cost == 0) or (zero == n and n > 0)
        if invalid: print(f"  [INVALID] {name}: all-$0 EXCLUDED", flush=True)
        results["legs"][name] = {"SR": round(sr,4), "SE": round(se,4), "cost": round(cost,4),
                                 "CoP": round(cop,4) if cop else None, "n": n,
                                 "per_task": pt, "per_task_cost": ptc, "zero_cost_tasks": zero,
                                 "invalid_zero_cost": invalid}
        print(f"  [{name}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}", flush=True)
        json.dump(results, open(RES, "w"), indent=2)
    print("TAU3CV DONE", flush=True)

if __name__ == "__main__":
    main()
