"""Compare GPT and Claude planner–executor pairs on held-out ALFWorld tasks.

The runner evaluates both directions of each selected pair, pairs from the
same provider, and single-model agents. It records task outcomes and costs
from both roles so success, spending, and task overlap can be compared."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

GPT = "gpt-5.5"
OPUS = "claude-opus-5"
SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

def pipe(planner, executor):
    return {"roles": {"planner": {"model": planner, "harness": "react", "params": {}},
                      "executor": {"model": executor, "harness": "react", "params": {}}},
            "topology": "pipeline", "topology_params": {}, "_lineage_id": "cv"}

def single(model):
    return {"roles": {"solver": {"model": model, "harness": "react", "params": {}}},
            "topology": "single", "topology_params": {}, "_lineage_id": "cv"}

GENOTYPES = {
    # cross-vendor
    "GPT->opus":      pipe(GPT, OPUS),
    "GPT->sonnet":    pipe(GPT, SONNET),
    "GPT->haiku":     pipe(GPT, HAIKU),
    "opus->GPT":      pipe(OPUS, GPT),
    "sonnet->GPT":    pipe(SONNET, GPT),
    # same-vendor controls
    "GPT->GPT":       pipe(GPT, GPT),
    "opus->sonnet":   pipe(OPUS, SONNET),
    # single baselines
    "single-GPT":     single(GPT),
    "single-opus":    single(OPUS),
    "single-sonnet":  single(SONNET),
}

def _task_key(t):
    return t[1] if (isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], str)) else str(t)

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sel5 = get_tasks("alfworld", "valid_seen", 5, seed=0, max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld", "valid_seen", 5 + N + 8, seed=0, max_steps=30)
    heldout = [t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files), "held-out leaks selection!"
    print(f"[setup] cross-vendor on {len(heldout)} held-out ALFWorld tasks; {len(GENOTYPES)} genotypes", flush=True)

    proto = SameXProtocol(max_turns=30, token_budget=100000)
    enf = SameXEnforcer(proto)
    client = LLMClient(budget_usd_hard_cap=250.0,
                       tracker_path="./results/crossvendor_tracker.json")
    results = {"config": {"mechanism": "GPT x Claude cross-vendor pipeline (complementarity test)",
                          "n_heldout": len(heldout), "note": "held-out, fair cost (both vendors), per-task solve vectors for complementarity"},
               "legs": {}}
    for name, g in GENOTYPES.items():
        passes = 0; cost = 0.0; pt = []; ptc = []
        for j, spec in enumerate(heldout):
            task = make_task(spec)
            try:
                r = run_genotype_on_task(g, task, client, protocol=proto, enforcer=enf,
                                         purpose=f"cv_{name.replace('->','_')}_{j}")
                passes += r["success"]; cost += r["cost_usd"]
                pt.append(1 if r["success"] else 0); ptc.append(round(r["cost_usd"], 5))
            except Exception as ex:
                print(f"  {name} t{j} ERR {str(ex)[:45]}", flush=True)
                pt.append(0); ptc.append(0.0)
        n = len(heldout); sr = passes/n
        se = math.sqrt(sr*(1-sr)/n) if n else 0
        cop = cost/passes if passes else None
        zero = sum(1 for x in ptc if x == 0)
        results["legs"][name] = {"SR": round(sr,4), "SE": round(se,4), "cost": round(cost,4),
                                 "CoP": round(cop,4) if cop else None, "n": n,
                                 "per_task": pt, "per_task_cost": ptc, "zero_cost_tasks": zero}
        print(f"  [{name}] SR={sr:.3f}±{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f} zero={zero}", flush=True)
        json.dump(results, open("./results/crossvendor_heldout.json", "w"), indent=2)

    # complementarity: GPT-solved vs Claude-solved task overlap (single baselines)
    if "single-GPT" in results["legs"] and "single-opus" in results["legs"]:
        gv = results["legs"]["single-GPT"]["per_task"]
        cv = results["legs"]["single-opus"]["per_task"]
        g_only = sum(1 for a,b in zip(gv,cv) if a and not b)
        c_only = sum(1 for a,b in zip(gv,cv) if b and not a)
        both = sum(1 for a,b in zip(gv,cv) if a and b)
        neither = sum(1 for a,b in zip(gv,cv) if not a and not b)
        union = g_only + c_only + both
        results["complementarity"] = {"gpt_only": g_only, "opus_only": c_only, "both": both,
                                      "neither": neither, "union_solvable": union,
                                      "note": "high g_only+c_only = complementary; union>max(single) = cross-vendor headroom"}
        print(f"\nCOMPLEMENTARITY (GPT vs opus single): gpt_only={g_only} opus_only={c_only} both={both} neither={neither}", flush=True)
        print(f"  union solvable={union} vs best-single={max(sum(gv),sum(cv))} → headroom={union-max(sum(gv),sum(cv))}", flush=True)
    json.dump(results, open("./results/crossvendor_heldout.json", "w"), indent=2)
    print("CROSSVENDOR DONE", flush=True)

if __name__ == "__main__":
    main()
