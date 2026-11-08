"""Re-evaluate selected single-model and multi-role agents on held-out tasks.

The runner excludes the ALFWorld tasks used for selection and evaluates the
selected agents on the next task set. It saves success and cost for each setup
and summaries for the two groups."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from phase2.genotype import genotype_signature
from core.llm_client import LLMClient

MODEL_MAP = {"c-opus-5":"claude-opus-5","c-opus-4-8":"claude-opus-4-8",
             "c-sonnet-5":"claude-sonnet-5","c-sonnet-4-6":"claude-sonnet-4-6",
             "c-haiku-4-5-20251001":"claude-haiku-4-5-20251001"}
TOPO_ROLES = {"single":["solver"],"pipeline":["planner","executor"],
              "router":["router","coder","planner"],"debate":["a","b","judge"]}

def sig_to_genotype(sig):
    parts = sig.split()
    topo = parts[0]
    roles = {}
    for p in parts[1:]:
        role, spec = p.split("=", 1)
        m, h = spec.rsplit("/", 1)
        roles[role] = {"model": MODEL_MAP[m], "harness": h, "params": {}}
    return {"roles": roles, "topology": topo, "topology_params": {}, "_lineage_id": "highn"}

def _task_key(t):
    # alfworld tuple = (name, game_file, max_steps); game_file unique. Other benches: use idx.
    return t[1] if (isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], str)) else str(t)

def main():
    BENCH = sys.argv[1] if len(sys.argv) > 1 else "alfworld"
    N_HELDOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    DOMAIN = {"alfworld":"valid_seen","taubench":"retail"}.get(BENCH,"valid_seen")
    sel = json.load(open("./results/highn_genotypes.json"))
    # held-out tasks: get big pool, drop the selection-set 5 (by task key)
    sel5 = get_tasks(BENCH,DOMAIN,5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks(BENCH,DOMAIN,5+N_HELDOUT+8,seed=0,max_steps=30)
    heldout = [t for t in big if _task_key(t) not in sel_files][:N_HELDOUT]
    print(f"[setup] bench={BENCH} selection={len(sel_files)} files, held-out={len(heldout)} DISTINCT tasks", flush=True)
    if len(heldout) < N_HELDOUT:
        print(f"[setup] WARN only {len(heldout)} held-out available (pool limited) — using all, n marked", flush=True)
        N_HELDOUT = len(heldout)
    assert N_HELDOUT >= 1, "no held-out tasks"
    assert not (set(_task_key(t) for t in heldout) & sel_files), "held-out leaks selection!"

    proto = SameXProtocol(max_turns=30, token_budget=100000)
    enf = SameXEnforcer(proto)
    client = LLMClient(budget_usd_hard_cap=250.0,
                       tracker_path=f"./results/highn_{BENCH}_tracker.json")
    results = {"config":{"benchmark":BENCH,"n_heldout":N_HELDOUT,"selection_set":f"{BENCH} {DOMAIN} first5 seed0",
                         "note":"winner-curse-safe: eval on held-out tasks NOT in selection set"},
               "comp":{}, "single":{}}
    MAX_RETRY = 2  # retry a genotype if it shows the transient-$0 reverse-artifact
    for group in ["comp","single"]:
        for sig in sel[group]:
            g = sig_to_genotype(sig)
            attempt = 0
            while True:
                attempt += 1
                passes = 0; n = 0; cost = 0.0; per_task = []; per_task_cost = []; zero_cost_tasks = 0
                for j, spec in enumerate(heldout):
                    try:
                        task = make_task(spec)
                        r = run_genotype_on_task(g, task, client, protocol=proto,
                                                 enforcer=enf, purpose=f"highn_{group}_{j}")
                        passes += r["success"]; n += 1; cost += r["cost_usd"]
                        per_task.append(1 if r["success"] else 0)
                        per_task_cost.append(round(r["cost_usd"], 5))  # per-task cost for bootstrap CoP error bars
                        if r["cost_usd"] == 0:
                            zero_cost_tasks += 1
                    except Exception as ex:
                        print(f"  ERR {group} {sig[:30]} t{j}: {str(ex)[:50]}", flush=True)
                        n += 1; per_task.append(0); per_task_cost.append(0.0); zero_cost_tasks += 1
                # ANTI-REVERSE-ARTIFACT (orchestrator 2026-08-08): a genotype with $0 total cost,
                # or where EVERY task had $0 cost (never actually ran → transient proxy/exec
                # failure silently swallowed by agent_exec's `except: break`), is a FALSE 0 —
                # NOT a real SR. Must NOT enter the decisive comp-vs-single verdict. Retry;
                # if still all-zero after retries, mark INVALID (excluded, flagged).
                all_zero = (cost == 0) or (zero_cost_tasks == n and n > 0)
                if not all_zero or attempt > MAX_RETRY:
                    break
                print(f"  [RETRY {attempt}] {group} {sig[:40]}: all-$0 (transient exec fail), "
                      f"re-running (not a real 0)", flush=True)
            sr = passes/n if n else 0
            se = math.sqrt(sr*(1-sr)/n) if n else 0  # binomial SE
            invalid = (cost == 0) or (zero_cost_tasks == n and n > 0)
            if invalid:
                print(f"  [INVALID] {group} {sig[:40]}: still all-$0 after {attempt} attempts "
                      f"→ EXCLUDED from verdict (transient exec failure, NOT real SR)", flush=True)
            results[group][sig] = {"SR":round(sr,4),"passes":passes,"n":n,
                                   "invalid_zero_cost": invalid, "zero_cost_tasks": zero_cost_tasks,
                                   "SE":round(se,4),"cost":round(cost,4),"per_task":per_task,"per_task_cost":per_task_cost}
            print(f"  [{group}] SR={sr:.3f}±{se:.3f} ({passes}/{n}) ${cost:.2f}  {sig[:50]}", flush=True)
            json.dump(results, open(f"./results/highn_{BENCH}_heldout.json","w"), indent=2)
    # group-level aggregate — EXCLUDE invalid ($0/transient-fail) genotypes from the verdict.
    # orchestrator 2026-08-08: the decisive comparison is GROUP-BEST vs GROUP-BEST (each group's
    # strongest genotype on held-out), NOT reflexion-only (using the weakest single = the
    # mirror of winner's curse / weak-single artifact). Report BOTH group-best and pooled.
    for group in ["comp","single"]:
        valid = {s:v for s,v in results[group].items() if not v.get("invalid_zero_cost")}
        excluded = [s for s,v in results[group].items() if v.get("invalid_zero_cost")]
        srs = [v["SR"] for v in valid.values()]
        allp = [x for v in valid.values() for x in v["per_task"]]
        pooled = sum(allp)/len(allp) if allp else 0
        pse = math.sqrt(pooled*(1-pooled)/len(allp)) if allp else 0
        # GROUP-BEST: the strongest valid genotype (by held-out SR), with its own binomial SE
        best_sig, best = (None, None)
        for s, v in valid.items():
            if best is None or v["SR"] > best["SR"]:
                best_sig, best = s, v
        best_sr = best["SR"] if best else 0
        best_se = best.get("SE", 0) if best else 0
        results[group+"_summary"] = {"mean_of_genotype_SR":round(sum(srs)/len(srs),4) if srs else None,
                                     "pooled_SR":round(pooled,4),"pooled_SE":round(pse,4),
                                     "pooled_n":len(allp),"n_valid_genotypes":len(valid),
                                     "excluded_invalid":excluded,
                                     "best_genotype":best_sig,"best_SR":best_sr,"best_SE":best_se}
        if excluded:
            print(f"GROUP {group}: EXCLUDED {len(excluded)} invalid($0) genotypes: {excluded}", flush=True)
        print(f"GROUP {group}: BEST={best_sr:.3f}±{best_se:.3f} ({best_sig[:40] if best_sig else None}) | "
              f"pooled={pooled:.3f}±{pse:.3f} (valid_genotypes={len(valid)})", flush=True)
    cg=results["comp_summary"]; sg=results["single_summary"]
    # DECISIVE verdict = GROUP-BEST vs GROUP-BEST (orchestrator-specified)
    bgap = cg["best_SR"]-sg["best_SR"]
    bcombined_se = math.sqrt(cg["best_SE"]**2 + sg["best_SE"]**2)
    pgap = cg["pooled_SR"]-sg["pooled_SR"]
    pcombined_se = math.sqrt(cg["pooled_SE"]**2 + sg["pooled_SE"]**2)
    results["verdict"] = {
        "PRIMARY_group_best": {"comp_best":cg["best_SR"],"single_best":sg["best_SR"],
            "gap":round(bgap,4),"combined_SE":round(bcombined_se,4),
            "gap_over_2SE":round(bgap/(2*bcombined_se),2) if bcombined_se else None,
            "significant_95":abs(bgap) > 2*bcombined_se},
        "secondary_pooled": {"comp_pooled":cg["pooled_SR"],"single_pooled":sg["pooled_SR"],
            "gap":round(pgap,4),"combined_SE":round(pcombined_se,4),
            "significant_95":abs(pgap) > 2*pcombined_se}}
    json.dump(results, open(f"./results/highn_{BENCH}_heldout.json","w"), indent=2)
    print(f"\nVERDICT [{BENCH}] PRIMARY group-best: comp {cg['best_SR']:.3f}±{cg['best_SE']:.3f} vs "
          f"single {sg['best_SR']:.3f}±{sg['best_SE']:.3f} gap={bgap:+.3f} (±{bcombined_se:.3f}); "
          f"significant@95%={results['verdict']['PRIMARY_group_best']['significant_95']}", flush=True)
    print(f"VERDICT [{BENCH}] secondary pooled: comp {cg['pooled_SR']:.3f} vs single {sg['pooled_SR']:.3f} "
          f"gap={pgap:+.3f}", flush=True)
    print("HIGHN DONE", flush=True)

if __name__ == "__main__":
    main()
