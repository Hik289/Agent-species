"""Backfill ALFWorld (5 tasks) for specific haiku-only genotypes (orchestrator 2026-08-03).

Parses each genotype signature back to a genotype dict, runs it on ALFWorld under the same
same-X protocol, and writes {sig: {SR, cost, passes, n}} so the haiku-tier full-20 frontier
can be recomputed. haiku-only → quota-safe (no opus/sonnet multi-turn 429).
"""
import sys, json, re, time
sys.path.insert(0, ".")
from phase2 import genotype as GT
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def sig_to_genotype(sig):
    parts = sig.split()
    topo = parts[0]
    roles = {}
    for p in parts[1:]:
        r, spec = p.split("=", 1)
        model, harness = spec.rsplit("/", 1)
        model = model.replace("c-", "claude-")
        roles[r] = {"model": model, "harness": harness, "params": {}}
    return {"roles": roles, "topology": topo, "topology_params": {}, "_lineage_id": "backfill"}

def main():
    sigs = json.load(open("./results/haiku_backfill_genotypes.json"))
    sigs = [x["sig"] for x in sigs]
    print(f"backfilling ALFWorld for {len(sigs)} haiku genotypes", flush=True)

    protocol = SameXProtocol(max_turns=30, token_budget=100000)
    enforcer = SameXEnforcer(protocol)
    # separate tracker + high cap: the shared budget_tracker.json shows ~$472 cumulative
    # (cumulative across the whole project / race-inflated); reusing it with a $200 cap
    # blocks ALL calls (client refuses when over cap → 0 tokens, the earlier all-$0 bug).
    client = LLMClient(budget_usd_hard_cap=100.0,
                       tracker_path="./results/backfill_tracker.json")

    from phase0.eval_harness import get_tasks, make_task
    from phase2.agent_exec import run_genotype_on_task
    specs = get_tasks("alfworld", "valid_seen", 5, seed=0, max_steps=30)

    out = {}
    for i, sig in enumerate(sigs):
        g = sig_to_genotype(sig)
        p = n = 0; cost = 0.0
        for j, spec in enumerate(specs):
            try:
                task = make_task(spec)
                r = run_genotype_on_task(g, task, client, protocol=protocol,
                                         enforcer=enforcer, purpose=f"backfill_{i}_alf_t{j}")
                p += r["success"]; n += 1; cost += r["cost_usd"]
            except Exception as ex:
                print(f"  [{i}] {sig[:40]} task{j} ERR {str(ex)[:80]}", flush=True)
                n += 1
        out[sig] = {"SR": round(p/n, 4) if n else 0, "passes": p, "n": n, "cost": round(cost, 6)}
        print(f"  [{i+1}/{len(sigs)}] {sig[:45]} → SR={out[sig]['SR']} cost=${out[sig]['cost']:.4f}", flush=True)
        json.dump(out, open("./results/haiku_alfworld_backfill.json", "w"), indent=2)
    print("BACKFILL DONE", flush=True)

if __name__ == "__main__":
    main()
