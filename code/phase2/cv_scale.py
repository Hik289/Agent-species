import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient
from phase2.crossvendor_heldout import pipe, single, _task_key

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
GENOTYPES = {
    "sonnet->GPT":   pipe("claude-sonnet-5", "gpt-5.5"),   # cheapest cross-vendor winner
    "opus->GPT":     pipe("claude-opus-5", "gpt-5.5"),      # cross-vendor
    "GPT->opus":     pipe("gpt-5.5", "claude-opus-5"),      # cross-vendor other dir
    "GPT->GPT":      pipe("gpt-5.5", "gpt-5.5"),            # same-vendor control
    "opus->sonnet":  pipe("claude-opus-5", "claude-sonnet-5"),  # same-vendor control
    "single-GPT":    single("gpt-5.5"),
    "single-opus":   single("claude-opus-5"),
    "single-sonnet": single("claude-sonnet-5"),
}

def main():
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld","valid_seen",5+N+20,seed=0,max_steps=30)
    heldout=[t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files)
    assert len(heldout)==N, f"only {len(heldout)} held-out"
    print(f"[setup] cv_scale n={N} held-out tasks, {len(GENOTYPES)} genotypes (SAME tasks all legs)",flush=True)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    client=LLMClient(budget_usd_hard_cap=400.0,tracker_path="./results/cvscale_tracker.json")
    RES="./results/cvscale_heldout.json"
    results={"config":{"n":N,"note":"n-scaled cross-vendor confirmation, same held-out tasks all legs, winner-curse-safe"},"legs":{}}
    for name,g in GENOTYPES.items():
        attempt=0
        while True:
            attempt+=1
            passes=0;cost=0.0;pt=[];ptc=[]
            for j,spec in enumerate(heldout):
                task=make_task(spec)
                try:
                    r=run_genotype_on_task(g,task,client,protocol=proto,enforcer=enf,purpose=f"cvs_{name.replace('->','_')}_{j}")
                    passes+=r["success"];cost+=r["cost_usd"];pt.append(1 if r["success"] else 0);ptc.append(round(r["cost_usd"],5))
                except Exception as ex:
                    print(f"  {name} t{j} ERR {str(ex)[:45]}",flush=True);pt.append(0);ptc.append(0.0)
            n=len(heldout);zero=sum(1 for x in ptc if x==0)
            # GUARD: all-$0 (disk-full / proxy-wedge artifact, NOT real) -> retry once then INVALID
            all_zero=(cost==0) or (zero==n and n>0)
            if not all_zero or attempt>2:
                break
            print(f"  [RETRY {attempt}] {name}: all-$0 (infra artifact, not real), re-running",flush=True)
        sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
        invalid=(cost==0) or (zero==n and n>0)
        if invalid: print(f"  [INVALID] {name}: all-$0 after {attempt} attempts -> EXCLUDED (infra artifact not real SR)",flush=True)
        results["legs"][name]={"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"CoP":round(cop,4) if cop else None,"n":n,"per_task":pt,"per_task_cost":ptc,"zero_cost_tasks":zero,"invalid_zero_cost":invalid}
        print(f"  [{name}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}",flush=True)
        json.dump(results,open(RES,"w"),indent=2)
    # complementarity across pairs
    def comp(a,b):
        av=results["legs"][a]["per_task"];bv=results["legs"][b]["per_task"]
        ao=sum(1 for x,y in zip(av,bv) if x and not y);bo=sum(1 for x,y in zip(av,bv) if y and not x)
        both=sum(1 for x,y in zip(av,bv) if x and y);un=ao+bo+both
        return {"a_only":ao,"b_only":bo,"both":both,"union":un,"best_single":max(sum(av),sum(bv)),"headroom":un-max(sum(av),sum(bv))}
    results["complementarity"]={"GPT_vs_opus":comp("single-GPT","single-opus"),"GPT_vs_sonnet":comp("single-GPT","single-sonnet")}
    json.dump(results,open(RES,"w"),indent=2)
    print(f"\nCOMPLEMENTARITY GPT_vs_opus: {results['complementarity']['GPT_vs_opus']}",flush=True)
    print("CVSCALE DONE",flush=True)

if __name__=="__main__": main()
