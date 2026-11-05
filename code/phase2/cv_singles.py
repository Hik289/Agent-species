import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient
from phase2.crossvendor_heldout import single, _task_key

SINGLES = {"single-GPT": single("gpt-5.5"), "single-opus": single("claude-opus-5"),
           "single-sonnet": single("claude-sonnet-5")}

def main():
    N = 20
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld","valid_seen",5+N+8,seed=0,max_steps=30)
    heldout=[t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files)
    print(f"[setup] cv singles on {len(heldout)} held-out tasks",flush=True)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    client=LLMClient(budget_usd_hard_cap=250.0,tracker_path="./results/crossvendor_tracker.json")
    # load existing 7-leg results to merge
    RES="./results/crossvendor_heldout.json"
    results=json.load(open(RES))
    for name,g in SINGLES.items():
        passes=0;cost=0.0;pt=[];ptc=[]
        for j,spec in enumerate(heldout):
            task=make_task(spec)
            try:
                r=run_genotype_on_task(g,task,client,protocol=proto,enforcer=enf,purpose=f"cvs_{name}_{j}")
                passes+=r["success"];cost+=r["cost_usd"];pt.append(1 if r["success"] else 0);ptc.append(round(r["cost_usd"],5))
            except Exception as ex:
                print(f"  {name} t{j} ERR {str(ex)[:45]}",flush=True);pt.append(0);ptc.append(0.0)
        n=len(heldout);sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
        results["legs"][name]={"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"CoP":round(cop,4) if cop else None,"n":n,"per_task":pt,"per_task_cost":ptc,"zero_cost_tasks":sum(1 for x in ptc if x==0)}
        print(f"  [{name}] SR={sr:.3f}±{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}",flush=True)
        json.dump(results,open(RES,"w"),indent=2)
    # complementarity (now with real single-GPT + single-opus)
    gv=results["legs"]["single-GPT"]["per_task"];cv=results["legs"]["single-opus"]["per_task"]
    g_only=sum(1 for a,b in zip(gv,cv) if a and not b);c_only=sum(1 for a,b in zip(gv,cv) if b and not a)
    both=sum(1 for a,b in zip(gv,cv) if a and b);neither=sum(1 for a,b in zip(gv,cv) if not a and not b)
    union=g_only+c_only+both
    results["complementarity"]={"gpt_only":g_only,"opus_only":c_only,"both":both,"neither":neither,"union_solvable":union,"best_single":max(sum(gv),sum(cv)),"headroom":union-max(sum(gv),sum(cv))}
    json.dump(results,open(RES,"w"),indent=2)
    print(f"\nCOMPLEMENTARITY: gpt_only={g_only} opus_only={c_only} both={both} neither={neither} union={union} best_single={max(sum(gv),sum(cv))} headroom={union-max(sum(gv),sum(cv))}",flush=True)
    print("CVSINGLES DONE",flush=True)

if __name__=="__main__": main()
