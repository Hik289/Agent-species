"""Evaluate one agent setup and write its results to a separate file.

Mode A keeps the full plan. Mode ABC also updates the plan during execution
and adds instructions for following it.

Arguments are the mode, agent setup name, and task count."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.pipeline_fix import run_pipeline_fixed, single, PIPELINES, SINGLES, _task_key
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

MODE = sys.argv[1]           # A or ABC
GENO = sys.argv[2]           # e.g. opus->opus or single-opus
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150

def main():
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld","valid_seen",5+N+40,seed=0,max_steps=30)
    heldout=[t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files)
    n=len(heldout)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    tag=GENO.replace("->","_")
    client=LLMClient(budget_usd_hard_cap=400.0,tracker_path=f"./results/pf_{MODE}_{tag}_tracker.json")
    RES=f"./results/pf_{MODE}_{tag}.json"
    print(f"[setup] {MODE} {GENO} n={n}",flush=True)
    is_pipe = GENO in PIPELINES
    passes=0;cost=0.0;pt=[];ptc=[]
    for j,spec in enumerate(heldout):
        task=make_task(spec)
        try:
            if is_pipe:
                pl,ex=PIPELINES[GENO]
                r=run_pipeline_fixed(task,{"model":pl,"harness":"react"},{"model":ex,"harness":"react"},client,proto,enf.new_task_meter(),f"pf_{tag}_{j}",MODE)
            else:
                r=run_genotype_on_task(single(SINGLES[GENO]),task,client,protocol=proto,enforcer=enf,purpose=f"pf_{tag}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
        except Exception as ex:
            print(f"  {GENO} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0)
        if j%10==0:
            json.dump({"geno":GENO,"mode":MODE,"n_done":j+1,"passes":passes,"cost":round(cost,3),"per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
    zero=sum(1 for x in ptc if x==0)
    json.dump({"geno":GENO,"mode":MODE,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),
               "CoP":round(cop,4) if cop else None,"per_task":pt,"per_task_cost":ptc,"zero_cost_tasks":zero,
               "invalid_zero_cost":(cost==0 or zero==n)},open(RES,"w"),indent=2)
    print(f"[{MODE} {GENO}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}",flush=True)
    print(f"PFONE_{MODE}_{tag} DONE",flush=True)

if __name__=="__main__": main()
