"""Compare single-model and planner–executor agents on ScienceWorld.

The runner uses the shared execution code with the ScienceWorld task wrapper.

Arguments are the agent setup name and task count."""
import sys, json, math
sys.path.insert(0,".")
from phase0.scienceworld_bench import get_scienceworld_tasks, make_sw_task
from phase2.pipeline_fix import run_pipeline_fixed, single, PIPELINES, SINGLES
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def main():
    GENO=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 50
    specs=get_scienceworld_tasks(N, seed=0, max_steps=30); n=len(specs)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    tag=GENO.replace("->","_")
    client=LLMClient(budget_usd_hard_cap=400.0,tracker_path=f"./results/sw_{tag}_tracker.json")
    RES=f"./results/sw_{tag}.json"
    is_pipe=GENO in PIPELINES
    print(f"[setup] scienceworld {GENO} n={n} pipe={is_pipe}",flush=True)
    passes=0;cost=0.0;pt=[];ptc=[];scores=[]
    for j,spec in enumerate(specs):
        task=make_sw_task(spec)
        try:
            if is_pipe:
                pl,ex=PIPELINES[GENO]
                r=run_pipeline_fixed(task,{"model":pl,"harness":"react"},{"model":ex,"harness":"react"},client,proto,enf.new_task_meter(),f"sw_{tag}_{j}","A")
            else:
                r=run_genotype_on_task(single(SINGLES[GENO]),task,client,protocol=proto,enforcer=enf,purpose=f"sw_{tag}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
            scores.append(getattr(task,"_best_score",0))
        except Exception as ex:
            print(f"  {GENO} t{j} ERR {str(ex)[:60]}",flush=True);pt.append(0);ptc.append(0.0);scores.append(0)
        if j%5==0: json.dump({"geno":GENO,"benchmark":"scienceworld","n_done":j+1,"passes":passes,"cost":round(cost,3),"per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    sr=passes/n if n else 0;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
    mean_score=(sum(scores)/len(scores)/100.0) if scores else 0  # normalized partial credit 0-1
    json.dump({"geno":GENO,"benchmark":"scienceworld","n":n,"SR":round(sr,4),"SE":round(se,4),
               "partial_score":round(mean_score,4),"per_task_score":scores,
               "cost":round(cost,4),"CoP":round(cop,4) if cop else None,"passes":passes,
               "per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    print(f"[SW {GENO}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f}",flush=True); print(f"SW_{tag} DONE",flush=True)

if __name__=="__main__": main()
