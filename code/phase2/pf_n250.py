"""Evaluate a planner–executor pair on seen and unseen ALFWorld tasks.

The executor receives the full plan and recent action outcomes. Saved results
include each task's outcome, cost, and seen or unseen split.

Arguments are the agent setup name and task count."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.pipeline_fix import run_pipeline_fixed, single, PIPELINES, SINGLES, _task_key
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def main():
    GENO = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    # pooled held-out: valid_seen (excl selection 5) + valid_unseen
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    vs = get_tasks("alfworld","valid_seen",400,seed=0,max_steps=30)
    seen = [t for t in vs if _task_key(t) not in sel_files]
    # dedup seen
    seenk=set(); seen2=[]
    for t in seen:
        if _task_key(t) not in seenk: seenk.add(_task_key(t)); seen2.append(t)
    vu = get_tasks("alfworld","valid_unseen",400,seed=0,max_steps=30)
    unk=set(); unseen=[]
    for t in vu:
        if _task_key(t) not in unk: unk.add(_task_key(t)); unseen.append(t)
    # build pooled list with split labels: all seen held-out + fill rest from unseen
    pooled = [(t,"seen") for t in seen2] + [(t,"unseen") for t in unseen]
    pooled = pooled[:N]
    n = len(pooled)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    tag=GENO.replace("->","_")
    client=LLMClient(budget_usd_hard_cap=500.0,tracker_path=f"./results/n250_{tag}_tracker.json")
    RES=f"./results/n250_{tag}.json"
    print(f"[setup] n250 {GENO}: {n} tasks ({sum(1 for _,s in pooled if s=='seen')} seen + {sum(1 for _,s in pooled if s=='unseen')} unseen)",flush=True)
    is_pipe=GENO in PIPELINES
    passes=0;cost=0.0;pt=[];ptc=[];splits=[]
    for j,(spec,split) in enumerate(pooled):
        task=make_task(spec)
        try:
            if is_pipe:
                pl,ex=PIPELINES[GENO]
                r=run_pipeline_fixed(task,{"model":pl,"harness":"react"},{"model":ex,"harness":"react"},client,proto,enf.new_task_meter(),f"n250_{tag}_{j}","A")
            else:
                r=run_genotype_on_task(single(SINGLES[GENO]),task,client,protocol=proto,enforcer=enf,purpose=f"n250_{tag}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5));splits.append(split)
        except Exception as ex:
            print(f"  {GENO} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0);splits.append(split)
        if j%15==0: json.dump({"geno":GENO,"n_done":j+1,"passes":passes,"cost":round(cost,3)},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
    # seen/unseen split SR
    def split_sr(sp):
        idx=[i for i,s in enumerate(splits) if s==sp]
        if not idx: return None,0
        p=sum(pt[i] for i in idx); return p/len(idx), len(idx)
    ssr,sn=split_sr("seen"); usr,un=split_sr("unseen")
    json.dump({"geno":GENO,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"CoP":round(cop,4) if cop else None,
               "seen_SR":round(ssr,4) if ssr is not None else None,"seen_n":sn,
               "unseen_SR":round(usr,4) if usr is not None else None,"unseen_n":un,
               "per_task":pt,"per_task_cost":ptc,"splits":splits},open(RES,"w"),indent=2)
    print(f"[n250 {GENO}] SR={sr:.3f}+-{se:.3f} seen={ssr} unseen={usr} CoP=${cop if cop else 0:.3f}",flush=True)
    print(f"N250_{tag} DONE",flush=True)

if __name__=="__main__": main()
