import sys, json
sys.path.insert(0,".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient
proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
client=LLMClient(budget_usd_hard_cap=50.0,tracker_path="./results/mtdiffers_tracker.json")
specs=get_tasks("alfworld","valid_seen",8,seed=0,max_steps=30)  # 8 tasks for a bit more power
HARNESSES=["direct","cot","react","reflexion"]
out={}
for h in HARNESSES:
    g={"roles":{"solver":{"model":"claude-haiku-4-5-20251001","harness":h,"params":{}}},"topology":"single","topology_params":{},"_lineage_id":f"differs_{h}"}
    p=n=0;cost=0.0
    for j,spec in enumerate(specs):
        try:
            task=make_task(spec)
            r=run_genotype_on_task(g,task,client,protocol=proto,enforcer=enf,purpose=f"mtdiff_{h}_t{j}")
            p+=r["success"];n+=1;cost+=r["cost_usd"]
        except Exception as ex:
            print(f"  {h} t{j} ERR {str(ex)[:60]}",flush=True); n+=1
    out[h]={"SR":round(p/n,4) if n else 0,"passes":p,"n":n,"cost":round(cost,4)}
    print(f"  {h}: SR={out[h]['SR']} cost=${out[h]['cost']}",flush=True)
    json.dump(out,open("./results/mt_harness_differs.json","w"),indent=2)
srs=[v['SR'] for v in out.values()]
print(f"MT HARNESS-DIFFERS DONE: spread={max(srs)-min(srs):.3f} {[(h,out[h]['SR']) for h in HARNESSES]}",flush=True)
