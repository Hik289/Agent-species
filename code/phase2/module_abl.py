"""Measure how plan content affects execution on ALFWorld.

The single setting uses the saved single-Opus baseline. noplan uses a two-role
workflow with an empty plan. fullplan uses an Opus planner and executor with
the planner's strategy included.

Arguments are the setting and task count."""
import sys, json, math
sys.path.insert(0,".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action2, _harness_sys, _build_msgs, run_genotype_on_task
from phase2.pipeline_fix import run_pipeline_fixed, single, _task_key
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def run_noplan(task, client, proto, meter, purpose):
    # pipeline scaffold, planner replaced with EMPTY plan (isolate structure vs content)
    obs,goal=task.reset(); cost=0.0; success=0; step=0; history=[]
    strat="(no strategy provided)"
    exec_sys=_harness_sys("react",proto,interactive=True)
    while meter.can_step():
        step+=1
        adm=obs["admissible_commands"]
        if not adm: break
        recent=("History (past actions and results):\n"+"\n".join(history[-30:])+"\n\n") if history else ""
        user=(f"GOAL: {goal}\nSTRATEGY:\n{strat}\n\n{recent}OBSERVATION: {obs['observation']}\nADMISSIBLE:\n"+"\n".join("- "+a for a in adm)+"\n\nYour command (reply 'ACTION: <admissible command>'):")
        try: text,c=_chat(client,"claude-opus-5",_build_msgs(exec_sys,user,"claude-opus-5"),meter,f"{purpose}_exec"); cost+=c
        except Exception: break
        action,_ok=_pick_action2(text,adm)
        obs,reward,done,info=task.step(action)
        res=(obs.get("observation","") or "")[:500]
        history.append(f"> {action} => {res}")
        if reward>0 or info.get("won"): success=1; break
        if done: break
    task.close()
    return {"success":success,"cost_usd":cost,"steps":step}

def main():
    CFG=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 135
    sel5=get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30); sel=set(_task_key(t) for t in sel5)
    big=get_tasks("alfworld","valid_seen",5+N+30,seed=0,max_steps=30)
    ho=[t for t in big if _task_key(t) not in sel][:N]; n=len(ho)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    client=LLMClient(budget_usd_hard_cap=300.0,tracker_path=f"./results/modabl_{CFG}_tracker.json")
    RES=f"./results/modabl_{CFG}.json"
    print(f"[setup] module-abl {CFG} n={n}",flush=True)
    passes=0;cost=0.0;pt=[];ptc=[]
    for j,spec in enumerate(ho):
        task=make_task(spec)
        try:
            if CFG=="noplan":
                r=run_noplan(task,client,proto,enf.new_task_meter(),f"mab_{CFG}_{j}")
            elif CFG=="fullplan":
                r=run_pipeline_fixed(task,{"model":"claude-opus-5","harness":"react"},{"model":"claude-opus-5","harness":"react"},client,proto,enf.new_task_meter(),f"mab_{CFG}_{j}","A")
            else: # single
                r=run_genotype_on_task(single("claude-opus-5"),task,client,protocol=proto,enforcer=enf,purpose=f"mab_{CFG}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
        except Exception as ex:
            print(f"  {CFG} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0)
        if j%15==0: json.dump({"cfg":CFG,"n_done":j+1,"passes":passes},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0
    json.dump({"cfg":CFG,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    print(f"[MODABL {CFG}] SR={sr:.3f}+-{se:.3f}",flush=True); print(f"MODABL_{CFG} DONE",flush=True)

if __name__=="__main__": main()
