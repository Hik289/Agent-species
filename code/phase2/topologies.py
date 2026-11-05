"""GAP1 — real interactive composition TOPOLOGIES at newest mechanism (de-truncation+memory).
Not pipeline variants: genuinely different composition structures.
  debate    : 2 solvers each propose next ACTION (with memory+full obs) -> judge picks
  router    : router classifies state -> dispatches to one of {explorer, manipulator} executor
  parallel  : k executors independently propose action -> majority vote (tie -> judge)
All interactive on ALFWorld, full-context (no truncation), memory (action=>result history).
argv: <topology> <n>
"""
import sys, json, math, re, collections
sys.path.insert(0,".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action2, _harness_sys, _build_msgs
from phase2.pipeline_fix import _task_key
from phase2.samex_protocol import SameXProtocol, SameXEnforcer

OPUS="claude-opus-5"

def _ctx(goal, strat, history, obs):
    adm=obs["admissible_commands"]
    recent=("History (past actions and results):\n"+"\n".join(history[-30:])+"\n\n") if history else ""
    adm_list="\n".join("- "+a for a in adm)
    extra=(f"STRATEGY:\n{strat}\n\n" if strat else "")
    return adm, (f"GOAL: {goal}\n{extra}{recent}OBSERVATION: {obs['observation']}\nADMISSIBLE:\n{adm_list}\n\n")

def run_debate(task, client, proto, meter, purpose):
    obs,goal=task.reset(); cost=0.0; success=0; step=0; history=[]
    sys_e=_harness_sys("react",proto,interactive=True)
    while meter.can_step():
        step+=1
        adm,base=_ctx(goal,"",history,obs)
        if not adm: break
        props=[]
        for i in range(2):  # 2 solvers propose an action
            if not meter.can_step(): break
            u=base+"Propose the single best admissible command. Reply 'ACTION: <admissible command>'."
            try: t,c=_chat(client,OPUS,_build_msgs(sys_e,u,OPUS),meter,f"{purpose}_solver{i}_s{step}"); cost+=c
            except Exception: t=""
            a,_=_pick_action2(t,adm); props.append(a)
        # judge picks (if they differ)
        if len(set(props))>1 and meter.can_step():
            ju=base+f"Two proposals:\nA: {props[0]}\nB: {props[1]}\nWhich better advances the goal? Reply only A or B."
            try: jt,c=_chat(client,OPUS,_build_msgs("You are a judge. Reply only A or B.",ju,OPUS),meter,f"{purpose}_judge_s{step}"); cost+=c
            except Exception: jt="A"
            action=props[1] if jt.strip().upper().startswith("B") else props[0]
        else:
            action=props[0] if props else (adm[0] if adm else "look")
        obs,reward,done,info=task.step(action)
        history.append(f"> {action}  => {(obs.get('observation','') or '')[:800]}")
        if reward>0 or info.get("won"): success=1; break
        if done: break
    task.close(); return {"success":success,"cost_usd":cost,"steps":step}

def run_router(task, client, proto, meter, purpose):
    obs,goal=task.reset(); cost=0.0; success=0; step=0; history=[]
    sys_e=_harness_sys("react",proto,interactive=True)
    while meter.can_step():
        step+=1
        adm,base=_ctx(goal,"",history,obs)
        if not adm: break
        # router classifies current sub-task
        ru=base+"Classify the CURRENT step: reply only EXPLORE (need to find/navigate to an object) or ACT (manipulate/use an object you can see)."
        mode="ACT"
        if meter.can_step():
            try: rt,c=_chat(client,OPUS,_build_msgs("You are a router. Reply only EXPLORE or ACT.",ru,OPUS),meter,f"{purpose}_router_s{step}"); cost+=c; mode="EXPLORE" if "EXPLORE" in rt.upper() else "ACT"
            except Exception: pass
        spec=("You are an EXPLORER specialist: navigate efficiently to locate the needed object."
              if mode=="EXPLORE" else
              "You are a MANIPULATION specialist: pick/place/use/toggle the object to achieve the goal.")
        u=base+"Reply 'ACTION: <admissible command>'."
        try: t,c=_chat(client,OPUS,_build_msgs(sys_e+" "+spec,u,OPUS),meter,f"{purpose}_exec_{mode}_s{step}"); cost+=c
        except Exception: break
        action,_=_pick_action2(t,adm)
        obs,reward,done,info=task.step(action)
        history.append(f"> {action}  => {(obs.get('observation','') or '')[:800]}")
        if reward>0 or info.get("won"): success=1; break
        if done: break
    task.close(); return {"success":success,"cost_usd":cost,"steps":step}

def run_parallel(task, client, proto, meter, purpose, k=3):
    obs,goal=task.reset(); cost=0.0; success=0; step=0; history=[]
    sys_e=_harness_sys("react",proto,interactive=True)
    while meter.can_step():
        step+=1
        adm,base=_ctx(goal,"",history,obs)
        if not adm: break
        votes=[]
        for i in range(k):
            if not meter.can_step(): break
            u=base+"Reply 'ACTION: <admissible command>'."
            try: t,c=_chat(client,OPUS,_build_msgs(sys_e,u,OPUS),meter,f"{purpose}_v{i}_s{step}"); cost+=c
            except Exception: t=""
            a,_=_pick_action2(t,adm); votes.append(a)
        if votes:
            cnt=collections.Counter(votes); top,n=cnt.most_common(1)[0]
            action=top
        else: action=adm[0] if adm else "look"
        obs,reward,done,info=task.step(action)
        history.append(f"> {action}  => {(obs.get('observation','') or '')[:800]}")
        if reward>0 or info.get("won"): success=1; break
        if done: break
    task.close(); return {"success":success,"cost_usd":cost,"steps":step}

RUNNERS={"debate":run_debate,"router":run_router,"parallel":run_parallel}

def main():
    TOPO=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 135
    sel5=get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30); sel=set(_task_key(t) for t in sel5)
    big=get_tasks("alfworld","valid_seen",5+N+30,seed=0,max_steps=30)
    ho=[t for t in big if _task_key(t) not in sel][:N]; n=len(ho)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    from core.llm_client import LLMClient
    client=LLMClient(budget_usd_hard_cap=400.0,tracker_path=f"./results/topo_{TOPO}_tracker.json")
    RES=f"./results/topo_{TOPO}.json"
    run=RUNNERS[TOPO]
    print(f"[setup] topology {TOPO} n={n}",flush=True)
    passes=0;cost=0.0;pt=[];ptc=[]
    for j,spec in enumerate(ho):
        task=make_task(spec)
        try:
            r=run(task,client,proto,enf.new_task_meter(),f"topo_{TOPO}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
        except Exception as ex:
            print(f"  {TOPO} t{j} ERR {str(ex)[:50]}",flush=True);pt.append(0);ptc.append(0.0)
        if j%10==0: json.dump({"topo":TOPO,"n_done":j+1,"passes":passes,"cost":round(cost,3)},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
    json.dump({"topo":TOPO,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),
               "CoP":round(cop,4) if cop else None,"per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    print(f"[TOPO {TOPO}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f}",flush=True); print(f"TOPO_{TOPO} DONE",flush=True)

if __name__=="__main__": main()
