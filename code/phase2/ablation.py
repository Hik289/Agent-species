"""S1 Ablation: per-bug-fix contribution (差距即 bug quantification) — 2026-10-14.
Toggles each fix to measure its SR contribution:
  config "baseline" = OLD BUGS (strat[:300]-style trunc + action-only history + blind parse)
  config "mem"      = baseline + memory (history stores action=>obs)
  config "mem_notrunc" = mem + de-truncation (full obs/history/admissible)
  config "full"     = mem_notrunc + safe parse (ACTION format + 'look' fallback)
Single-model (opus/sonnet) on held-out n. argv: <config> <model> <n>.
"""
import sys, json, math, re
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action, _pick_action2, _harness_sys, _build_msgs
from phase2.pipeline_fix import _task_key
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

def run_ablation(task, model, client, proto, meter, purpose, cfg):
    mem = cfg in ("mem","mem_notrunc","full")
    notrunc = cfg in ("mem_notrunc","full","notrunc_only")
    safeparse = cfg == "full"
    sys_prompt = _harness_sys("react", proto, interactive=True)
    obs, goal = task.reset()
    cost=0.0; success=0; step=0; history=[]
    while meter.can_step():
        step+=1
        adm = obs["admissible_commands"]
        if not adm: break
        hist_window = history[-30:] if notrunc else history[-3:]
        recent = ("History:\n" + "\n".join(hist_window) + "\n\n") if history else ""
        adm_list = "\n".join("- " + a for a in (adm if notrunc else adm[:40]))
        inv = (obs.get("inventory") or "") if notrunc else (obs.get("inventory") or "")[:150]
        ob = obs['observation'] if notrunc else obs['observation'][:600]
        tail = "\n\nYour command (reply exactly 'ACTION: <one admissible command>'):" if safeparse else "\n\nYour command:"
        user = f"GOAL: {goal}\n\n{recent}OBSERVATION: {ob}\nINVENTORY: {inv}\nADMISSIBLE COMMANDS:\n{adm_list}{tail}"
        try:
            text,c=_chat(client,model,_build_msgs(sys_prompt,user,model),meter,f"{purpose}_step"); cost+=c
        except Exception: break
        if safeparse:
            action,_ok=_pick_action2(text,adm)
        else:
            action=_pick_action(text,adm)
        obs,reward,done,info=task.step(action)
        if mem:
            res=(obs.get("observation","") or "")[:500 if notrunc else 150]
            history.append(f"> {action} => {res}")
        else:
            history.append(f"> {action}")  # OLD BUG: action only, no observation
        if reward>0 or info.get("won"): success=1; break
        if done: break
    task.close()
    return {"success":success,"cost_usd":cost,"steps":step}

def main():
    CFG=sys.argv[1]; MODEL=sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 135
    sel5=get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel=set(_task_key(t) for t in sel5)
    big=get_tasks("alfworld","valid_seen",5+N+30,seed=0,max_steps=30)
    ho=[t for t in big if _task_key(t) not in sel][:N]; n=len(ho)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    tag=f"{CFG}_{MODEL.replace('claude-','').replace('-','')}"
    client=LLMClient(budget_usd_hard_cap=300.0,tracker_path=f"./results/abl_{tag}_tracker.json")
    RES=f"./results/abl_{tag}.json"
    print(f"[setup] ablation cfg={CFG} model={MODEL} n={n}",flush=True)
    passes=0;cost=0.0;pt=[];ptc=[]
    for j,spec in enumerate(ho):
        task=make_task(spec)
        try:
            r=run_ablation(task,MODEL,client,proto,enf.new_task_meter(),f"abl_{tag}_{j}",CFG)
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
        except Exception as ex:
            print(f"  {CFG} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0)
        if j%15==0: json.dump({"cfg":CFG,"model":MODEL,"n_done":j+1,"passes":passes},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0
    json.dump({"cfg":CFG,"model":MODEL,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"per_task":pt},open(RES,"w"),indent=2)
    print(f"[ABL {CFG} {MODEL}] SR={sr:.3f}+-{se:.3f} cost=${cost:.2f}",flush=True)
    print(f"ABL_{tag} DONE",flush=True)

if __name__=="__main__": main()
