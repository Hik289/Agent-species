"""Compare plan delivery and plan updates during interactive tasks.

Setting A gives the executor the full plan. B asks the planner to update it
at intervals. C adds instructions for following the current strategy. The
runner records task outcomes and spending from both roles."""
import sys, json, math
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action, _harness_sys, _build_msgs, run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

REPLAN_EVERY = 8  # B: re-plan cadence

def _plan(client, planner_model, goal, obs, meter, purpose):
    strat, c = _chat(client, planner_model,
                     _build_msgs("You are a planner. Output a short numbered strategy (max 4 steps) to achieve the goal. Include the FINAL steps (e.g. taking and placing/delivering the target object), not just the search.",
                                 f"GOAL: {goal}\nOBS: {obs['observation']}", planner_model),
                     meter, purpose)
    return strat, c

def run_pipeline_fixed(task, planner, executor, client, proto, meter, purpose, mode):
    obs, goal = task.reset()
    cost = 0.0; strat = ""; step = 0; success = 0
    if meter.can_step():
        try:
            strat, c = _plan(client, planner["model"], goal, obs, meter, f"{purpose}_plan"); cost += c
        except Exception:
            strat = ""
    # C: pure-execution prompt vs default executor harness
    if "C" in mode:
        exec_sys = ("You are an EXECUTOR. A planner gave you a STRATEGY. Execute the strategy "
                    "step by step. Follow the plan's steps in order; do NOT re-plan from scratch. "
                    "Pick the single admissible command that best advances the current strategy step. "
                    "Output exactly: ACTION: <command>")
    else:
        exec_sys = _harness_sys(executor["harness"], proto, interactive=True)
    history = []
    while meter.can_step():
        step += 1
        # B: re-plan every REPLAN_EVERY turns (planner sees current obs)
        if "B" in mode and step > 1 and (step - 1) % REPLAN_EVERY == 0 and meter.can_step():
            try:
                strat, c = _plan(client, planner["model"], goal, obs, meter, f"{purpose}_replan{step}"); cost += c
            except Exception:
                pass
        adm = obs["admissible_commands"]
        if not adm:
            break
        # BUG2 FIX: history stores action + resulting observation (memory), show last 6
        recent = ("History (past actions and results):\n" + "\n".join(history[-30:]) + "\n\n") if history else ""
        adm_list = "\n".join("- " + a for a in adm)
        # A: FULL plan (no [:300] truncation)
        plan_shown = strat  # no truncation
        user = (f"GOAL: {goal}\nSTRATEGY:\n{plan_shown}\n\n{recent}"
                f"OBSERVATION: {obs['observation']}\nADMISSIBLE:\n{adm_list}\n\nYour command (reply 'ACTION: <admissible command>'):")
        try:
            text, c = _chat(client, executor["model"], _build_msgs(exec_sys, user, executor["model"]), meter, f"{purpose}_exec")
            cost += c
        except Exception:
            break
        from phase2.agent_exec import _pick_action2
        action, _ok = _pick_action2(text, adm)
        obs, reward, done, info = task.step(action)
        res = (obs.get("observation","") or "")[:800]
        history.append(f"> {action}  => {res}")
        if reward > 0 or info.get("won"):
            success = 1; break
        if done:
            break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step}

def single(model):
    return {"roles": {"solver": {"model": model, "harness": "react", "params": {}}},
            "topology": "single", "topology_params": {}, "_lineage_id": "fix"}

PIPELINES = {  # planner, executor
    "opus->opus": ("claude-opus-5","claude-opus-5"),
    "opus->sonnet": ("claude-opus-5","claude-sonnet-5"),
    "opus->haiku": ("claude-opus-5","claude-haiku-4-5-20251001"),
    "sonnet->sonnet": ("claude-sonnet-5","claude-sonnet-5"),
    "sonnet->haiku": ("claude-sonnet-5","claude-haiku-4-5-20251001"),
    "opus->GPT": ("claude-opus-5","gpt-5.6-luna"),
    "opus->sol": ("claude-opus-5","gpt-5.6-sol"),
    "sol->opus": ("gpt-5.6-sol","claude-opus-5"),
    "luna->opus": ("gpt-5.6-luna","claude-opus-5"),
    "sonnet->opus": ("claude-sonnet-5","claude-opus-5"),
    "sonnet->sol": ("claude-sonnet-5","gpt-5.6-sol"),
    "haiku->opus": ("claude-haiku-4-5-20251001","claude-opus-5"),
    "haiku->sonnet": ("claude-haiku-4-5-20251001","claude-sonnet-5"),
    "haiku->haiku": ("claude-haiku-4-5-20251001","claude-haiku-4-5-20251001"),
    "haiku->sol": ("claude-haiku-4-5-20251001","gpt-5.6-sol"),
    "sol->sonnet": ("gpt-5.6-sol","claude-sonnet-5"),
    "sol->haiku": ("gpt-5.6-sol","claude-haiku-4-5-20251001"),
    "sol->sol": ("gpt-5.6-sol","gpt-5.6-sol"),
}
SINGLES = {"single-opus":"claude-opus-5","single-sonnet":"claude-sonnet-5","single-haiku":"claude-haiku-4-5-20251001","single-GPT":"gpt-5.6-luna","single-sol":"gpt-5.6-sol"}

def _task_key(t):
    return t[1] if (isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], str)) else str(t)

def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else "A"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld","valid_seen",5+N+20,seed=0,max_steps=30)
    heldout=[t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files)
    print(f"[setup] pipeline-fix mode={MODE} n={len(heldout)} held-out", flush=True)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    client=LLMClient(budget_usd_hard_cap=300.0,tracker_path=f"./results/pipefix_{MODE}_tracker.json")
    RES=f"./results/pipefix_{MODE}.json"
    results={"config":{"mode":MODE,"n":len(heldout),"replan_every":REPLAN_EVERY,"note":"fix strat[:300] truncation (A) + replan (B) + pure-exec (C)"},"legs":{}}
    def run_leg(name, runner):
        for attempt in range(1,3):
            passes=0;cost=0.0;pt=[];ptc=[]
            for j,spec in enumerate(heldout):
                task=make_task(spec)
                try:
                    r=runner(task,j); passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
                except Exception as ex:
                    print(f"  {name} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0)
            zero=sum(1 for x in ptc if x==0)
            if not((cost==0) or (zero==len(heldout))) or attempt>=2: break
            print(f"  [RETRY] {name} all-$0",flush=True)
        n=len(heldout);sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
        results["legs"][name]={"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"CoP":round(cop,4) if cop else None,"n":n,"per_task":pt,"per_task_cost":ptc,"zero_cost_tasks":zero,"invalid_zero_cost":(cost==0 or zero==n)}
        print(f"  [{name}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}",flush=True)
        json.dump(results,open(RES,"w"),indent=2)
    for name,(pl,ex) in PIPELINES.items():
        run_leg(name, lambda task,j,pl=pl,ex=ex,name=name: run_pipeline_fixed(task,{"model":pl,"harness":"react"},{"model":ex,"harness":"react"},client,proto,enf.new_task_meter(),f"pf_{name.replace('->','_')}_{j}",MODE))
    for name,m in SINGLES.items():
        run_leg(name, lambda task,j,m=m,name=name: run_genotype_on_task(single(m),task,client,protocol=proto,enforcer=enf,purpose=f"pf_{name}_{j}"))
    print(f"PIPEFIX_{MODE} DONE",flush=True)

if __name__=="__main__": main()
