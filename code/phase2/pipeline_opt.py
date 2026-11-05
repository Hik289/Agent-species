"""Optimized pipeline: injection + retrieval (author 2026-10-14).
Builds on the de-truncation + memory fix (which already lifted single 0.58->0.874, opus->opus
0.941). Two optimizations to push composition past significance AND fix slowness (full-history
token bloat):
  INJECTION: plan is re-injected every turn with a STEP POINTER (which plan step is current) and
             completed-step marking, so the executor always knows what to do next (not a stale
             one-shot plan buried under history).
  RETRIEVAL: instead of dumping full raw history (token bloat = slow), extract KEY FACTS (object
             locations discovered, containers opened+empty) as a compact memory, plus the last
             2 raw turns. This gives the executor the RELEVANT info for the current step at low
             token cost (fixes slowness + improves SR by surfacing what matters).
Mode A2 (optimized). argv: <genotype> <n>.
"""
import sys, json, math, re
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action2, _harness_sys, _build_msgs, run_genotype_on_task
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from phase2.pipeline_fix import single, PIPELINES, SINGLES, _task_key
from core.llm_client import LLMClient

def _extract_key_facts(facts, action, obs_text):
    """RETRIEVAL: update a compact key-facts dict from the latest action+observation.
    Captures: what was found where, which containers were opened and were empty."""
    ot = (obs_text or "").lower()
    a = (action or "").lower()
    # "you open X. the X is empty" / "you see a Y"
    if "open" in a and ("empty" in ot or "nothing" in ot):
        facts.setdefault("empty_containers", set())
        m = re.search(r"open the (\w+)", a) or re.search(r"open (\w+)", a)
        if m: facts["empty_containers"].add(m.group(1))
    # found objects: "you see a watch_1" etc. in obs
    for m in re.finditer(r"a ([a-z]+_\d+)", ot):
        facts.setdefault("seen", set()).add(m.group(1))
    # picked up
    if "take" in a or "pick" in a:
        facts.setdefault("holding", a)
    return facts

def _facts_str(facts):
    parts = []
    if facts.get("holding"): parts.append(f"You are holding: result of '{facts['holding']}'")
    if facts.get("empty_containers"): parts.append("Already-checked EMPTY: " + ", ".join(sorted(facts["empty_containers"])))
    if facts.get("seen"): parts.append("Objects seen: " + ", ".join(sorted(list(facts["seen"]))[:15]))
    return "\n".join(parts)

def _plan_with_steps(client, planner_model, goal, obs, meter, purpose):
    strat, c = _chat(client, planner_model,
        _build_msgs("You are a planner. Output a numbered step-by-step plan (3-6 steps) to achieve the goal. "
                    "Include the FINAL steps (take the target, go to destination, put/place it). Number each step.",
                    f"GOAL: {goal}\nINITIAL OBSERVATION: {obs['observation']}", planner_model), meter, purpose)
    return strat, c

def run_pipeline_opt(task, planner, executor, client, proto, meter, purpose):
    obs, goal = task.reset()
    cost = 0.0; step = 0; success = 0
    strat, c = _plan_with_steps(client, planner["model"], goal, obs, meter, f"{purpose}_plan"); cost += c
    facts = {}
    last_turns = []  # last 2 raw (action => short obs)
    cur_step = 1
    exec_sys = ("You are an EXECUTOR following a numbered PLAN. Each turn you are told the CURRENT PLAN STEP "
                "and KEY FACTS discovered so far. Pick the single admissible command that advances the current "
                "plan step. When a step is clearly done, move to the next. Reply exactly 'ACTION: <admissible command>'.")
    while meter.can_step():
        step += 1
        adm = obs["admissible_commands"]
        if not adm: break
        recent = ("\n".join(last_turns[-2:]) + "\n") if last_turns else ""
        kf = _facts_str(facts)
        # INJECTION: plan + current step pointer every turn; RETRIEVAL: key facts + last 2 turns (compact)
        user = (f"GOAL: {goal}\n\nPLAN:\n{strat}\n\nCURRENT PLAN STEP: ~{cur_step}\n\n"
                f"KEY FACTS (memory):\n{kf}\n\nRECENT:\n{recent}\n"
                f"CURRENT OBSERVATION: {obs['observation']}\nADMISSIBLE:\n" + "\n".join("- "+a for a in adm) +
                "\n\nYour command (reply 'ACTION: <admissible command>'):")
        try:
            text, c = _chat(client, executor["model"], _build_msgs(exec_sys, user, executor["model"]), meter, f"{purpose}_exec"); cost += c
        except Exception:
            break
        action, _ok = _pick_action2(text, adm)
        obs, reward, done, info = task.step(action)
        otxt = obs.get("observation","") or ""
        facts = _extract_key_facts(facts, action, otxt)
        last_turns.append(f"> {action} => {otxt[:150]}")
        # advance step pointer heuristically: if action matches a 'take'/'put' advance
        if any(k in action.lower() for k in ["take","put","place","go to"]) and cur_step < 6:
            cur_step += 1
        if reward > 0 or info.get("won"):
            success = 1; break
        if done: break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step}

def main():
    GENO = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 135
    sel5 = get_tasks("alfworld","valid_seen",5,seed=0,max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld","valid_seen",5+N+40,seed=0,max_steps=30)
    heldout=[t for t in big if _task_key(t) not in sel_files][:N]
    n=len(heldout)
    proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
    tag=GENO.replace("->","_")
    client=LLMClient(budget_usd_hard_cap=400.0,tracker_path=f"./results/opt_{tag}_tracker.json")
    RES=f"./results/opt_{tag}.json"
    print(f"[setup] OPT {GENO} n={n}",flush=True)
    is_pipe=GENO in PIPELINES
    passes=0;cost=0.0;pt=[];ptc=[]
    for j,spec in enumerate(heldout):
        task=make_task(spec)
        try:
            if is_pipe:
                pl,ex=PIPELINES[GENO]
                r=run_pipeline_opt(task,{"model":pl,"harness":"react"},{"model":ex,"harness":"react"},client,proto,enf.new_task_meter(),f"opt_{tag}_{j}")
            else:
                r=run_genotype_on_task(single(SINGLES[GENO]),task,client,protocol=proto,enforcer=enf,purpose=f"opt_{tag}_{j}")
            passes+=r["success"];cost+=r["cost_usd"];pt.append(r["success"]);ptc.append(round(r["cost_usd"],5))
        except Exception as ex:
            print(f"  {GENO} t{j} ERR {str(ex)[:40]}",flush=True);pt.append(0);ptc.append(0.0)
        if j%10==0: json.dump({"geno":GENO,"n_done":j+1,"passes":passes,"cost":round(cost,3)},open(RES,"w"),indent=2)
    sr=passes/n;se=math.sqrt(sr*(1-sr)/n) if n else 0;cop=cost/passes if passes else None
    json.dump({"geno":GENO,"n":n,"SR":round(sr,4),"SE":round(se,4),"cost":round(cost,4),"CoP":round(cop,4) if cop else None,"per_task":pt,"per_task_cost":ptc},open(RES,"w"),indent=2)
    print(f"[OPT {GENO}] SR={sr:.3f}+-{se:.3f} CoP=${cop if cop else 0:.3f} cost=${cost:.2f}",flush=True)
    print(f"OPT_{tag} DONE",flush=True)

if __name__=="__main__": main()
