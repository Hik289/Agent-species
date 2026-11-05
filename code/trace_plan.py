import sys; sys.path.insert(0,".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _build_msgs
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient
proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
client=LLMClient(budget_usd_hard_cap=5.0,tracker_path="./results/trace.json")
spec=get_tasks("alfworld","valid_seen",6,seed=0,max_steps=30)[5]
task=make_task(spec); obs,goal=task.reset()
m=enf.new_task_meter()
ob=obs["observation"][:500]
strat,c=_chat(client,"claude-opus-5",_build_msgs("You are a planner. Output a short numbered strategy (max 4 steps).",f"GOAL: {goal}\nOBS: {ob}","claude-opus-5"),m,"plan")
print("GOAL:",goal)
print("INITIAL OBS:",ob[:200])
print("=== PLANNER (opus) upfront strategy (full) ===")
print(strat)
print(f"=== strat len={len(strat)}, truncated[:300] executor sees ===")
print(strat[:300])
