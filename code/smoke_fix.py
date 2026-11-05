import sys; sys.path.insert(0,".")
from phase0.eval_harness import get_tasks, make_task
from phase2.pipeline_fix import run_pipeline_fixed
from phase2.agent_exec import run_interactive, _pick_action2
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient
# verify BUG3 parse
a,ok=_pick_action2("ACTION: go to drawer_1",["go to drawer_1","look"]); print("BUG3 parse_ok match:",a,ok)
a,ok=_pick_action2("hmm not sure",["go to drawer_1","look"]); print("BUG3 parse_fail->look:",a,ok)
proto=SameXProtocol(max_turns=30,token_budget=100000); enf=SameXEnforcer(proto)
client=LLMClient(budget_usd_hard_cap=5.0,tracker_path="./results/smokefix.json")
task=make_task(get_tasks("alfworld","valid_seen",6,seed=0,max_steps=30)[5])
# single with memory fix
r=run_interactive(task,{"model":"claude-sonnet-5","harness":"react"},client,proto,enf.new_task_meter(),"smokesingle")
print("SINGLE(mem fix) sonnet: success=",r["success"],"steps=",r["steps"],"cost=$%.3f"%r["cost_usd"])
