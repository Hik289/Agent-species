"""GAP3 — τ³ pipeline transfer: planner→executor on τ³ retail (external benchmark replication).
Planner (opus) produces a strategy from the task's user intent + domain, then the executor
(NativeHarnessAgent subclass) runs the tool-call loop with the strategy prepended to its
instruction. Single-primary executor preserved (user-sim conversation ownership unchanged) —
the pipeline is planner-then-execute, NOT multi-agent-in-loop (that's the τ³ architectural limit).

argv: <genotype: single-opus|opus->opus|opus->haiku> <task-index> <domain> <max-steps>
Prints: BRIDGE_RESULT: {"reward":..,"cost":..,"success":0/1}
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.tau3_robust_proxy import install_robust_completion
install_robust_completion(min_gap=1.5, max_attempts=30)

PLAN_MODELS={"single-opus":("claude-opus-5",None),
             "opus->opus":("claude-opus-5","claude-opus-5"),
             "opus->haiku":("claude-opus-5","claude-haiku-4-5-20251001")}

def _make_plan(planner_model, domain, policy, user_intent):
    """One planner call → high-level strategy (no tools, just strategic guidance)."""
    import litellm
    sys_p=("You are a PLANNER for a customer-service agent operating under a strict domain policy. "
           "Given the policy and the user's situation, produce a concise step-by-step STRATEGY "
           "(5-8 steps) the executor should follow: which info to gather, which tools/actions in "
           "what order, and policy constraints to respect. Output only the numbered strategy.")
    u=f"DOMAIN: {domain}\n\nPOLICY (excerpt):\n{policy[:3000]}\n\nUSER SITUATION:\n{user_intent}\n\nStrategy:"
    r=litellm.completion(model=f"openai/{planner_model}" if planner_model.startswith("gpt") else f"anthropic/{planner_model}",
                         messages=[{"role":"system","content":sys_p},{"role":"user","content":u}],
                         timeout=180)
    txt=r.choices[0].message.content or ""
    cost=0.0
    try: cost=float(getattr(r,"_hidden_params",{}).get("response_cost",0) or 0)
    except Exception: pass
    return txt, cost

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--geno", default="opus->opus")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--user-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--db-only", action="store_true")
    args=ap.parse_args()

    planner_model, exec_model = PLAN_MODELS[args.geno]
    if exec_model is None: exec_model = planner_model  # single: executor = planner model, NO plan injected

    from tau2.registry import registry
    from phase2.tau3_native_harness import NativeHarnessAgent

    # planner runs first (skipped for single-opus to keep it a true monolithic baseline)
    plan_cost=0.0; strategy=""
    from tau2.run import get_tasks
    tasks=get_tasks(args.domain)
    if args.db_only:
        def _has_db(t):
            ev=getattr(t,"evaluation_criteria",None)
            return bool(ev and (getattr(ev,"actions",None) or getattr(ev,"env_assertions",None)))
        tasks=[t for t in tasks if _has_db(t)]
    if args.task_index>=len(tasks):
        print(f"BRIDGE_RESULT: {json.dumps({'error':'oob','reward':0,'cost':0,'success':0})}"); return
    task=tasks[args.task_index]

    # extract user intent for the planner
    intent=""
    try:
        intent=getattr(task,"description",None) or ""
        if not intent and getattr(task,"user_scenario",None):
            intent=str(task.user_scenario)[:1500]
    except Exception: pass

    is_pipeline = (args.geno!="single-opus")

    def _factory(tools, domain_policy, **kwargs):
        nonlocal plan_cost, strategy
        if is_pipeline and not strategy:
            try:
                strategy, plan_cost = _make_plan(planner_model, args.domain, domain_policy, intent or "(see policy)")
            except Exception as e:
                strategy=""
        agent = NativeHarnessAgent(tools=tools, domain_policy=domain_policy,
                                   llm=exec_model, harness="react", policy_placement="user")
        if is_pipeline and strategy:
            # prepend strategy to the agent instruction (executor follows planner's plan)
            orig = type(agent).agent_instruction.fget(agent)
            agent.__dict__["_strategy_prefix"] = strategy
            # monkeypatch instance agent_instruction via subclass-less override
            import types
            def _ai(self):
                base = NativeHarnessAgent.agent_instruction.fget(self)
                return (f"A PLANNER has given you this STRATEGY; follow it step by step:\n<strategy>\n{self.__dict__['_strategy_prefix']}\n</strategy>\n\n" + base)
            agent.__class__ = type("PipelineAgent",(NativeHarnessAgent,),{"agent_instruction":property(_ai)})
        return agent

    try: registry.register_agent_factory(_factory,"harness_agent")
    except Exception: pass

    from tau2.runner.batch import run_single_task
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    cfg=TextRunConfig(domain=args.domain, agent="harness_agent", user="user_simulator",
                      llm_agent=exec_model, llm_args_agent={}, llm_user=args.user_model,
                      llm_args_user={}, max_steps=args.max_steps, max_errors=10)
    try:
        res=run_single_task(cfg,task,seed=0,evaluation_type=EvaluationType.ENV)
        ri=getattr(res,"reward_info",None)
        reward=ri.reward if ri else getattr(res,"reward",0.0)
        cost=(getattr(res,"agent_cost",0.0) or 0.0)+plan_cost
        print(f"BRIDGE_RESULT: {json.dumps({'reward':float(reward),'cost':float(cost),'success':int(reward>=1.0),'plan_cost':plan_cost})}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"BRIDGE_RESULT: {json.dumps({'error':str(e)[:150],'reward':0,'cost':0,'success':0})}")

if __name__=="__main__": main()
