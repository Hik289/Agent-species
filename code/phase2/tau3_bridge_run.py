"""Run a retail task through the tau2 benchmark in a separate Python process.

The bridge uses the configured model, execution method, task settings, and
API client wrapper. It reports a BRIDGE_RESULT JSON record with reward, cost,
and success. Reward is based on the environment's database state."""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- route litellm to proxy BEFORE importing tau2 (shared robust wrapper) ----
from phase2.tau3_robust_proxy import install_robust_completion
install_robust_completion(min_gap=1.5, max_attempts=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--agent-model", default="gpt-5.4-mini")
    ap.add_argument("--harness", default="react")
    ap.add_argument("--user-model", default="gpt-5.4-mini")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--db-only", action="store_true",
                    help="index into the DB/action (deterministic) task subset only")
    args = ap.parse_args()

    from tau2.registry import registry
    from phase2.tau3_native_harness import NativeHarnessAgent

    # native-FC harness-agent factory (harness + model captured via closure; policy→user)
    def _factory(tools, domain_policy, **kwargs):
        return NativeHarnessAgent(tools=tools, domain_policy=domain_policy,
                                  llm=args.agent_model, harness=args.harness,
                                  policy_placement="user")

    try:
        registry.register_agent_factory(_factory, "harness_agent")
    except Exception:
        pass  # already registered

    from tau2.run import get_tasks
    from tau2.runner.batch import run_single_task
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType

    tasks = get_tasks(args.domain)
    if args.db_only:
        # deterministic subset: tasks with DB/action reward criteria (no NL-only)
        def _has_db(t):
            ev = getattr(t, "evaluation_criteria", None)
            return bool(ev and (getattr(ev, "actions", None) or getattr(ev, "env_assertions", None)))
        tasks = [t for t in tasks if _has_db(t)]
    if args.task_index >= len(tasks):
        print(f"BRIDGE_RESULT: {json.dumps({'error': 'task_index_oob', 'reward': 0, 'cost': 0, 'success': 0})}")
        return
    task = tasks[args.task_index]
    cfg = TextRunConfig(domain=args.domain, agent="harness_agent", user="user_simulator",
                        llm_agent=args.agent_model, llm_args_agent={},
                        llm_user=args.user_model, llm_args_user={},
                        max_steps=args.max_steps, max_errors=10)
    try:
        # ENV = deterministic DB-state + env-assertions only (NO gpt-4.1 NL-assertion judge)
        res = run_single_task(cfg, task, seed=0, evaluation_type=EvaluationType.ENV)
        ri = getattr(res, "reward_info", None)
        reward = ri.reward if ri else getattr(res, "reward", 0.0)
        cost = getattr(res, "agent_cost", 0.0) or 0.0
        print(f"BRIDGE_RESULT: {json.dumps({'reward': float(reward), 'cost': float(cost), 'success': int(reward >= 1.0)})}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"BRIDGE_RESULT: {json.dumps({'error': str(e)[:120], 'reward': 0, 'cost': 0, 'success': 0})}")


if __name__ == "__main__":
    main()
