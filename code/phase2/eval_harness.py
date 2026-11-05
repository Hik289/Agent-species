"""Phase 0 eval harness: Cost-of-Pass fitness over benchmarks (orchestrator 2026-07-25).

Evaluates a genotype (or baseline) on a set of tasks from pluggable benchmarks,
returns success rate + total cost + Cost-of-Pass = cost / #passes (lower = better).

Fitness (Cost-of-Pass): for a genotype g over N tasks,
  passes = #tasks solved, cost = total $ spent
  CoP = cost / max(passes, eps)   (dollars per solved task; ∞-ish if 0 passes)
We report success_rate, total_cost, CoP, and a fitness = success_rate / (1 + cost_norm)
option too. orchestrator's Cost-of-Pass is the primary.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from core.llm_client import LLMClient, BudgetExceeded
from phase0 import genotype as GT
from phase0 import alfworld_bench as ALF
from phase0.agent_exec import run_genotype_on_task


def get_alfworld_tasks(split, n, seed=0, max_steps=20):
    games = ALF.list_games(split)
    rng = random.Random(seed)
    rng.shuffle(games)
    chosen = games[:n]
    return [("alfworld", g, max_steps) for g in chosen]


def get_humaneval_tasks(n, seed=0):
    from phase0 import humaneval_bench as HE
    return HE.list_problems(n=n, seed=seed)


def get_tasks(benchmark, split, n, seed=0, max_steps=20):
    if benchmark == "alfworld":
        return get_alfworld_tasks(split, n, seed=seed, max_steps=max_steps)
    if benchmark == "humaneval":
        return get_humaneval_tasks(n, seed=seed)
    if benchmark == "planbench":
        from phase2 import planbench_bench as PB
        return PB.list_problems(n=n, seed=seed)
    raise ValueError(f"unknown benchmark {benchmark}")


def make_task(spec):
    kind, ref, max_steps = spec
    if kind == "alfworld":
        return ALF.ALFWorldTask(ref, max_steps=max_steps)
    if kind == "humaneval":
        from phase0 import humaneval_bench as HE
        return HE.HumanEvalTask(ref, max_steps=max_steps)
    if kind == "planbench":
        from phase2 import planbench_bench as PB
        return PB.PlanBenchTask(ref)
    raise ValueError(f"unknown benchmark {kind}")


def evaluate(genotype, task_specs, client, purpose):
    """Run genotype on all task_specs. Returns aggregate + per-task records."""
    passes = 0
    total_cost = 0.0
    recs = []
    for i, spec in enumerate(task_specs):
        task = make_task(spec)
        try:
            r = run_genotype_on_task(genotype, task, client, purpose=f"{purpose}_t{i}")
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            r = {"success": 0, "cost_usd": 0.0, "steps": 0, "error": str(e)[:80]}
        passes += r["success"]
        total_cost += r["cost_usd"]
        ref = spec[1]
        task_name = ref.split("/")[-2] if isinstance(ref, str) and "/" in ref else str(ref)
        recs.append({"task": task_name,
                     "success": r["success"], "cost": round(r["cost_usd"], 6),
                     "steps": r.get("steps")})
    n = len(task_specs)
    cop = total_cost / passes if passes > 0 else None  # None = no passes (∞)
    return {"n_tasks": n, "passes": passes, "success_rate": round(passes / n, 4) if n else 0,
            "total_cost": round(total_cost, 6),
            "cost_of_pass": round(cop, 6) if cop is not None else None,
            "per_task": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="alfworld")
    ap.add_argument("--split", default="valid_seen")
    ap.add_argument("--n-tasks", dest="n_tasks", type=int, default=5)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="smoke", help="smoke=baselines+few compositions")
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", default="./results/phase0_smoke.json")
    ap.add_argument("--tracker", default="./budget_tracker.json")
    args = ap.parse_args()

    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="phase0_eval", max_retries=6)
    initial = client.total_usd
    task_specs = get_alfworld_tasks(args.split, args.n_tasks, seed=args.seed, max_steps=args.max_steps)

    # candidates: 2 single-model baselines + a few compositions
    candidates = {
        "baseline_mini_react": GT.make_genotype("single", "gpt-5.4-mini", "react", "base_mini"),
        "baseline_gpt54_react": GT.make_genotype("single", "gpt-5.4", "react", "base_gpt54"),
        "single_mini_cot": GT.make_genotype("single", "gpt-5.4-mini", "cot", "s_mini_cot"),
        "pipeline_mini": GT.make_genotype("pipeline", "gpt-5.4-mini", "react", "pipe_mini"),
        "pipeline_hetero": {**GT.make_genotype("pipeline", "gpt-5.4-mini", "react", "pipe_het"),
                            "roles": {"planner": GT.default_role("gpt-5.4", "cot"),
                                      "executor": GT.default_role("gpt-5.4-mini", "react")}},
    }

    results = {}
    t0 = time.time()
    print(f"=== Phase 0 smoke: {args.benchmark}/{args.split} n={args.n_tasks} ===", flush=True)
    for name, g in candidates.items():
        try:
            res = evaluate(g, task_specs, client, purpose=f"phase0_{name}")
        except BudgetExceeded as e:
            print(f"BudgetExceeded at {name}: {e}", flush=True)
            break
        results[name] = {"genotype_sig": GT.genotype_signature(g), **res}
        cop = res["cost_of_pass"]
        print(f"  {name:24} {GT.genotype_signature(g):40} "
              f"pass={res['passes']}/{res['n_tasks']} SR={res['success_rate']} "
              f"cost=${res['total_cost']:.4f} CoP={cop if cop is not None else 'inf'}", flush=True)

    spent = client.total_usd - initial
    out = {"benchmark": args.benchmark, "split": args.split, "n_tasks": args.n_tasks,
           "seed": args.seed, "spent_usd": round(spent, 4),
           "elapsed_sec": round(time.time() - t0, 1), "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n=== DONE: spent ${spent:.4f}, wrote {args.out} ===", flush=True)


if __name__ == "__main__":
    main()
