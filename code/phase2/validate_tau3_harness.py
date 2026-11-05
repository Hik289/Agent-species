"""τ³ validation gate (orchestrator-specified). Run inside tau2_py312 subprocess.

Gate criteria (ALL must pass before any headline re-run):
  1. Reference sanity: tau2 native reference agent reward>0 on the test retail tasks
     (confirms tasks solvable + reward-read correct).
  2. Our NativeHarnessAgent: reward>0 on >=3-5 retail tasks across >=2 Claude tiers
     (opus+haiku) AND >=2 harnesses (react+direct).
  3. SR range sane (approaching published tau-bench ~0.3-0.6, not <0.1).
  4. Harnesses DIFFER: react vs direct (vs reflexion/cot) produce different SR/behavior
     (thesis requires harness to matter).

Writes results to ./results/tau3_validation.json and prints a summary. Throttled via the
robust proxy wrapper (bursty Claude proxy).

Usage:
  python validate_tau3_harness.py --tasks 5 --tiers claude-opus-4-8,claude-haiku-4-5-20251001 \
      --harnesses react,direct,reflexion --min-gap 2.0 --reference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, ".")


def run_task(domain, task, agent_name, agent_model, harness, user_model, max_steps,
             evaluation_type=None):
    from tau2.runner.batch import run_single_task
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    llm_args = {}
    cfg = TextRunConfig(
        domain=domain, agent=agent_name, user="user_simulator",
        llm_agent=agent_model, llm_args_agent=llm_args,
        llm_user=user_model, llm_args_user={}, max_steps=max_steps, max_errors=10,
    )
    t0 = time.time()
    # ENV = deterministic DB-state + env-assertion only (NO LLM judge → no gpt-4.1 leak,
    # aligns with our reviewer-reproducibility principle). Default ALL includes NL asserts.
    ev = EvaluationType.ENV if evaluation_type == "env" else EvaluationType.ALL
    res = run_single_task(cfg, task, seed=0, evaluation_type=ev)
    ri = getattr(res, "reward_info", None)
    msgs = res.get_messages() if hasattr(res, "get_messages") else []
    ntool = sum(1 for m in msgs if getattr(m, "tool_calls", None))
    return {
        "reward": (ri.reward if ri else None),
        "n_tool_calls": ntool,
        "n_msgs": len(msgs),
        "time_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--task-offset", type=int, default=0,
                    help="start index into the task list (to target harder mid-range tasks)")
    ap.add_argument("--tiers", default="claude-opus-4-8,claude-haiku-4-5-20251001")
    ap.add_argument("--harnesses", default="react,direct,reflexion")
    ap.add_argument("--user-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--reference", action="store_true", help="also run tau2 native reference agent")
    ap.add_argument("--policy-placement", default="user", choices=["user", "system"],
                    help="fixed same-X: domain policy in first user msg (user) or system prompt")
    ap.add_argument("--eval-type", default="env", choices=["env", "all"],
                    help="env=deterministic DB-state only (no LLM judge); all=incl NL asserts (gpt-4.1)")
    ap.add_argument("--db-only", action="store_true",
                    help="filter to tasks with DB/action reward criteria (skip NL-only tasks)")
    ap.add_argument("--out", default="./results/tau3_validation.json")
    args = ap.parse_args()

    # install robust proxy wrapper BEFORE importing/registering agents that call it
    from phase2.tau3_robust_proxy import install_robust_completion
    install_robust_completion(min_gap=args.min_gap, max_attempts=args.max_attempts)

    # register our native harness agent factory
    from tau2.registry import registry
    from phase2.tau3_native_harness import NativeHarnessAgent

    # tau2 registers agent factories by name; we register one per harness via closure
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    harnesses = [h.strip() for h in args.harnesses.split(",") if h.strip()]

    def make_factory(harness):
        def _f(tools, domain_policy, **kw):
            # tau2 build_agent passes: llm, llm_args, task, audio_native_config, audio_taps_dir
            llm = kw.get("llm") or "claude-opus-4-8"
            llm_args = kw.get("llm_args") or {}
            return NativeHarnessAgent(tools=tools, domain_policy=domain_policy,
                                      llm=llm, harness=harness, llm_args=llm_args,
                                      policy_placement=args.policy_placement)
        return _f

    for h in harnesses:
        try:
            registry.register_agent_factory(make_factory(h), f"native_harness_{h}")
        except Exception:
            pass

    from tau2.run import get_tasks
    _all = get_tasks(args.domain)
    if args.db_only:
        # keep tasks whose reward criteria include DB-state / actions (deterministic);
        # drop NL-only tasks (which need the gpt-4.1 NL-assertion judge).
        def _has_db(t):
            ev = getattr(t, "evaluation_criteria", None)
            return bool(ev and (getattr(ev, "actions", None) or getattr(ev, "env_assertions", None)))
        _all = [t for t in _all if _has_db(t)]
        print(f"db_only filter: {len(_all)} deterministic tasks", flush=True)
    tasks = _all[args.task_offset: args.task_offset + args.tasks]

    results = {"domain": args.domain, "n_tasks": args.tasks, "tiers": tiers,
               "harnesses": harnesses, "user_model": args.user_model,
               "min_gap": args.min_gap, "cells": []}

    # 1. reference agent (optional)
    if args.reference:
        print("=== REFERENCE (tau2 native llm_agent) ===", flush=True)
        for i, task in enumerate(tasks):
            try:
                r = run_task(args.domain, task, "llm_agent", tiers[0], None,
                             args.user_model, args.max_steps, evaluation_type=args.eval_type)
                r.update({"cell": "reference", "tier": tiers[0], "harness": "native_ref", "task": i})
                results["cells"].append(r)
                print(f"  ref task{i}: reward={r['reward']} tools={r['n_tool_calls']} t={r['time_s']}s", flush=True)
            except Exception as e:
                results["cells"].append({"cell": "reference", "tier": tiers[0], "task": i,
                                         "error": str(e)[:150]})
                print(f"  ref task{i}: ERR {str(e)[:100]}", flush=True)
            _dump(results, args.out)

    # 2/3/4. our harness across tiers x harnesses x tasks
    for tier in tiers:
        for h in harnesses:
            print(f"=== NativeHarness tier={tier} harness={h} ===", flush=True)
            for i, task in enumerate(tasks):
                try:
                    r = run_task(args.domain, task, f"native_harness_{h}", tier, h,
                                 args.user_model, args.max_steps, evaluation_type=args.eval_type)
                    r.update({"cell": "harness", "tier": tier, "harness": h, "task": i})
                    results["cells"].append(r)
                    print(f"  {tier}/{h} task{i}: reward={r['reward']} tools={r['n_tool_calls']} t={r['time_s']}s", flush=True)
                except Exception as e:
                    results["cells"].append({"cell": "harness", "tier": tier, "harness": h,
                                             "task": i, "error": str(e)[:150]})
                    print(f"  {tier}/{h} task{i}: ERR {str(e)[:100]}", flush=True)
                _dump(results, args.out)

    _summ(results)
    _dump(results, args.out)


def _dump(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def _summ(results):
    from collections import defaultdict
    print("\n=== SUMMARY ===", flush=True)
    agg = defaultdict(list)
    for c in results["cells"]:
        if "reward" in c and c["reward"] is not None:
            agg[(c["cell"], c.get("tier"), c.get("harness"))].append(c["reward"])
    for key, rws in sorted(agg.items()):
        cell, tier, h = key
        mean = sum(rws) / len(rws)
        nonzero = sum(1 for x in rws if x > 0)
        print(f"  {cell:9} {tier or '-':28} {h or '-':12} SR(mean)={mean:.3f} nonzero={nonzero}/{len(rws)}", flush=True)
    # harness-differ check (per tier)
    print("\n=== HARNESS-DIFFERS (per tier, mean SR by harness) ===", flush=True)
    by_tier = defaultdict(dict)
    for key, rws in agg.items():
        cell, tier, h = key
        if cell == "harness":
            by_tier[tier][h] = sum(rws) / len(rws)
    for tier, hd in by_tier.items():
        vals = list(hd.values())
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        print(f"  {tier}: {hd}  spread={spread:.3f}", flush=True)


if __name__ == "__main__":
    main()
