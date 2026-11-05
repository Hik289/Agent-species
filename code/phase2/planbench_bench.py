"""PlanBench benchmark wrapper for Phase 2 (agentic, non-docker).

Static single-answer task: agent produces a PDDL plan for a blocksworld problem;
VAL validates it deterministically (no LLM). Reuses Stage 1 planbench_eval helpers.

Provides PlanBenchTask with the static-benchmark interface (kind="static"):
  .prompt()  -> the natural-language planning query
  .score(answer) -> 1 if VAL validates the parsed plan, else 0
Same-X: the tool/env here = the fixed VAL validator + blocksworld domain (identical for
all compositions); harness varies how the plan text is produced.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# reuse Stage 1 planbench machinery
_EXP = "."
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from niches.planbench_eval import (
    load_prompt_json, load_domain_config, get_problem_actions, get_problem_pddl,
    DOMAIN_PDDL, VAL_VALIDATE, _DOMAIN_TO_FN,
)
import subprocess

_DOMAIN = "blocksworld"
_INSTANCES = None


def _instances():
    global _INSTANCES
    if _INSTANCES is None:
        _INSTANCES = load_prompt_json(_DOMAIN, "task_1_plan_generation")
    return _INSTANCES


def list_problems(n=0, seed=0):
    inst = _instances()
    idx = list(range(len(inst)))
    if seed:
        import random
        random.Random(seed).shuffle(idx)
    if n:
        idx = idx[:n]
    return [("planbench", i, 30) for i in idx]


class PlanBenchTask:
    kind = "static"

    def __init__(self, index: int, max_steps: int = 1):
        self.inst = _instances()[index]
        self.index = index
        self._domain_data = load_domain_config(_DOMAIN)

    def prompt(self) -> str:
        return (self.inst["query"] +
                "\n\nProvide the plan as a sequence of actions, one per line, in the "
                "[action arg1 arg2] bracket format shown above.")

    def score(self, answer: str) -> int:
        iid = self.inst["instance_id"]
        try:
            action_set = get_problem_actions(_DOMAIN, iid)
            fn = _DOMAIN_TO_FN[_DOMAIN]
            with tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False, dir="/tmp") as tf:
                plan_file = tf.name
            try:
                fn(answer, action_set, plan_file, self._domain_data, False)
            except Exception:
                try:
                    os.unlink(plan_file)
                except OSError:
                    pass
                return 0
            domain_pddl = DOMAIN_PDDL[_DOMAIN]
            problem_pddl = get_problem_pddl(_DOMAIN, iid)
            try:
                out = subprocess.run([str(VAL_VALIDATE), str(domain_pddl), str(problem_pddl), plan_file],
                                     capture_output=True, text=True, timeout=20)
                ok = "Plan valid" in out.stdout or "Successful plans" in out.stdout
            finally:
                try:
                    os.unlink(plan_file)
                except OSError:
                    pass
            return 1 if ok else 0
        except Exception:
            return 0

    def reset(self):
        return {"observation": self.prompt(), "goal": "", "admissible_commands": [],
                "inventory": ""}, ""

    def close(self):
        pass


if __name__ == "__main__":
    probs = list_problems(n=2)
    print("planbench problems sample:", probs[:2])
    t = PlanBenchTask(probs[0][1])
    print("prompt head:", t.prompt()[:150])
    print("empty answer scores:", t.score("[unstack a b]"), "(likely 0)")
