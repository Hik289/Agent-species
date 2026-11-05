"""τ³-bench task wrapper for the py3.11 harness (calls the py3.12 bridge subprocess).

A TauBenchTask is special: it does NOT run through the normal run_genotype_on_task (which
uses our agent_exec). Instead it shells out to phase2/tau3_bridge_run.py in the tau2_py312
env, passing the composition's agent model + harness, and reads BRIDGE_RESULT JSON
(reward + cost). This keeps τ³ (py3.12) isolated from the main harness (py3.11).

Because τ³ drives its OWN agent loop (with our HarnessAgent), the genotype's PRIMARY role
model + harness are passed to the bridge. Topology beyond single is approximated by the
primary role (τ³ is inherently a single tool-calling agent).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

BRIDGE = Path(__file__).with_name("tau3_bridge_run.py")

# module-level user-sim model (set by search.py). Default gpt-5.4-mini (current run);
# authoritative fully-Claude run sets claude-haiku-4-5-20251001.
TAU_USER_MODEL = "gpt-5.4-mini"


def list_problems(n=0, seed=0, domain="retail"):
    # task refs = (domain, task_index). We don't need to import tau2 here (py3.11);
    # the bridge runs with --db-only, indexing into the DETERMINISTIC (DB/action reward)
    # subset. Counts below are the db_only subset sizes so indices align with the bridge.
    # The bridge validates task_index bounds (returns error if OOB).
    import random
    N_BY_DOMAIN = {"retail": 112, "airline": 40, "telecom": 100, "mock": 10}
    total = N_BY_DOMAIN.get(domain, 40)
    idx = list(range(total))
    if seed:
        random.Random(seed).shuffle(idx)
    if n:
        idx = idx[:n]
    return [("taubench", (domain, i), 30) for i in idx]


class TauBenchTask:
    kind = "taubench_bridge"  # special: run_genotype_on_task dispatches to .run_bridge()

    def __init__(self, ref, max_steps: int = 20):
        self.domain, self.task_index = ref
        self.max_steps = max_steps

    def run_bridge(self, agent_model: str, harness: str, user_model: str = None,
                   timeout: int = 300):
        """Invoke the py3.12 bridge subprocess. Returns (reward, cost, success)."""
        um = user_model or TAU_USER_MODEL
        cmd = [os.environ.get("TAUBENCH_PYTHON") or sys.executable, str(BRIDGE), "--domain", self.domain, "--task-index", str(self.task_index),
               "--agent-model", agent_model, "--harness", harness,
               "--user-model", um, "--max-steps", str(self.max_steps), "--db-only"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                 cwd=os.environ.get("TAUBENCH_SOURCE_DIR") or None)
            for line in out.stdout.splitlines():
                if line.startswith("BRIDGE_RESULT:"):
                    d = json.loads(line[len("BRIDGE_RESULT:"):].strip())
                    return float(d.get("reward", 0.0)), float(d.get("cost", 0.0)), int(d.get("success", 0))
            return 0.0, 0.0, 0
        except Exception:
            return 0.0, 0.0, 0

    def close(self):
        pass
