"""Phase 0 agent executor: run a genotype on a benchmark task (orchestrator 2026-07-25).

Executes a composition genotype (roles→model/scaffold + topology) on one task via the
local proxy (gpt-5.4-mini / gpt-5.4). Tracks cost. Returns success + $ spent for the
Cost-of-Pass fitness.

Scaffolds:
  direct    : one shot answer
  cot       : chain-of-thought then answer
  react     : Thought/Action loop over the task's action space (for ALFWorld-style envs)
  reflexion : react + a self-reflection retry on failure

Topologies:
  single    : one solver role runs the scaffold
  pipeline  : planner drafts a strategy, executor runs react with that strategy in context
  debate    : two solvers propose, judge picks (for QA/coding-style single-answer tasks)
  router    : router picks a specialist role to handle the task

The benchmark provides an environment interface (see benchmark.py). For interactive envs
(ALFWorld) the executor drives the react loop with admissible_commands. For static tasks
(HumanEval/QA) the scaffold produces a single answer scored by the benchmark.
"""
from __future__ import annotations

import json
import re
from typing import Any

from phase0.genotype import TOPOLOGY_ROLES


def _chat(client, model, messages, max_tokens=800, purpose="phase0"):
    """Proxy chat with per-model routing. Returns (text, cost)."""
    out = client.chat_model(model=model, messages=messages, max_tokens=max_tokens, purpose=purpose)
    return out["text"], out["cost_usd"]


# ---------------- interactive (ALFWorld-style) react loop ----------------

_REACT_SYS = (
    "You are an agent solving a household task in a text world. Each turn you receive the "
    "observation and a list of ADMISSIBLE commands. Respond with EXACTLY ONE command copied "
    "verbatim from the admissible list that best progresses toward the goal. Output only the "
    "command, nothing else."
)

_REACT_COT_SYS = (
    "You are an agent solving a text-world household task. Think step by step briefly, then "
    "on the LAST line output 'ACTION: <one admissible command copied verbatim>'."
)


def _pick_action(text, admissible):
    """Extract a valid admissible command from model output (robust)."""
    t = (text or "").strip()
    # explicit ACTION: line
    m = re.search(r"ACTION:\s*(.+)$", t, re.MULTILINE)
    if m:
        cand = m.group(1).strip()
        if cand in admissible:
            return cand
        for a in admissible:
            if a in cand or cand in a:
                return a
    # exact match anywhere
    for a in admissible:
        if a == t:
            return a
    # substring match
    for a in admissible:
        if a in t:
            return a
    # fallback: first admissible
    return admissible[0] if admissible else "look"


def run_interactive(task, role_spec, client, purpose):
    """Drive an interactive env (reset/step) with a react/reflexion scaffold. Returns dict."""
    scaffold = role_spec["scaffold"]
    model = role_spec["model"]
    max_steps = int(role_spec.get("params", {}).get("max_steps", 20))
    sys_prompt = _REACT_COT_SYS if scaffold in ("cot", "reflexion") else _REACT_SYS

    obs, goal = task.reset()
    cost = 0.0
    history = []
    success = 0
    for step in range(max_steps):
        adm = obs["admissible_commands"]
        if not adm:
            break
        # keep prompt bounded: last 3 history entries
        hist_txt = "\n".join(history[-3:])
        recent = f"Recent:\\n{hist_txt}\\n\\n" if hist_txt else ""
        user = (f"GOAL: {goal}\n\n{recent}"
                f"OBSERVATION: {obs['observation'][:600]}\n"
                f"INVENTORY: {obs.get('inventory','')[:150]}\n"
                f"ADMISSIBLE COMMANDS:\n" + "\n".join(f"- {a}" for a in adm[:40]) +
                "\n\nYour command:")
        try:
            text, c = _chat(client, model, [{"role": "system", "content": sys_prompt},
                                            {"role": "user", "content": user}],
                            max_tokens=250, purpose=f"{purpose}_step")
            cost += c
        except Exception:
            break
        action = _pick_action(text, adm)
        obs, reward, done, info = task.step(action)
        history.append(f"> {action}")
        if reward > 0 or info.get("won"):
            success = 1
            break
        if done:
            break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step + 1}


# ---------------- static (single-answer) scaffolds ----------------

def run_static(task, role_spec, client, purpose):
    """Static task: scaffold produces one answer; benchmark scores it. Returns dict."""
    scaffold = role_spec["scaffold"]
    model = role_spec["model"]
    prompt = task.prompt()
    cost = 0.0
    if scaffold == "direct":
        sys = "Solve the task. Output only the final answer/code."
        text, c = _chat(client, model, [{"role": "system", "content": sys},
                                        {"role": "user", "content": prompt}],
                        max_tokens=1200, purpose=purpose)
        cost += c
        answer = text
    else:  # cot / react / reflexion → think then answer
        sys = "Think step by step, then give the final answer after 'FINAL:'."
        text, c = _chat(client, model, [{"role": "system", "content": sys},
                                        {"role": "user", "content": prompt}],
                        max_tokens=1500, purpose=purpose)
        cost += c
        m = re.split(r"FINAL:\s*", text, maxsplit=1)
        answer = m[1].strip() if len(m) > 1 else text
    success = task.score(answer)
    if scaffold == "reflexion" and not success:
        # one reflective retry
        sys2 = "Your previous answer was wrong. Reflect briefly and give a corrected FINAL: answer."
        text2, c2 = _chat(client, model,
                          [{"role": "system", "content": sys2},
                           {"role": "user", "content": prompt + f"\n\nPrevious attempt:\n{answer}"}],
                          max_tokens=1500, purpose=f"{purpose}_reflex")
        cost += c2
        m2 = re.split(r"FINAL:\s*", text2, maxsplit=1)
        answer2 = m2[1].strip() if len(m2) > 1 else text2
        success = task.score(answer2)
    return {"success": int(bool(success)), "cost_usd": cost, "steps": 1}


# ---------------- topology dispatch ----------------

def run_genotype_on_task(genotype, task, client, purpose="phase0"):
    """Execute a composition genotype on a task. Returns {success, cost_usd, steps, ...}.

    Interactive vs static determined by task.kind.
    Topology single → the sole role runs. pipeline/debate/router → composed (Phase 0:
    pipeline supported for interactive; debate/router for static; others fall back to
    the first role running solo to keep Phase 0 tractable — full topology exec is Phase 1).
    """
    topo = genotype["topology"]
    roles = genotype["roles"]
    interactive = getattr(task, "kind", "interactive") == "interactive"

    # Phase 0: run the primary role's scaffold; topology composition for pipeline (interactive)
    # and debate (static) implemented; others use primary role (logged as topology).
    primary_role = TOPOLOGY_ROLES[topo][0]
    role_spec = roles[primary_role]

    if interactive:
        if topo == "pipeline" and "planner" in roles and "executor" in roles:
            return _run_pipeline_interactive(genotype, task, client, purpose)
        return run_interactive(task, role_spec, client, purpose)
    else:
        if topo == "debate" and len([r for r in roles if r.startswith("solver")]) >= 2:
            return _run_debate_static(genotype, task, client, purpose)
        return run_static(task, role_spec, client, purpose)


def _run_pipeline_interactive(genotype, task, client, purpose):
    """Planner drafts a high-level strategy (one call), executor runs react with it."""
    planner = genotype["roles"]["planner"]
    executor = genotype["roles"]["executor"]
    obs, goal = task.reset()
    cost = 0.0
    try:
        strat_text, c = _chat(client, planner["model"],
                              [{"role": "system", "content": "You are a planner. Given the goal and "
                                "initial observation, output a short numbered strategy (max 4 steps)."},
                               {"role": "user", "content": f"GOAL: {goal}\nOBS: {obs['observation'][:500]}"}],
                              max_tokens=250, purpose=f"{purpose}_plan")
        cost += c
    except Exception:
        strat_text = ""
    # executor react loop with strategy in context
    max_steps = int(executor.get("params", {}).get("max_steps", 20))
    history = []
    success = 0
    step = 0
    for step in range(max_steps):
        adm = obs["admissible_commands"]
        if not adm:
            break
        hist_txt = "\n".join(history[-3:])
        recent = f"Recent:\\n{hist_txt}\\n\\n" if hist_txt else ""
        user = (f"GOAL: {goal}\nSTRATEGY:\n{strat_text[:300]}\n\n"
                f"{recent}"
                f"OBSERVATION: {obs['observation'][:500]}\n"
                f"ADMISSIBLE:\n" + "\n".join(f"- {a}" for a in adm[:40]) + "\n\nYour command:")
        try:
            text, c = _chat(client, executor["model"],
                           [{"role": "system", "content": _REACT_SYS},
                            {"role": "user", "content": user}], max_tokens=250,
                           purpose=f"{purpose}_exec")
            cost += c
        except Exception:
            break
        action = _pick_action(text, adm)
        obs, reward, done, info = task.step(action)
        history.append(f"> {action}")
        if reward > 0 or info.get("won"):
            success = 1; break
        if done:
            break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step + 1}


def _run_debate_static(genotype, task, client, purpose):
    """Two solvers answer; judge picks. Static tasks only."""
    roles = genotype["roles"]
    solvers = [roles[r] for r in roles if r.startswith("solver")][:2]
    judge = roles.get("judge", solvers[0])
    prompt = task.prompt()
    cost = 0.0
    answers = []
    for i, s in enumerate(solvers):
        try:
            text, c = _chat(client, s["model"],
                           [{"role": "system", "content": "Solve the task. FINAL: <answer>"},
                            {"role": "user", "content": prompt}], max_tokens=1200,
                           purpose=f"{purpose}_solver{i}")
            cost += c
            m = re.split(r"FINAL:\s*", text, maxsplit=1)
            answers.append(m[1].strip() if len(m) > 1 else text)
        except Exception:
            answers.append("")
    # judge
    try:
        jtext, c = _chat(client, judge["model"],
                        [{"role": "system", "content": "Pick the better answer. Output only 'A' or 'B'."},
                         {"role": "user", "content": f"Task:\n{prompt}\n\nA:\n{answers[0]}\n\nB:\n{answers[1] if len(answers)>1 else ''}"}],
                        max_tokens=10, purpose=f"{purpose}_judge")
        cost += c
        pick = 1 if "B" in jtext.upper() else 0
    except Exception:
        pick = 0
    chosen = answers[pick] if pick < len(answers) else answers[0]
    success = task.score(chosen)
    return {"success": int(bool(success)), "cost_usd": cost, "steps": 1}
