"""Run an agent using shared task settings.

The task protocol supplies tools, prompts, turn limits, and token budgets.
The agent setup supplies model choices, execution methods, and the connections
between roles. Each model call is charged to the task budget."""
from __future__ import annotations

import re

from phase2.genotype import TOPOLOGY_ROLES
from phase2.samex_protocol import DEFAULT_PROTOCOL


import os as _os
import time as _time
# AGENTSPECIES_CLAUDE_THROTTLE_S sets the pause between Opus or Sonnet calls.
_CLAUDE_THROTTLE = float(_os.environ.get("AGENTSPECIES_CLAUDE_THROTTLE_S", "0"))


def _chat(client, model, messages, meter, purpose):
    """One call, metered against the same-X token budget. Returns (text, cost)."""
    if _CLAUDE_THROTTLE > 0 and ("opus" in model or "sonnet" in model):
        _time.sleep(_CLAUDE_THROTTLE)
    out = client.chat_model(model=model, messages=messages, max_tokens=2048, purpose=purpose)
    meter.record_turn(out.get("prompt_tokens", 0), out.get("completion_tokens", 0))
    return out["text"], out["cost_usd"]


def _pick_action(text, admissible):
    t = (text or "").strip()
    m = re.search(r"ACTION:\s*(.+)$", t, re.MULTILINE)
    if m:
        cand = m.group(1).strip()
        if cand in admissible:
            return cand
        for a in admissible:
            if a in cand or cand in a:
                return a
    for a in admissible:
        if a == t:
            return a
    for a in admissible:
        if a in t:
            return a
    return admissible[0] if admissible else "look"


def _pick_action2(text, admissible):
    """BUG3 FIX (2026-10-13): like _pick_action but reports whether a real match was found.
    On genuine parse failure, prefer 'look' (safe no-op that re-reveals the scene) over blindly
    executing admissible[0] (which silently did a wrong action, polluting trajectories).
    Returns (action, parsed_ok)."""
    t = (text or "").strip()
    m = re.search(r"ACTION:\s*(.+)$", t, re.MULTILINE)
    if m:
        cand = m.group(1).strip()
        if cand in admissible:
            return cand, True
        for a in admissible:
            if a in cand or cand in a:
                return a, True
    for a in admissible:
        if a == t:
            return a, True
    for a in admissible:
        if a in t:
            return a, True
    # genuine parse failure: prefer a safe 'look' if available, else first admissible (flagged)
    if "look" in admissible:
        return "look", False
    return (admissible[0] if admissible else "look"), False


def _harness_sys(harness, protocol, interactive):
    """Baseline system prompt (same-X) + harness-specific reasoning instruction.

    The TASK framing is identical (protocol baseline); the harness only changes the
    REASONING style layered on top — that's the evolvable dimension.
    """
    base = protocol.system_prompt_interactive if interactive else protocol.system_prompt_static
    if harness == "direct":
        style = " Respond with only the required output/action."
    elif harness == "cot":
        style = " Think step by step briefly, then give the action/answer on the last line."
    elif harness == "react":
        style = " Reason about the observation, then output one action from the admissible list." if interactive \
            else " Reason step by step, then give the final answer after 'FINAL:'."
    elif harness == "reflexion":
        style = " Reason, act; if you detect failure, reflect and correct." if interactive \
            else " Solve; if unsure, reflect and give a corrected 'FINAL:' answer."
    else:
        style = ""
    return base + style


def _build_msgs(sys_prompt, user, model):
    """Build chat messages. For Claude via our proxy, ANY system-role message triggers the
    proxy's tight system-token quota (429 on opus/sonnet, even for a tiny system prompt) —
    verified 2026-08-05: opus 0/15 with a 90-char system role, 15/15 with the same content
    merged into user. So for claude-* we MERGE the system prompt into the user message (no
    system role). This is the same class of fix as the τ³ policy-user workaround, applied to
    the interactive/static agent loops so opus/sonnet multi-turn episodes don't 429."""
    if "claude" in model:
        return [{"role": "user", "content": sys_prompt + "\n\n" + user}]
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def run_interactive(task, role_spec, client, protocol, meter, purpose):
    harness = role_spec["harness"]
    model = role_spec["model"]
    sys_prompt = _harness_sys(harness, protocol, interactive=True)
    obs, goal = task.reset()
    cost = 0.0
    history = []
    success = 0
    step = 0
    while meter.can_step():
        step += 1
        adm = obs["admissible_commands"]
        if not adm:
            break
        # BUG2 FIX (2026-10-13): history now stores ACTION + resulting OBSERVATION (was action-only
        # → agent had NO memory of outcomes, re-tried the same failed actions, capped SR for single
        # AND pipeline). Show last 6 action→result pairs so the agent remembers what it found.
        recent = ("History (your past actions and what happened):\n" + "\n".join(history[-30:]) + "\n\n") if history else ""
        adm_list = "\n".join("- " + a for a in adm)
        inv = (obs.get("inventory") or "")
        user = (f"GOAL: {goal}\n\n{recent}OBSERVATION: {obs['observation']}\n"
                f"INVENTORY: {inv}\nADMISSIBLE COMMANDS:\n{adm_list}\n\nYour command (reply exactly 'ACTION: <one admissible command>'):")
        try:
            text, c = _chat(client, model, _build_msgs(sys_prompt, user, model), meter, f"{purpose}_step")
            cost += c
        except Exception:
            break
        action, parsed_ok = _pick_action2(text, adm)
        obs, reward, done, info = task.step(action)
        # store action + the resulting observation (memory)
        res = (obs.get("observation", "") or "")[:500]
        history.append(f"> {action}  => {res}")
        if reward > 0 or info.get("won"):
            success = 1
            break
        if done:
            break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step,
            "tokens_used": meter.tokens, "turns_used": meter.turns}


def run_static(task, role_spec, client, protocol, meter, purpose):
    harness = role_spec["harness"]
    model = role_spec["model"]
    sys_prompt = _harness_sys(harness, protocol, interactive=False)
    prompt = task.prompt()
    cost = 0.0
    text, c = _chat(client, model, _build_msgs(sys_prompt, prompt, model), meter, purpose)
    cost += c
    if harness in ("direct",):
        answer = text
    else:
        m = re.split(r"FINAL:\s*", text, maxsplit=1)
        answer = m[1].strip() if len(m) > 1 else text
    success = task.score(answer)
    if harness == "reflexion" and not success and meter.can_step():
        try:
            text2, c2 = _chat(client, model,
                              _build_msgs("Your previous answer was wrong. Reflect and give a corrected FINAL: answer.",
                                          prompt + f"\n\nPrevious:\n{answer}", model), meter, f"{purpose}_reflex")
            cost += c2
            m2 = re.split(r"FINAL:\s*", text2, maxsplit=1)
            success = task.score(m2[1].strip() if len(m2) > 1 else text2)
        except Exception:
            pass
    return {"success": int(bool(success)), "cost_usd": cost, "steps": 1,
            "tokens_used": meter.tokens, "turns_used": meter.turns}


def run_genotype_on_task(genotype, task, client, protocol=DEFAULT_PROTOCOL, enforcer=None,
                         purpose="phase2"):
    """Execute a genotype under the same-X protocol. Fixed items from protocol; only
    harness/model/topology vary."""
    meter = enforcer.new_task_meter() if enforcer is not None else _local_meter(protocol)
    topo = genotype["topology"]
    roles = genotype["roles"]
    primary = TOPOLOGY_ROLES[topo][0]
    # τ³ bridge dispatch: τ³ drives its own tool-calling loop via the py3.12 subprocess
    # using the composition's primary role model + harness (same-X: tau2 fixes
    # tools/env/max_steps; harness varies via our HarnessAgent).
    if getattr(task, "kind", "") == "taubench_bridge":
        rs = roles[primary]
        reward, cost, success = task.run_bridge(agent_model=rs["model"], harness=rs["harness"],
                                                 timeout=int(protocol.max_turns * 12 + 120))
        return {"success": int(success), "cost_usd": cost, "steps": 1,
                "tokens_used": 0, "turns_used": 1, "reward": reward}
    interactive = getattr(task, "kind", "interactive") == "interactive"
    role_spec = roles[primary]
    if interactive:
        if topo == "pipeline" and "planner" in roles and "executor" in roles:
            return _pipeline_interactive(genotype, task, client, protocol, meter, purpose)
        return run_interactive(task, role_spec, client, protocol, meter, purpose)
    else:
        if topo == "debate" and sum(1 for r in roles if r.startswith("solver")) >= 2:
            return _debate_static(genotype, task, client, protocol, meter, purpose)
        return run_static(task, role_spec, client, protocol, meter, purpose)


def _local_meter(protocol):
    from phase2.samex_protocol import _TaskMeter
    return _TaskMeter(protocol)


def _pipeline_interactive(genotype, task, client, protocol, meter, purpose):
    planner = genotype["roles"]["planner"]
    executor = genotype["roles"]["executor"]
    obs, goal = task.reset()
    cost = 0.0
    strat = ""
    if meter.can_step():
        try:
            strat, c = _chat(client, planner["model"],
                            _build_msgs("You are a planner. Output a short numbered strategy (max 4 steps).",
                                        f"GOAL: {goal}\nOBS: {obs['observation']}", planner["model"]), meter, f"{purpose}_plan")
            cost += c
        except Exception:
            strat = ""
    sys_prompt = _harness_sys(executor["harness"], protocol, interactive=True)
    history = []
    success = 0
    step = 0
    while meter.can_step():
        step += 1
        adm = obs["admissible_commands"]
        if not adm:
            break
        recent = ("Recent:\n" + "\n".join(history[-30:]) + "\n\n") if history else ""
        adm_list = "\n".join("- " + a for a in adm)
        user = (f"GOAL: {goal}\nSTRATEGY:\n{strat}\n\n{recent}"
                f"OBSERVATION: {obs['observation']}\nADMISSIBLE:\n{adm_list}\n\nYour command:")
        try:
            text, c = _chat(client, executor["model"], _build_msgs(sys_prompt, user, executor["model"]), meter, f"{purpose}_exec")
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
    return {"success": success, "cost_usd": cost, "steps": step,
            "tokens_used": meter.tokens, "turns_used": meter.turns}


def _debate_static(genotype, task, client, protocol, meter, purpose):
    roles = genotype["roles"]
    solvers = [roles[r] for r in roles if r.startswith("solver")][:2]
    judge = roles.get("judge", solvers[0])
    prompt = task.prompt()
    cost = 0.0
    answers = []
    for i, s in enumerate(solvers):
        if not meter.can_step():
            answers.append("")
            continue
        try:
            text, c = _chat(client, s["model"], _build_msgs(protocol.system_prompt_static + " FINAL: <answer>",
                                                            prompt, s["model"]), meter, f"{purpose}_solver{i}")
            cost += c
            m = re.split(r"FINAL:\s*", text, maxsplit=1)
            answers.append(m[1].strip() if len(m) > 1 else text)
        except Exception:
            answers.append("")
    pick = 0
    if meter.can_step() and len(answers) > 1:
        try:
            jt, c = _chat(client, judge["model"], _build_msgs("Pick better answer. Output only A or B.",
                                                              f"Task:\n{prompt}\n\nA:\n{answers[0]}\n\nB:\n{answers[1]}", judge["model"]), meter, f"{purpose}_judge")
            cost += c
            pick = 1 if jt.strip().upper().startswith("B") else 0
        except Exception:
            pick = 0
    chosen = answers[pick] if pick < len(answers) else (answers[0] if answers else "")
    success = task.score(chosen)
    return {"success": int(bool(success)), "cost_usd": cost, "steps": 1,
            "tokens_used": meter.tokens, "turns_used": meter.turns}
