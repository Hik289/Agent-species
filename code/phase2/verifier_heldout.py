"""Test whether a second model can help an executor correct its actions.

The executor proposes an action and a verifier reviews it. If the verifier
rejects it, the executor can revise once. Spending includes both models.
The comparison uses the same held-out tasks as the single-model agent."""
import sys, json, math, re
sys.path.insert(0, ".")
from phase0.eval_harness import get_tasks, make_task
from phase2.agent_exec import _chat, _pick_action, _harness_sys, _build_msgs
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from core.llm_client import LLMClient

GENERATORS = ["claude-opus-5", "claude-sonnet-5"]  # strong generators to test verifier on
VERIFIER = "claude-haiku-4-5-20251001"             # cheap verifier
HARNESS = "react"

def _task_key(t):
    return t[1] if (isinstance(t, tuple) and len(t) >= 2 and isinstance(t[1], str)) else str(t)

def _verify(client, obs, goal, action, meter, purpose):
    """Cheap verifier: does this action look sensible toward the goal? Returns (ok:bool, cost)."""
    vprompt = (f"You are a verifier. GOAL: {goal}\nOBSERVATION: {obs['observation'][:500]}\n"
               f"PROPOSED ACTION: {action}\n\nIs this action sensible progress toward the goal? "
               f"Answer exactly 'OK' if yes, or 'BAD: <one-line reason>' if it is clearly wrong "
               f"(e.g. irrelevant, repeats a failed action, ignores the goal).")
    msgs = [{"role": "user", "content": vprompt}]
    try:
        text, c = _chat(client, VERIFIER, msgs, meter, purpose)
        ok = not text.strip().upper().startswith("BAD")
        return ok, c, text.strip()[:60]
    except Exception:
        return True, 0.0, "verify-err"  # on verifier failure, don't block (fail-open)

def run_gen_verifier(task, gen_model, client, proto, meter, purpose):
    """Interactive loop: generator proposes, haiku verifier reviews, generator revises once if flagged."""
    sys_prompt = _harness_sys(HARNESS, proto, interactive=True)
    obs, goal = task.reset()
    cost = 0.0; success = 0; step = 0; n_flagged = 0
    history = []
    while meter.can_step():
        step += 1
        adm = obs["admissible_commands"]
        if not adm:
            break
        recent = ("Recent:\n" + "\n".join(history[-3:]) + "\n\n") if history else ""
        adm_list = "\n".join("- " + a for a in adm[:40])
        inv = (obs.get("inventory") or "")[:150]
        user = (f"GOAL: {goal}\n\n{recent}OBSERVATION: {obs['observation'][:600]}\n"
                f"INVENTORY: {inv}\nADMISSIBLE COMMANDS:\n{adm_list}\n\nYour command:")
        try:
            text, c = _chat(client, gen_model, _build_msgs(sys_prompt, user, gen_model), meter, f"{purpose}_gen")
            cost += c
        except Exception:
            break
        action = _pick_action(text, adm)
        # VERIFIER STEP (haiku) — cost counted
        ok, vc, vreason = _verify(client, obs, goal, action, meter, f"{purpose}_verify")
        cost += vc
        if not ok and meter.can_step():
            n_flagged += 1
            # generator revises given the verifier's objection
            revise = (user + f"\n\nA verifier flagged your intended action '{action}' as: {vreason}\n"
                      f"Reconsider and give a better command:")
            try:
                text2, c2 = _chat(client, gen_model, _build_msgs(sys_prompt, revise, gen_model), meter, f"{purpose}_revise")
                cost += c2
                action = _pick_action(text2, adm)
            except Exception:
                pass
        obs, reward, done, info = task.step(action)
        history.append(f"> {action}")
        if reward > 0 or info.get("won"):
            success = 1
            break
        if done:
            break
    task.close()
    return {"success": success, "cost_usd": cost, "steps": step, "n_flagged": n_flagged}

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sel5 = get_tasks("alfworld", "valid_seen", 5, seed=0, max_steps=30)
    sel_files = set(_task_key(t) for t in sel5)
    big = get_tasks("alfworld", "valid_seen", 5 + N + 8, seed=0, max_steps=30)
    heldout = [t for t in big if _task_key(t) not in sel_files][:N]
    assert not (set(_task_key(t) for t in heldout) & sel_files), "held-out leaks selection!"
    print(f"[setup] C verifier on {len(heldout)} held-out ALFWorld tasks; gens={GENERATORS} verifier={VERIFIER}", flush=True)

    proto = SameXProtocol(max_turns=30, token_budget=100000)
    enf = SameXEnforcer(proto)
    client = LLMClient(budget_usd_hard_cap=250.0,
                       tracker_path="./results/verifier_tracker.json")
    results = {"config": {"mechanism": "generator + cheap-haiku-verifier self-repair",
                          "generators": GENERATORS, "verifier": VERIFIER, "harness": HARNESS,
                          "n_heldout": len(heldout), "note": "verifier cost counted; no gold-peeking"},
               "gen_verifier": {}}
    for gen in GENERATORS:
        passes = 0; cost = 0.0; pt = []; ptc = []; flagged = 0
        for j, spec in enumerate(heldout):
            task = make_task(spec)
            meter = enf.new_task_meter()
            try:
                r = run_gen_verifier(task, gen, client, proto, meter, purpose=f"verif_{gen.split('-')[1]}_{j}")
                passes += r["success"]; cost += r["cost_usd"]; flagged += r["n_flagged"]
                pt.append(r["success"]); ptc.append(round(r["cost_usd"], 5))
            except Exception as ex:
                print(f"  {gen} t{j} ERR {str(ex)[:40]}", flush=True)
                pt.append(0); ptc.append(0.0)
        n = len(heldout); sr = passes/n
        se = math.sqrt(sr*(1-sr)/n) if n else 0
        cop = cost/passes if passes else None
        results["gen_verifier"][gen] = {"SR": round(sr,4), "SE": round(se,4), "cost": round(cost,4),
                                        "CoP": round(cop,4) if cop else None, "n": n,
                                        "total_flagged": flagged, "per_task": pt, "per_task_cost": ptc}
        print(f"  [{gen}] gen+verifier SR={sr:.3f}±{se:.3f} CoP=${cop if cop else 0:.3f} flagged={flagged}", flush=True)
        json.dump(results, open("./results/verifier_heldout.json", "w"), indent=2)
    print("VERIFIER DONE", flush=True)

if __name__ == "__main__":
    main()
