"""Run a population control with one planning task group.

Agents use random parent selection and fixed communication schemas. The runner
evaluates new agents at checkpoints and records whether the population splits
into groups whose members combine more successfully within each group."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from core.llm_client import LLMClient, BudgetExceeded
from core.agent_runner import run_founder_on_instance, AgentTrial
from core import mag, crossover, saet
from niches.planbench_eval import (
    load_prompt_json, load_domain_config, get_problem_actions, get_problem_pddl,
    DOMAIN_PDDL, VAL_VALIDATE, _DOMAIN_TO_FN,
)

FOUNDER_V1_PATH = Path("./data/founder_genome_v1.json")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- niche evaluator wrapper (single-call agent) ----------------

# Cache instance payloads
_PB_CACHE = {"blocksworld": None}


def _load_blocksworld_instances():
    if _PB_CACHE["blocksworld"] is None:
        inst = load_prompt_json("blocksworld", "task_1_plan_generation")
        _PB_CACHE["blocksworld"] = {i["instance_id"]: i for i in inst}
    return _PB_CACHE["blocksworld"]


def eval_agent_planbench(
    genome: dict[str, Any],
    instance_ids: list[int],
    llm_client: LLMClient,
    purpose: str,
) -> dict[str, Any]:
    """Run agent (via run_founder_on_instance) on a list of BW instances. Return q + cost."""
    by_id = _load_blocksworld_instances()
    data = load_domain_config("blocksworld")
    trials = []
    for iid in instance_ids:
        if iid not in by_id:
            continue
        try:
            t = run_founder_on_instance(
                genome=genome,
                domain="blocksworld",
                instance_id=iid,
                instance_payload=by_id[iid],
                llm_client=llm_client,
                domain_data=data,
                action_set=get_problem_actions("blocksworld", iid),
                text_to_plan_fn=_DOMAIN_TO_FN["blocksworld"],
                domain_pddl=DOMAIN_PDDL["blocksworld"],
                problem_pddl=get_problem_pddl("blocksworld", iid),
                val_bin=VAL_VALIDATE,
                purpose=purpose,
            )
            trials.append(t)
        except Exception as e:
            trials.append(None)
    successes = [t.success for t in trials if t is not None]
    costs = [t.cost_usd for t in trials if t is not None]
    if not successes:
        return {"q": 0.0, "cost_usd": 0.0, "n_success": 0, "n": 0}
    return {
        "q": sum(successes) / len(successes),
        "cost_usd": sum(costs),
        "cost_usd_mean": sum(costs) / len(costs),
        "n_success": sum(successes),
        "n": len(successes),
    }


def evaluate_population_BW(
    population: list[dict[str, Any]],
    llm_client: LLMClient,
    n_tasks: int = 5,
    cost_model: dict | None = None,
    purpose_prefix: str = "m5_pop_eval",
    seed: int = 42,
) -> list[dict[str, float]]:
    """For each agent, run on a small BW subset and compute F = q - λ_c · c_norm."""
    rng = random.Random(seed)
    by_id = _load_blocksworld_instances()
    pool = list(by_id.keys())
    rng.shuffle(pool)
    subset = pool[:n_tasks]
    results = []
    for i, ag in enumerate(population):
        r = eval_agent_planbench(ag["genome"], subset, llm_client, purpose=f"{purpose_prefix}_{i}")
        if cost_model:
            fit = saet.fitness_with_cost(r["q"], r["cost_usd_mean"] if r["n"] > 0 else 0.0,
                                          "planbench_blocksworld", cost_model)
            r["F"] = fit["F"]
            r["c_norm"] = fit["c_norm"]
        else:
            r["F"] = r["q"]
            r["c_norm"] = None
        results.append(r)
    return results


# ---------------- pair viability for RCM ----------------

def pair_hybrid_viability(
    g1: dict[str, Any], g2: dict[str, Any],
    llm_client: LLMClient,
    R: int = 4,
    eval_tasks: int = 3,
    purpose: str = "m5_rcm_pair",
    rng_seed: int = 42,
) -> float:
    """Generate R hybrids from typed_subgraph_crossover, eval each on small BW task subset,
    return mean viability ∈ [0,1] (we use q as viability ≈ V)."""
    rng = random.Random(rng_seed)
    by_id = _load_blocksworld_instances()
    pool = list(by_id.keys())
    rng.shuffle(pool)
    subset = pool[:eval_tasks]
    viab_vals = []
    for r in range(R):
        try:
            child = crossover.typed_subgraph_crossover(g1, g2, rng=rng)
            res = eval_agent_planbench(child, subset, llm_client,
                                        purpose=f"{purpose}_R{r}")
            viab_vals.append(res["q"])
        except Exception:
            viab_vals.append(0.0)
    return float(sum(viab_vals) / max(len(viab_vals), 1))


# ---------------- M5 main loop ----------------

def m5_run(
    N: int = 16, T: int = 20, eval_every: int = 5,
    R_rcm: int = 4, rcm_eval_tasks: int = 3,
    pop_eval_tasks: int = 5,
    seed: int = 42,
    budget_cap: float = 5.0,
    mu: float = 0.05,
    beta: float = 0.0,  # random mating per Thm 3 Assumption 6
    out_path: Path | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    """Single-niche SAET-lite. Disable interface-drift mutations to honor Thm 3."""

    print(f"=== M5 anchor_4 single-niche control ===", flush=True)
    print(f"  N={N}, T={T}, R_rcm={R_rcm}, beta={beta} (random mating),  budget=${budget_cap}",
          flush=True)
    genome_template = mag.load_mag(FOUNDER_V1_PATH)
    cost_model = genome_template.get("cost_model")
    if cost_model is None:
        # fallback
        cost_model = {"lambda_c": 0.5, "c_normalization": {"c_max_usd": {"planbench_blocksworld": 0.001}}}
    print(f"  cost_model: λ_c={cost_model['lambda_c']}, c_max_BW={cost_model['c_normalization']['c_max_usd'].get('planbench_blocksworld')}",
          flush=True)

    # Disable interface-drift mutation ops (per Thm 3 Assumption 6: no interface drift)
    # Re-normalize remaining ops
    up = genome_template["modules"]["update_policy"]
    DROP_OPS = {"modify_tool_interface", "modify_memory_schema", "modify_communication_schema"}
    new_dist = {k: v for k, v in up["mutation_distribution"].items() if k not in DROP_OPS}
    s = sum(new_dist.values())
    new_dist = {k: v / s for k, v in new_dist.items()}
    # Apply to template
    genome_template["modules"]["update_policy"]["mutation_distribution"] = new_dist
    print(f"  M5 disabled mutation ops: {DROP_OPS} (Thm 3 'no interface drift')", flush=True)
    print(f"  remaining ops: {list(new_dist.keys())}", flush=True)

    client = LLMClient(
        budget_usd_hard_cap=budget_cap,
        tracker_path="./budget_tracker.json",
        purpose_tag="m5_anchor4",
    )
    initial_spend = client.total_usd

    rng = random.Random(seed)
    # Init population: N copies of founder_v1 + epsilon mutations
    population = []
    for i in range(N):
        g = mutate(genome_template, mu * 0.5, rng)  # small initial perturbation
        population.append({
            "id": i, "lineage_seed": str(i), "genome": g,
            "parent_ids": [], "birth_gen": 0,
        })
    next_id = N

    # Logs
    log = {
        "phase": "M5_anchor4",
        "config": {
            "N": N, "T": T, "eval_every": eval_every, "R_rcm": R_rcm,
            "rcm_eval_tasks": rcm_eval_tasks, "pop_eval_tasks": pop_eval_tasks,
            "seed": seed, "beta": beta, "mu": mu,
            "niche": "planbench_blocksworld",
            "founder": str(FOUNDER_V1_PATH),
            "mutation_ops_disabled": list(DROP_OPS),
        },
        "started_at": time.time(),
        "rcc_history": [],
        "fitness_history": [],
        "cost_history": [],
        "abort_gate_violations": [],
    }

    t0 = time.time()

    for t in range(T):
        gen_start_spend = client.total_usd
        # 1. evaluate population
        try:
            fit_results = evaluate_population_BW(
                [{"genome": ag["genome"], "id": ag["id"]} for ag in population],
                client, n_tasks=pop_eval_tasks, cost_model=cost_model,
                purpose_prefix=f"m5_t{t}_eval", seed=seed + t,
            )
        except BudgetExceeded as e:
            log["error"] = f"BudgetExceeded at gen {t}: {e}"
            break

        fits = [r["F"] for r in fit_results]
        qs = [r["q"] for r in fit_results]
        log["fitness_history"].append({
            "gen": t,
            "F_mean": float(sum(fits) / len(fits)),
            "F_max": float(max(fits)),
            "F_min": float(min(fits)),
            "q_mean": float(sum(qs) / len(qs)),
        })

        # 2. parent selection (random, β=0)
        n_offspring = N // 2  # half new each gen
        pairs = saet.select_parent_pairs(
            [ag["genome"] for ag in population], fits,
            compat_predictor=None, beta=beta, n_pairs=n_offspring, rng=rng,
        )

        # 3. crossover + mutation
        offspring = []
        for (i, j) in pairs:
            child_g = crossover.typed_subgraph_crossover(
                population[i]["genome"], population[j]["genome"], rng=rng,
            )
            child_g = mutate(child_g, mu, rng)
            offspring.append({
                "id": next_id, "genome": child_g,
                "parent_ids": [population[i]["id"], population[j]["id"]],
                "birth_gen": t + 1,
            })
            next_id += 1

        # 4. survivor selection: top-N from parents + offspring
        try:
            # evaluate offspring
            off_fit = evaluate_population_BW(
                [{"genome": ag["genome"], "id": ag["id"]} for ag in offspring],
                client, n_tasks=pop_eval_tasks, cost_model=cost_model,
                purpose_prefix=f"m5_t{t}_offspring", seed=seed + t + 1000,
            )
        except BudgetExceeded as e:
            log["error"] = f"BudgetExceeded at gen {t} offspring eval: {e}"
            break

        off_fits = [r["F"] for r in off_fit]
        combined = list(zip(population, fits)) + list(zip(offspring, off_fits))
        combined.sort(key=lambda x: -x[1])
        population = [c[0] for c in combined[:N]]

        gen_cost = client.total_usd - gen_start_spend
        log["cost_history"].append({"gen": t, "gen_cost_usd": gen_cost,
                                     "cumulative_usd_in_m5": client.total_usd - initial_spend})

        print(f"  [gen {t+1}/{T}] F̄={log['fitness_history'][-1]['F_mean']:.3f}, "
              f"q̄={log['fitness_history'][-1]['q_mean']:.3f}, "
              f"gen_cost=${gen_cost:.4f}, cum_M5=${client.total_usd-initial_spend:.4f}",
              flush=True)

        # 5. RCM + RCC every eval_every gens
        if (t + 1) % eval_every == 0:
            try:
                rcm = saet.estimate_rcm(
                    [{"genome": ag["genome"], "id": ag["id"]} for ag in population],
                    lambda a, b, sd: pair_hybrid_viability(
                        a["genome"], b["genome"], client, R=R_rcm,
                        eval_tasks=rcm_eval_tasks,
                        purpose=f"m5_t{t+1}_rcm", rng_seed=sd),
                    R=R_rcm, rng_seed=seed + t * 99,
                )
                rcc = saet.rcc_partition(rcm, n_clusters_max=4,
                                          tau_in=0.30, tau_out=0.60, n_min=3)
                rii_info = saet.compute_rii(rcm, rcc["labels"])
                rcc_entry = {
                    "gen": t + 1,
                    "n_clusters": rcc["n_clusters"],
                    "valid_cluster_count": len(rcc["valid_clusters"]),
                    "rii_mean": rii_info["rii_mean"],
                    "K_within_mean": rii_info["K_within_mean"],
                    "K_between_mean": rii_info["K_between_mean"],
                    "valid_clusters": rcc["valid_clusters"],
                    "rii_pairs": rii_info["rii_pairs"],
                }
                log["rcc_history"].append(rcc_entry)
                # ABORT gate check
                if rii_info["rii_mean"] > 0.10:
                    violation = {"gen": t + 1, "rii": rii_info["rii_mean"],
                                  "threshold": 0.10, "violation": True}
                    log["abort_gate_violations"].append(violation)
                    print(f"  ⚠️ GA.M5 alert: gen {t+1} RII={rii_info['rii_mean']:.3f} > 0.10",
                          flush=True)
                else:
                    print(f"  ✅ RCC@gen{t+1}: K_clusters={rcc['n_clusters']}, "
                          f"RII={rii_info['rii_mean']:.3f} < 0.10 (Thm 3 holds)",
                          flush=True)
            except BudgetExceeded as e:
                log["error"] = f"BudgetExceeded at gen {t} RCM: {e}"
                break

        # checkpoint
        if progress_path:
            progress_path.write_text(json.dumps(log, indent=2, default=str))

    # Final analysis
    log["finished_at"] = time.time()
    log["elapsed_sec"] = time.time() - t0
    log["m5_total_spend_usd"] = client.total_usd - initial_spend

    # ABORT critical check: 3 consecutive RCC time-points with RII > 0.10
    consecutive_violations = 0
    max_consec = 0
    for h in log["rcc_history"]:
        if h["rii_mean"] > 0.10:
            consecutive_violations += 1
            max_consec = max(max_consec, consecutive_violations)
        else:
            consecutive_violations = 0
    log["max_consecutive_violations"] = max_consec
    log["ga_m5_critical_abort"] = max_consec >= 3
    log["m5_pass"] = not log["ga_m5_critical_abort"] and len(log["rcc_history"]) >= 1

    if out_path:
        out_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"\n=== M5 DONE: elapsed {log['elapsed_sec']:.0f}s, spent ${log['m5_total_spend_usd']:.4f} ===",
          flush=True)
    print(f"  RCC time-points: {len(log['rcc_history'])}", flush=True)
    print(f"  max consecutive RII > 0.10 violations: {max_consec}", flush=True)
    print(f"  GA.M5 CRITICAL ABORT: {log['ga_m5_critical_abort']}", flush=True)
    print(f"  M5 PASS: {log['m5_pass']}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--R-rcm", type=int, default=4)
    ap.add_argument("--rcm-eval-tasks", type=int, default=3)
    ap.add_argument("--pop-eval-tasks", type=int, default=5)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "m5_anchor4.json"))
    ap.add_argument("--progress", type=str, default=str(RESULTS_DIR / "m5_anchor4_progress.json"))
    args = ap.parse_args()

    log = m5_run(
        N=args.N, T=args.T, eval_every=args.eval_every,
        R_rcm=args.R_rcm, rcm_eval_tasks=args.rcm_eval_tasks,
        pop_eval_tasks=args.pop_eval_tasks,
        seed=args.seed, budget_cap=args.budget,
        out_path=Path(args.out), progress_path=Path(args.progress),
    )
    return 0 if log.get("m5_pass") else 1


# Wire mutate from saet
mutate = saet.mutate


if __name__ == "__main__":
    sys.exit(main())
