"""Run population experiments with configurable interface checks.

The runner evaluates agents on their assigned tasks, selects parents, exchanges
parts, and measures the performance of the new agents. Options control task
groups, migration, mutation, founder agents, and how type mismatches are scored."""
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
from core.agent_runner import run_founder_on_instance
from core import mag, crossover, saet
from niches.planbench_eval import (
    load_prompt_json, load_domain_config, get_problem_actions, get_problem_pddl,
    DOMAIN_PDDL, VAL_VALIDATE, _DOMAIN_TO_FN,
)
from niches.locomo_eval import evaluate_one as locomo_eval_one, sample_locomo_subset
from niches.hotpotqa_eval import evaluate_one as hotpotqa_eval_one, sample_hotpotqa_subset


FOUNDER_V1_PATH = Path("./data/founder_genome_v1.json")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- niche evaluators ----------------

# v5 globals — set by exp1_run
_MISMATCH_MODE = "hard"
_SOFT_PENALTY = 0.3
_MUTATE_TYPE_WEIGHT = None  # None → use mutation_distribution defaults

_PB_BW = None
def _pb_bw_instances():
    global _PB_BW
    if _PB_BW is None:
        inst = load_prompt_json("blocksworld", "task_1_plan_generation")
        _PB_BW = {i["instance_id"]: i for i in inst}
    return _PB_BW


def eval_on_planbench_bw(genome, llm_client, n_tasks=4, purpose="exp1_pb", seed=42):
    by_id = _pb_bw_instances()
    rng = random.Random(seed)
    pool = list(by_id.keys()); rng.shuffle(pool)
    subset = pool[:n_tasks]
    data = load_domain_config("blocksworld")
    s_count = 0; total_cost = 0.0; n_eval = 0
    for iid in subset:
        try:
            t = run_founder_on_instance(
                genome=genome, domain="blocksworld", instance_id=iid,
                instance_payload=by_id[iid], llm_client=llm_client,
                domain_data=data, action_set=get_problem_actions("blocksworld", iid),
                text_to_plan_fn=_DOMAIN_TO_FN["blocksworld"],
                domain_pddl=DOMAIN_PDDL["blocksworld"],
                problem_pddl=get_problem_pddl("blocksworld", iid),
                val_bin=VAL_VALIDATE, purpose=purpose,
                mismatch_mode=_MISMATCH_MODE,
            )
            s_count += int(t.success); total_cost += t.cost_usd; n_eval += 1
        except Exception:
            pass
    return {"q": s_count / max(n_eval, 1), "cost_usd": total_cost,
             "cost_per_task": total_cost / max(n_eval, 1), "n": n_eval}


_LOCOMO_POOL = None
def _locomo_pool():
    global _LOCOMO_POOL
    if _LOCOMO_POOL is None:
        _LOCOMO_POOL = sample_locomo_subset(n=100, seed=42)
    return _LOCOMO_POOL


def eval_on_locomo(genome, llm_client, n_tasks=3, purpose="exp1_locomo", seed=42):
    pool = _locomo_pool()
    rng = random.Random(seed)
    picks = rng.sample(pool, min(n_tasks, len(pool)))
    s_count = 0; total_cost = 0.0; n_eval = 0
    for item in picks:
        try:
            t = locomo_eval_one(
                conv_payload=item["conv_payload"], question=item["question"],
                gold_answer=item["gold_answer"], category=item["category"],
                conversation_id=item["conversation_id"], llm_client=llm_client,
                genome=genome, purpose=purpose,
                mismatch_mode=_MISMATCH_MODE,
            )
            s_count += int(t.success); total_cost += t.cost_usd; n_eval += 1
        except Exception:
            pass
    return {"q": s_count / max(n_eval, 1), "cost_usd": total_cost,
             "cost_per_task": total_cost / max(n_eval, 1), "n": n_eval}



_HQA_POOL_E = None
def _hqa_pool():
    global _HQA_POOL_E
    if _HQA_POOL_E is None:
        _HQA_POOL_E = sample_hotpotqa_subset(n=200, seed=42, config="fullwiki")
    return _HQA_POOL_E


def eval_on_hotpotqa(genome, llm_client, n_tasks=3, purpose="exp1_hqa", seed=42):
    import random as _rnd
    pool = _hqa_pool()
    rng = _rnd.Random(seed)
    picks = rng.sample(pool, min(n_tasks, len(pool)))
    s_count = 0; total_cost = 0.0; n_eval = 0
    for it in picks:
        try:
            t = hotpotqa_eval_one(it, llm_client, genome=genome, purpose=purpose, mismatch_mode=_MISMATCH_MODE)
            s_count += int(t.success); total_cost += t.cost_usd; n_eval += 1
        except Exception:
            pass
    return {"q": s_count / max(n_eval, 1), "cost_usd": total_cost,
             "cost_per_task": total_cost / max(n_eval, 1), "n": n_eval}


NICHE_EVAL = {
    "planbench_blocksworld": eval_on_planbench_bw,
    "locomo": eval_on_locomo,
    "hotpotqa": eval_on_hotpotqa,
}


def evaluate_agent_on_niche(genome, niche, llm_client, n_tasks, cost_model, purpose, seed,
                              niche_multiplier_table=None):
    fn = NICHE_EVAL[niche]
    res = fn(genome, llm_client, n_tasks=n_tasks, purpose=purpose, seed=seed)
    if res["n"] == 0:
        return {"F": 0.0, "q": 0.0, "c_norm": 0.0, "c_usd": 0.0, "n": 0, "niche": niche}
    # v5 soft penalty: q × _SOFT_PENALTY^mismatch_count
    q_eff = res["q"]
    mm_count = genome.get("_type_mismatch_count", 0)
    if _MISMATCH_MODE in ("soft", "rigid") and mm_count > 0:
        q_eff = q_eff * (_SOFT_PENALTY ** mm_count)
        res = dict(res); res["q"] = q_eff; res["_type_mismatch_count"] = mm_count
    # E2 (orchestrator 2026-06-26): pass genome + niche_multiplier_table to apply task_focus multiplier
    fit = saet.fitness_with_cost(q_eff, res["cost_per_task"], niche, cost_model,
                                   genome=genome, niche_multiplier_table=niche_multiplier_table)
    return {**res, **fit, "niche": niche}


def pair_hybrid_viability(g1, g2, niche, llm_client, R=4, eval_tasks=3,
                           purpose="exp1_rcm", rng_seed=42):
    rng = random.Random(rng_seed)
    fn = NICHE_EVAL[niche]
    viab_vals = []
    # E1: tag hybrid with combined lineage id so rigid mode's is_hybrid() detects it
    l1 = g1.get("_lineage_id", "L?")
    l2 = g2.get("_lineage_id", "L?")
    hybrid_lid = l1 if l1 == l2 else f"{l1}+{l2}"
    for r in range(R):
        try:
            child = crossover.typed_subgraph_crossover(g1, g2, rng=rng)
            child["_lineage_id"] = hybrid_lid
            res = fn(child, llm_client, n_tasks=eval_tasks,
                     purpose=f"{purpose}_R{r}", seed=rng_seed + r)
            viab_vals.append(res["q"])
        except Exception:
            viab_vals.append(0.0)
    return float(sum(viab_vals) / max(len(viab_vals), 1))


def _base_lineage(lid):
    """Strip hybrid '+' suffixes to the first base lineage token (e.g. 'L0+L1' -> 'L0')."""
    return str(lid).split("+")[0]


def dual_verification_probe(population, niches, client, cost_model, niche_multiplier_table,
                             gen, seed, n_pairs=6, eval_tasks=2, soft_penalty=0.3):
    """Compare type checks and task scores for the same new agents.

    The probe exchanges parts between lineages, records strict interface failures,
    and evaluates the resulting agents with the soft mismatch setting. It saves
    both measurements and their difference."""
    global _MISMATCH_MODE
    rng = random.Random(seed)
    # Enumerate candidate cross-lineage pairs (different base lineage)
    n = len(population)
    idxs = list(range(n))
    rng.shuffle(idxs)
    pairs = []
    for ai in range(n):
        a = idxs[ai]
        la = _base_lineage(population[a]["genome"].get("_lineage_id", str(a)))
        for bi in range(ai + 1, n):
            b = idxs[bi]
            lb = _base_lineage(population[b]["genome"].get("_lineage_id", str(b)))
            if la != lb:
                pairs.append((a, b))
        if len(pairs) >= n_pairs * 4:
            break
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    if not pairs:
        return {"external_rii": None, "internal_task_fitness": None,
                "internal_soft_q": None, "gap": None, "n_hybrid_pairs": 0,
                "n_mechanism_fail": 0}

    niche = niches[0]  # probe on primary niche (cheapest, planbench)
    fn = NICHE_EVAL[niche]
    ext_fail = []   # 1.0 if mechanism type-check FAILS (isolated), else 0.0
    int_q = []
    int_F = []
    saved_mode = _MISMATCH_MODE
    for (a, b) in pairs:
        ga = population[a]["genome"]
        gb = population[b]["genome"]
        la = ga.get("_lineage_id", str(a))
        lb = gb.get("_lineage_id", str(b))
        hybrid = crossover.typed_subgraph_crossover(ga, gb, rng=rng)
        hybrid["_lineage_id"] = f"{la}+{lb}"
        # --- EXTERNAL: mechanism-level strict interface check (NO LLM) ---
        ok_strict, _reason = saet.check_strict_interface(hybrid)
        ext_fail.append(0.0 if ok_strict else 1.0)
        # --- INTERNAL: forced-soft LLM eval (prompt-adaptation self-verification) ---
        try:
            _MISMATCH_MODE = "soft"
            res = fn(hybrid, client, n_tasks=eval_tasks,
                     purpose=f"selfrepair_g{gen}_dualprobe_{niche}",
                     seed=rng.randint(0, 10**6))
            q = res.get("q", 0.0)
            mm = hybrid.get("_type_mismatch_count", 0)
            q_eff = q * (soft_penalty ** mm) if mm > 0 else q
            fit = saet.fitness_with_cost(q_eff, res.get("cost_per_task", 0.0), niche,
                                         cost_model, genome=hybrid,
                                         niche_multiplier_table=niche_multiplier_table)
            int_q.append(q)
            int_F.append(fit["F"])
        except Exception:
            int_q.append(0.0)
            int_F.append(0.0)
        finally:
            _MISMATCH_MODE = saved_mode

    external_rii = sum(ext_fail) / len(ext_fail)
    internal_soft_q = sum(int_q) / len(int_q)
    internal_task_fitness = sum(int_F) / len(int_F)
    gap = external_rii - internal_soft_q
    return {
        "external_rii": external_rii,
        "internal_task_fitness": internal_task_fitness,
        "internal_soft_q": internal_soft_q,
        "gap": gap,
        "n_hybrid_pairs": len(pairs),
        "n_mechanism_fail": int(sum(ext_fail)),
    }


def assign_niches(N, niches, t, m_migration, rng, population=None):
    """Assign task groups using migration and task focus.

    With the configured migration probability, choose a random group. Otherwise,
    use the group matching the agent's task_focus, or distribute general agents
    across groups in order."""
    K = len(niches)
    FOCUS_TO_NICHE = {
        "planning": "planbench_blocksworld",
        "memory": "locomo",
        "retrieval": "hotpotqa",
    }
    out = []
    for i in range(N):
        if rng.random() < m_migration:
            base = rng.choice(niches)
        else:
            tf = "none"
            if population is not None and i < len(population):
                ag = population[i]
                if isinstance(ag, dict):
                    tf = ag.get("genome", {}).get("task_focus", "none")
            matched = FOCUS_TO_NICHE.get(tf)
            if matched is not None and matched in niches:
                base = matched
            else:
                base = niches[i % K]
        out.append(base)
    return out


# Assign task groups using each agent's task_focus setting.
# Specialist agents go to matched niche always; generalists round-robin.
TASK_FOCUS_TO_NICHE = {
    "planning": "planbench_blocksworld",
    "memory": "locomo",
    "retrieval": "hotpotqa",
}


def assign_niches_task_focus(population, niches, m_migration, rng):
    """Assign specialists to their matching task group.

    Agents without a specific task focus are distributed across task groups."""
    out = []
    gen_idx = 0
    for ag in population:
        tf = ag["genome"].get("task_focus", "none")
        if tf in TASK_FOCUS_TO_NICHE and TASK_FOCUS_TO_NICHE[tf] in niches:
            out.append(TASK_FOCUS_TO_NICHE[tf])
        else:
            base = niches[gen_idx % len(niches)]
            gen_idx += 1
            if rng.random() < m_migration:
                base = rng.choice(niches)
            out.append(base)
    return out


def niche_bucket_select(combined, niches, N):
    """Select survivors within task groups, then fill any remaining slots.

    Reserve an equal share of slots for each group and take its highest-scoring
    agents. Fill empty slots with the best remaining agents across all groups."""
    K = len(niches)
    per_niche = N // K  # e.g. 32//3 = 10
    selected = []
    selected_ids = set()
    # Bucket by assigned niche
    bucketed = {n: [] for n in niches}
    for item in combined:
        ag, fit, nch = item
        if nch in bucketed:
            bucketed[nch].append((ag, fit, nch))
    # Take top per_niche from each bucket
    for nch in niches:
        bucketed[nch].sort(key=lambda x: -x[1])
        chosen = bucketed[nch][:per_niche]
        for c in chosen:
            selected.append(c[0])
            selected_ids.add(id(c[0]))
    # Fill remaining N - K*per_niche slots from global top among unselected
    remaining_slots = N - len(selected)
    if remaining_slots > 0:
        leftover = [c for c in combined if id(c[0]) not in selected_ids]
        leftover.sort(key=lambda x: -x[1])
        for c in leftover[:remaining_slots]:
            selected.append(c[0])
            selected_ids.add(id(c[0]))
    # Edge case: if some niche bucket was empty (no agents assigned), we may have
    # < N. Fill any remaining slots from leftover (already attempted above).
    return selected[:N]


def exp1_run(
    niches, N=32, T=50, eval_every=5, R_rcm=6, rcm_eval_tasks=3,
    pop_eval_tasks=4, mu=0.05, beta=2.0, m_migration=0.10,
    seed=42, budget_cap=300.0, tau_in=0.30, tau_out=0.50,
    founder_path=None, lambda_c_override=None,
    mismatch_mode='hard', soft_penalty=0.3, mutate_type_weight=None,
    out_path=None, progress_path=None,
    hand_seed=False, n_lineages=3, no_crossover=False, crossover_op='typed',
    multi_niche_eval=False, dual_verification=False, dv_pairs=6,
    n_mismatch_edges=7, skip_population_rcc=False,
):
    print(f"=== Exp 1 cell_3 main run ===", flush=True)
    print(f"  niches={niches} (K={len(niches)})", flush=True)
    print(f"  N={N}, T={T}, eval_every={eval_every}, R_rcm={R_rcm}, beta={beta}, "
          f"m={m_migration}, budget=${budget_cap}", flush=True)
    # v5 mode flags (module-level so niche evaluators can read)
    global _MISMATCH_MODE, _SOFT_PENALTY, _MUTATE_TYPE_WEIGHT
    _MISMATCH_MODE = mismatch_mode
    _SOFT_PENALTY = soft_penalty
    _MUTATE_TYPE_WEIGHT = mutate_type_weight
    print(f"  mismatch_mode={mismatch_mode}, soft_penalty={soft_penalty}, mutate_type_weight={mutate_type_weight}", flush=True)

    fp = founder_path if founder_path else FOUNDER_V1_PATH
    genome_template = mag.load_mag(fp)
    cost_model = genome_template.get("cost_model", {})
    if lambda_c_override is not None:
        cost_model = dict(cost_model)
        cost_model["lambda_c"] = float(lambda_c_override)
        print(f"  lambda_c override: {cost_model['lambda_c']}", flush=True)

    # E2 (orchestrator 2026-06-26): pull niche_multiplier table from v3 genome if present
    nm_block = genome_template.get("niche_multiplier") or {}
    niche_multiplier_table = nm_block.get("multiplier_table")
    if niche_multiplier_table:
        print(f"  E2 niche_multiplier active: {list(niche_multiplier_table.keys())}", flush=True)

    client = LLMClient(
        budget_usd_hard_cap=budget_cap,
        tracker_path="./budget_tracker.json",
        purpose_tag="exp1_cell3",
    )
    initial_spend = client.total_usd

    rng = random.Random(seed)
    population = []
    if hand_seed:
        # v17 hand-seeded existence proof (orchestrator 2026-06-27 22:00 UTC):
        # 3 lineages × N/3 agents each. Each lineage has distinct planner.output_type
        # AND workflow.input_type so within-lineage agents pass RIGID interface check
        # (planner.out == workflow.in), but cross-lineage hybrids fail (planner.out
        # from lineage A != workflow.in from lineage B) → rigid mode q=0.
        #
        # task_focus per lineage matches its preferred niche (×10/×30/×15 mult).
        # v17.1 (2026-06-27 23:16 UTC): ALL 7 STRICT_TYPE_EDGES need lineage-specific
        # types. Single-edge override (v17.0) only blocked ~30% of cross-lineage crossovers
        # because typed_subgraph_crossover bias inherits planner+workflow as same subgraph
        # 71% of the time. Solution: apply lineage suffix to every edge endpoint.
        import copy
        # v17.2 (orchestrator 2026-06-28 04:13 UTC): support 2 or 3 lineages for Exp1.2 etc.
        # Lineages match niches in order: niche[0]→LA→planning, niche[1]→LB→memory, niche[2]→LC→retrieval
        ALL_LINEAGE_SPECS = [
            {"id": "LA", "suffix": "_LA", "focus": "planning"},
            {"id": "LB", "suffix": "_LB", "focus": "memory"},
            {"id": "LC", "suffix": "_LC", "focus": "retrieval"},
        ]
        lineage_specs = ALL_LINEAGE_SPECS[:n_lineages]
        ALL_EDGES_TO_OVERRIDE = [
            ("planner", "workflow"),
            ("workflow", "memory"),
            ("memory", "memory_compactor"),
            ("tools", "tool_chain_compiler"),
            ("verifier", "communication"),
            ("verifier", "update_policy"),
            ("verifier", "error_log"),
        ]
        # Self-repair severity sweep (orchestrator 2026-07-23): mismatch-severity grading.
        # n_mismatch_edges=7 → severe (all edges broken, both verifs fail, gap≈0).
        # n_mismatch_edges=3 → moderate (partial; soft LLM can fake-repair some, mechanism
        #   still flags residual → gap>0, the verification-reliability sweet spot).
        # n_mismatch_edges=1 → mild (mostly matchable, both verifs mostly pass, gap small).
        # Only the first n_mismatch_edges get lineage suffix; the rest stay pristine
        # (cross-lineage compatible).
        n_edges = max(1, min(int(n_mismatch_edges), len(ALL_EDGES_TO_OVERRIDE)))
        EDGES_TO_OVERRIDE = ALL_EDGES_TO_OVERRIDE[:n_edges]
        per_lineage = N // n_lineages
        if per_lineage * n_lineages != N:
            raise ValueError(f"hand_seed requires N divisible by n_lineages, got N={N}, n_lineages={n_lineages}")
        agent_idx = 0
        for spec in lineage_specs:
            for k in range(per_lineage):
                g = copy.deepcopy(genome_template)
                if "modules" in g:
                    # Dedupe (verifier appears as src 3 times): apply suffix once per
                    # (module, direction) pair.
                    out_targets = set()
                    in_targets = set()
                    for (src, dst) in EDGES_TO_OVERRIDE:
                        out_targets.add(src)
                        in_targets.add(dst)
                    for mod in out_targets:
                        if mod in g["modules"]:
                            ot = g["modules"][mod].get("output_type", "?")
                            g["modules"][mod]["output_type"] = f"{ot}{spec['suffix']}"
                    for mod in in_targets:
                        if mod in g["modules"]:
                            it = g["modules"][mod].get("input_type", "?")
                            g["modules"][mod]["input_type"] = f"{it}{spec['suffix']}"
                g["task_focus"] = spec["focus"]
                g["_lineage_id"] = f"{spec['id']}_{k}"
                population.append({"id": agent_idx, "lineage_seed": spec["id"], "genome": g,
                                    "parent_ids": [], "birth_gen": 0})
                agent_idx += 1
        print(f"  v17.2 HAND-SEEDED ({n_edges}-of-7 edges): {len(lineage_specs)} lineages × {per_lineage} agents = N={N}", flush=True)
        for spec in lineage_specs:
            print(f"    {spec['id']}: suffix='{spec['suffix']}' on {len(EDGES_TO_OVERRIDE)} edges, focus={spec['focus']}", flush=True)
    if no_crossover:
        print(f"  v17.2 NO-CROSSOVER mode: offspring = mutated parent only (no recombination)", flush=True)
    else:
        for i in range(N):
            g = saet.mutate(genome_template, mu * 0.5, rng, mutate_type_weight=_MUTATE_TYPE_WEIGHT)
            # E1: tag genome with unique founder lineage id
            g["_lineage_id"] = f"L{i}"
            population.append({"id": i, "lineage_seed": str(i), "genome": g,
                                "parent_ids": [], "birth_gen": 0})
    next_id = N

    log = {
        "phase": "Exp1_cell3",
        "config": {
            "niches": niches, "N": N, "T": T, "eval_every": eval_every,
            "R_rcm": R_rcm, "rcm_eval_tasks": rcm_eval_tasks,
            "pop_eval_tasks": pop_eval_tasks, "mu": mu, "beta": beta,
            "m_migration": m_migration, "seed": seed,
            "tau_in": tau_in, "tau_out": tau_out,
            "founder": str(fp), "lambda_c": cost_model.get("lambda_c"),
            "mismatch_mode": mismatch_mode, "soft_penalty": soft_penalty, "mutate_type_weight": mutate_type_weight,
            "multi_niche_eval": multi_niche_eval,
            "dual_verification": dual_verification,
            "n_mismatch_edges": n_mismatch_edges,
            "skip_population_rcc": skip_population_rcc,
        },
        "started_at": time.time(),
        "rcc_history": [], "fitness_history": [], "cost_history": [],
        "niche_assignment_history": [],
        "per_agent_niche_fitness": {},  # Tier A: per-agent multi-niche logging
        "verification_signals": {},     # Self-repair: external(mechanism) vs internal(LLM) per checkpoint
    }
    t0 = time.time()

    for t in range(T):
        gen_start = client.total_usd
        # E2 Fix B: use task_focus-aware assignment when niche_multiplier_table is present (v3+)
        if niche_multiplier_table is not None:
            niche_assign = assign_niches_task_focus(population, niches, m_migration, rng)
        else:
            niche_assign = assign_niches(N, niches, t, m_migration, rng, population=population)
        log["niche_assignment_history"].append({"gen": t, "assign": niche_assign})

        try:
            pop_eval = []
            for i, ag in enumerate(population):
                niche = niche_assign[i]
                res = evaluate_agent_on_niche(
                    ag["genome"], niche, client, pop_eval_tasks,
                    cost_model, purpose=f"exp1_t{t}_eval_{niche}",
                    seed=seed + t * 100 + i,
                    niche_multiplier_table=niche_multiplier_table,
                )
                pop_eval.append(res)
        except BudgetExceeded as e:
            log["error"] = f"BudgetExceeded at gen {t} pop_eval: {e}"; break

        fits = [r["F"] for r in pop_eval]
        qs = [r["q"] for r in pop_eval]
        by_niche = {}
        for niche in niches:
            subset = [r for r in pop_eval if r["niche"] == niche]
            if subset:
                by_niche[niche] = {
                    "F_mean": sum(r["F"] for r in subset) / len(subset),
                    "q_mean": sum(r["q"] for r in subset) / len(subset),
                    "n_agents": len(subset),
                }
        log["fitness_history"].append({
            "gen": t, "F_mean": sum(fits)/len(fits), "F_max": max(fits),
            "q_mean": sum(qs)/len(qs), "by_niche": by_niche,
        })

        n_offspring = N // 2
        pairs = saet.select_parent_pairs(
            [ag["genome"] for ag in population], fits,
            compat_predictor=None, beta=beta, n_pairs=n_offspring, rng=rng,
        )

        offspring = []
        import copy as _copy
        for (i, j) in pairs:
            if no_crossover:
                # Exp 2 b1: per-niche cluster, asexual reproduction. Child = mutated parent i only.
                child_g = saet.mutate(_copy.deepcopy(population[i]["genome"]), mu, rng,
                                       mutate_type_weight=_MUTATE_TYPE_WEIGHT)
                # No hybrid — lineage preserved from parent i
                child_g["_lineage_id"] = population[i]["genome"].get("_lineage_id", str(population[i]["id"]))
                offspring.append({"id": next_id, "genome": child_g,
                                   "parent_ids": [population[i]["id"]],
                                   "birth_gen": t + 1})
            else:
                if crossover_op == 'uniform':
                    child_g = crossover.uniform_genome_crossover(
                        population[i]["genome"], population[j]["genome"], rng=rng)
                else:  # 'typed' (default)
                    child_g = crossover.typed_subgraph_crossover(
                        population[i]["genome"], population[j]["genome"], rng=rng)
                child_g = saet.mutate(child_g, mu, rng, mutate_type_weight=_MUTATE_TYPE_WEIGHT)
                # E1: combine lineage ids — child is hybrid if parents differ
                l_i = population[i]["genome"].get("_lineage_id", str(population[i]["id"]))
                l_j = population[j]["genome"].get("_lineage_id", str(population[j]["id"]))
                child_g["_lineage_id"] = l_i if l_i == l_j else f"{l_i}+{l_j}"
                offspring.append({"id": next_id, "genome": child_g,
                                   "parent_ids": [population[i]["id"], population[j]["id"]],
                                   "birth_gen": t + 1})
            next_id += 1

        try:
            # Fix B' (orchestrator 2026-06-27 09:28 UTC): offspring niche assignment is
            # task_focus-aware too (was random — bug). Reuse assign_niches_task_focus
            # if multipliers are active, else round-robin/migration.
            if niche_multiplier_table is not None:
                off_assign = assign_niches_task_focus(offspring, niches, m_migration, rng)
            else:
                off_assign = [rng.choice(niches) for _ in offspring]
            off_eval = []
            for k, ag in enumerate(offspring):
                niche = off_assign[k]
                res = evaluate_agent_on_niche(
                    ag["genome"], niche, client, pop_eval_tasks,
                    cost_model, purpose=f"exp1_t{t}_off_{niche}",
                    seed=seed + t * 100 + 1000 + k,
                    niche_multiplier_table=niche_multiplier_table,
                )
                off_eval.append(res)
        except BudgetExceeded as e:
            log["error"] = f"BudgetExceeded at gen {t} offspring: {e}"; break

        off_fits = [r["F"] for r in off_eval]
        # Fix B' (orchestrator 2026-06-27 09:28 UTC): niche-conditional survivor selection.
        # Without this, global top-K lets one specialist sweep entire population (v10 bug).
        # Per-niche bucket reserves N/K slots per niche; remainder by global F.
        if niche_multiplier_table is not None:
            combined = list(zip(population, fits, niche_assign)) + list(zip(offspring, off_fits, off_assign))
            population = niche_bucket_select(combined, niches, N)
        else:
            combined = list(zip(population, fits)) + list(zip(offspring, off_fits))
            combined.sort(key=lambda x: -x[1])
            population = [c[0] for c in combined[:N]]

        gen_cost = client.total_usd - gen_start
        cum = client.total_usd - initial_spend
        log["cost_history"].append({"gen": t, "gen_cost": gen_cost, "cum": cum})
        print(f"  [gen {t+1}/{T}] F̄={log['fitness_history'][-1]['F_mean']:.3f}, "
              f"q̄={log['fitness_history'][-1]['q_mean']:.3f}, "
              f"gen_cost=${gen_cost:.4f}, cum=${cum:.4f}", flush=True)

        if (t + 1) % eval_every == 0:
            # orchestrator 2026-07-23 09:30 (option c): --skip-population-rcc disables the
            # expensive 24×24 population RCM (1800 soft LLM calls/checkpoint) while KEEPING
            # the cheap dual-verification probe (external_rii) + per-gen F̄ (internal
            # success). For soft self-repair arms where soft-population RII is not needed.
            if not skip_population_rcc:
                try:
                    rcm_niche = niches[0]
                    rcm = saet.estimate_rcm(
                        [{"genome": ag["genome"], "id": ag["id"]} for ag in population],
                        lambda a, b, sd: pair_hybrid_viability(
                            a["genome"], b["genome"], rcm_niche, client,
                            R=R_rcm, eval_tasks=rcm_eval_tasks,
                            purpose=f"exp1_t{t+1}_rcm", rng_seed=sd),
                        R=R_rcm, rng_seed=seed + t * 99,
                    )
                    rcc = saet.rcc_partition(rcm, n_clusters_max=5,
                                              tau_in=tau_in, tau_out=tau_out, n_min=3)
                    rii_info = saet.compute_rii(rcm, rcc["labels"])
                    entry = {
                        "gen": t + 1, "rcm_eval_niche": rcm_niche,
                        "n_clusters": rcc["n_clusters"],
                        "valid_cluster_count": len(rcc["valid_clusters"]),
                        "rii_mean": rii_info["rii_mean"],
                        "K_within_mean": rii_info["K_within_mean"],
                        "K_between_mean": rii_info["K_between_mean"],
                        "K_w_minus_K_b": (rii_info["K_within_mean"] - rii_info["K_between_mean"])
                                           if rii_info["K_between_mean"] is not None else None,
                        "valid_clusters": rcc["valid_clusters"],
                        "rii_pairs": rii_info["rii_pairs"],
                    }
                    log["rcc_history"].append(entry)
                    signal = "HOT" if rii_info["rii_mean"] > 0.25 else "warm" if rii_info["rii_mean"] > 0.10 else "cold"
                    print(f"  ✦ RCC@gen{t+1}: K_clusters={rcc['n_clusters']}, "
                          f"valid={len(rcc['valid_clusters'])}, RII={rii_info['rii_mean']:.3f} ({signal}), "
                          f"K_w-K_b={entry['K_w_minus_K_b']}", flush=True)
                except BudgetExceeded as e:
                    log["error"] = f"BudgetExceeded gen {t} RCM: {e}"; break

            # Tier A (orchestrator 2026-06-30 19:04 UTC): per-agent multi-niche fitness logging.
            # Evaluate every agent on ALL niches at each RCC checkpoint, so we can detect
            # individual super-agents (top across all 3 niches).
            if multi_niche_eval:
                try:
                    mne_start = client.total_usd
                    gen_label = f"gen_{t+1}"
                    log["per_agent_niche_fitness"][gen_label] = {}
                    for ag_idx, ag in enumerate(population):
                        agent_label = f"agent_{ag_idx}"
                        log["per_agent_niche_fitness"][gen_label][agent_label] = {}
                        for niche in niches:
                            res = evaluate_agent_on_niche(
                                ag["genome"], niche, client, pop_eval_tasks,
                                cost_model, purpose=f"exp1_t{t+1}_mne_{niche}",
                                seed=seed + (t + 1) * 1000 + ag_idx * 10 + niches.index(niche),
                                niche_multiplier_table=niche_multiplier_table,
                            )
                            log["per_agent_niche_fitness"][gen_label][agent_label][niche] = {
                                "F": res["F"], "q": res["q"], "q_eff": res.get("q_eff", res["q"]),
                                "niche_multiplier": res.get("niche_multiplier", 1.0),
                                "task_focus": res.get("task_focus", ag["genome"].get("task_focus", "none")),
                                "lineage_id": ag["genome"].get("_lineage_id", str(ag_idx)),
                            }
                    mne_cost = client.total_usd - mne_start
                    # orchestrator risk #2: cost-protect against runaway multi-niche eval
                    # Per-gen multi-niche budget = 3 niches × N agents × pop_eval_tasks tasks
                    # Soft cap: if it exceeds 3× pop_eval cost as warning. Hard abort if > 5x.
                    if "_warn_count" not in log:
                        log["_warn_count"] = 0
                    pop_eval_baseline = log["cost_history"][-1]["gen_cost"] if log["cost_history"] else mne_cost
                    if pop_eval_baseline > 0 and mne_cost > 5 * pop_eval_baseline:
                        log["error"] = f"multi_niche_eval cost runaway at gen {t+1}: ${mne_cost:.3f} (>5x pop baseline ${pop_eval_baseline:.3f})"
                        print(f"  ⚠️ multi-niche eval cost runaway, aborting", flush=True)
                        break
                    print(f"  ⊕ multi-niche eval gen{t+1}: $${mne_cost:.4f} (N={len(population)} × {len(niches)} niches)", flush=True)
                except BudgetExceeded as e:
                    log["error"] = f"BudgetExceeded gen {t} multi_niche_eval: {e}"; break

            # Self-repair pivot (orchestrator 2026-07-23): matched dual verification-signal probe.
            # For the SAME cross-lineage hybrids, log external(mechanism) RII vs internal(LLM)
            # soft-fitness. gap = external - internal quantifies verification-reliability gap.
            if dual_verification:
                try:
                    dv_start = client.total_usd
                    dv = dual_verification_probe(
                        population, niches, client, cost_model, niche_multiplier_table,
                        gen=t + 1, seed=seed + (t + 1) * 7, n_pairs=dv_pairs,
                        eval_tasks=rcm_eval_tasks, soft_penalty=soft_penalty,
                    )
                    # orchestrator 2026-07-23 09:17: population-level internal-success signal.
                    # Macro verification-reliability gap = population succeeds at its tasks
                    # (F̄ / q̄ high = internal 'system works') while mechanism check still
                    # flags residual interface incompatibility (external_rii > 0 = not
                    # actually repaired). This is the population-level gap, complementing
                    # the micro dual-verif probe's internal_soft_q.
                    cur_fit = log["fitness_history"][-1] if log["fitness_history"] else {}
                    pop_F_mean = cur_fit.get("F_mean")
                    pop_q_mean = cur_fit.get("q_mean")
                    pop_by_niche_F = {n: v.get("F_mean") for n, v in cur_fit.get("by_niche", {}).items()}
                    rcc_rii = log["rcc_history"][-1]["rii_mean"] if log["rcc_history"] else None
                    dv["pop_F_mean"] = pop_F_mean
                    dv["pop_q_mean"] = pop_q_mean
                    dv["pop_by_niche_F_mean"] = pop_by_niche_F
                    dv["pop_rcc_rii"] = rcc_rii
                    # macro gap (orchestrator 2026-07-23 10:23): use external_rii from the
                    # dual-verif probe (always available, even with --skip-population-rcc),
                    # NOT rcc_rii (None when population RCC skipped).
                    # external_rii = fraction FAILING mechanism check ("not repaired").
                    # (1 - external_rii) = fraction mechanism-verified-repaired.
                    # macro_gap = internal task success (q̄) - external verified-repair rate.
                    # Positive = population succeeds at tasks MORE than mechanism confirms
                    # repair = verification-reliability gap.
                    ext_rii = dv.get("external_rii")
                    dv["macro_gap_qbar_vs_external"] = (
                        (pop_q_mean - (1.0 - ext_rii))
                        if (pop_q_mean is not None and ext_rii is not None) else None
                    )
                    # keep legacy field (rcc-based) for back-compat; None when RCC skipped
                    dv["macro_gap_qbar_vs_rii"] = (
                        (pop_q_mean - rcc_rii) if (pop_q_mean is not None and rcc_rii is not None) else None
                    )
                    log["verification_signals"][f"gen_{t+1}"] = dv
                    dv_cost = client.total_usd - dv_start
                    _g = dv["gap"]
                    _e = dv["external_rii"]
                    _i = dv["internal_soft_q"]
                    print(f"  ⧉ dual-verif gen{t+1}: external_rii={_e}, internal_soft_q={_i}, "
                          f"gap={_g} ({dv['n_mechanism_fail']}/{dv['n_hybrid_pairs']} mech-fail) | "
                          f"pop_F̄={pop_F_mean}, pop_q̄={pop_q_mean}, "
                          f"macro_gap={dv['macro_gap_qbar_vs_external']}, ${dv_cost:.4f}", flush=True)
                except BudgetExceeded as e:
                    log["error"] = f"BudgetExceeded gen {t} dual_verification: {e}"; break

        if progress_path:
            progress_path.write_text(json.dumps(log, indent=2, default=str))

    persistent = []
    if len(log["rcc_history"]) >= 3:
        recent = log["rcc_history"][-3:]
        if all(r["rii_mean"] > 0.25 for r in recent):
            persistent.append("global_RII_persistent_3_timepoints")

    log["finished_at"] = time.time()
    log["elapsed_sec"] = time.time() - t0
    log["total_spend_usd"] = client.total_usd - initial_spend
    log["n_rcc_time_points"] = len(log["rcc_history"])
    log["max_rii_observed"] = max([h["rii_mean"] for h in log["rcc_history"]], default=0.0)
    log["final_rii"] = log["rcc_history"][-1]["rii_mean"] if log["rcc_history"] else None
    log["persistent_species_signals"] = persistent
    log["g1_pass_signal"] = (log["max_rii_observed"] > 0.25
                              and any(h["K_w_minus_K_b"] is not None and h["K_w_minus_K_b"] > 0.20
                                       for h in log["rcc_history"]))

    if out_path:
        out_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"\n=== Exp 1 cell_3 DONE: elapsed {log['elapsed_sec']:.0f}s, spent ${log['total_spend_usd']:.4f} ===", flush=True)
    print(f"  RCC time-points: {log['n_rcc_time_points']}", flush=True)
    print(f"  max RII observed: {log['max_rii_observed']:.3f}", flush=True)
    print(f"  final RII: {log['final_rii']}", flush=True)
    print(f"  G1 signal: {log['g1_pass_signal']}", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niches", nargs="+",
                    default=["planbench_blocksworld", "locomo", "hotpotqa"])
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--R-rcm", type=int, default=6)
    ap.add_argument("--rcm-eval-tasks", type=int, default=3)
    ap.add_argument("--pop-eval-tasks", type=int, default=4)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--m-migration", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--tau-in", type=float, default=0.30)
    ap.add_argument("--tau-out", type=float, default=0.50)
    ap.add_argument("--founder", type=str, default=None, help='Path to founder genome JSON. Defaults to FOUNDER_V1_PATH.')
    ap.add_argument("--lambda-c", dest='lambda_c', type=float, default=None, help='Override cost_model.lambda_c')
    ap.add_argument("--mismatch-mode", dest='mismatch_mode', type=str, default='hard', choices=['hard', 'soft', 'rigid'],
                    help='hard=v4 q=0 skip on TYPE_MISMATCH; soft=v5 continue with q × 0.3^count')
    ap.add_argument("--soft-penalty", dest='soft_penalty', type=float, default=0.3, help='Multiplier applied per mismatch when --mismatch-mode soft')
    ap.add_argument("--mutate-type-weight", dest='mutate_type_weight', type=float, default=None,
                    help='Override weight for mutate_output_type / mutate_input_type ops (e.g. 0.15)')
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "exp1_cell3.json"))
    ap.add_argument("--progress", type=str, default=str(RESULTS_DIR / "exp1_cell3_progress.json"))
    ap.add_argument("--hand-seed", dest='hand_seed', action='store_true',
                    help='v17: hand-seed 3 lineages × N/3 with distinct planner.output_type/'
                         'workflow.input_type per lineage. N must be divisible by 3.')
    ap.add_argument("--n-lineages", dest='n_lineages', type=int, default=3,
                    choices=[2, 3],
                    help='v17.2: number of lineages to hand-seed (2 or 3). N must be divisible by n_lineages.')
    ap.add_argument("--no-crossover", dest='no_crossover', action='store_true',
                    help='v17.2: disable crossover (offspring = mutated parent i only). For Exp 2 b1 per-niche cluster baseline.')
    ap.add_argument("--crossover-op", dest='crossover_op', type=str, default='typed',
                    choices=['typed', 'uniform'],
                    help='Exp 7C: crossover variant. typed=current typed_subgraph_crossover; '
                         'uniform=per-module 50/50 inherit (may produce type mismatches by design).')
    ap.add_argument("--multi-niche-eval", dest='multi_niche_eval', action='store_true',
                    help='Tier A: at each RCC checkpoint, evaluate EVERY agent on ALL niches '
                         '(not just assigned niche). Cost: ~3x baseline per checkpoint. '
                         'Enables per-agent super-agent detection.')
    ap.add_argument("--dual-verification", dest='dual_verification', action='store_true',
                    help='At each checkpoint, compare strict type checks with task scores '
                         'for the same new agents. Save both measurements and their difference.')
    ap.add_argument("--dv-pairs", dest='dv_pairs', type=int, default=6,
                    help='Number of cross-lineage hybrid pairs to probe per checkpoint '
                         'for --dual-verification (default 6).')
    ap.add_argument("--n-mismatch-edges", dest='n_mismatch_edges', type=int, default=7,
                    help='Self-repair severity sweep: number of STRICT_TYPE_EDGES (1-7) to '
                         'apply lineage suffix to for --hand-seed. 7=severe (all broken), '
                         '3=moderate (gap sweet spot), 1=mild. Rest stay cross-lineage '
                         'compatible.')
    ap.add_argument("--skip-population-rcc", dest='skip_population_rcc', action='store_true',
                    help='Option (c): skip the expensive 24×24 population RCM/RCC at each '
                         'checkpoint (soft mode = 1800 LLM calls). Keeps --dual-verification '
                         'probe (external_rii) + per-gen F̄ (internal success). For fast soft '
                         'self-repair arms where soft-population RII is not needed.')
    args = ap.parse_args()

    exp1_run(
        niches=args.niches, N=args.N, T=args.T, eval_every=args.eval_every,
        R_rcm=args.R_rcm, rcm_eval_tasks=args.rcm_eval_tasks,
        pop_eval_tasks=args.pop_eval_tasks, mu=args.mu, beta=args.beta,
        m_migration=args.m_migration, seed=args.seed, budget_cap=args.budget, tau_in=args.tau_in, tau_out=args.tau_out,
        founder_path=args.founder, lambda_c_override=args.lambda_c,
        mismatch_mode=args.mismatch_mode, soft_penalty=args.soft_penalty,
        mutate_type_weight=args.mutate_type_weight,
        out_path=Path(args.out), progress_path=Path(args.progress),
        hand_seed=args.hand_seed, n_lineages=args.n_lineages, no_crossover=args.no_crossover,
        crossover_op=args.crossover_op,
        multi_niche_eval=args.multi_niche_eval,
        dual_verification=args.dual_verification, dv_pairs=args.dv_pairs,
        n_mismatch_edges=args.n_mismatch_edges,
        skip_population_rcc=args.skip_population_rcc,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
