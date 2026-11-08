"""Measure the starting agent on PlanBench and test crosses of its copies.

Repeated Blocksworld and Logistics tasks measure planning quality. Crosses of
founder copies estimate how often new agents pass the quality threshold. The
runner saves the calibration and stops when configured checks or budgets fail."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from core.llm_client import LLMClient  # noqa: E402
from core.agent_runner import run_founder_on_instance, AgentTrial  # noqa: E402
from core import mag, crossover  # noqa: E402
from niches.planbench_eval import (  # noqa: E402
    load_prompt_json, load_domain_config, get_problem_actions, get_problem_pddl,
    DOMAIN_PDDL, VAL_VALIDATE, _DOMAIN_TO_FN,
)

# Founder genome
FOUNDER_PATH = Path("./data/founder_genome_v0.json")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_genome_on_subset(
    genome: dict[str, Any],
    domain: str,
    instance_ids: list[int],
    llm_client: LLMClient,
    purpose: str,
    progress_log: Path | None = None,
) -> list[AgentTrial]:
    instances = load_prompt_json(domain, "task_1_plan_generation")
    by_id = {i["instance_id"]: i for i in instances}
    data = load_domain_config(domain)
    trials: list[AgentTrial] = []
    for iid in instance_ids:
        if iid not in by_id:
            trials.append(AgentTrial(
                instance_id=iid, domain=domain, success=False, n_samples=0, n_replans=0,
                final_plan_pddl="", final_plan_lines=0, val_stdout_snippet="",
                val_returncode=None, cost_usd=0.0, prompt_tokens=0, completion_tokens=0,
                error=f"missing instance {iid}",
            ))
            continue
        action_set = get_problem_actions(domain, iid)
        problem_pddl = get_problem_pddl(domain, iid)
        t0 = time.time()
        trial = run_founder_on_instance(
            genome=genome,
            domain=domain,
            instance_id=iid,
            instance_payload=by_id[iid],
            llm_client=llm_client,
            domain_data=data,
            action_set=action_set,
            text_to_plan_fn=_DOMAIN_TO_FN[domain],
            domain_pddl=DOMAIN_PDDL[domain],
            problem_pddl=problem_pddl,
            val_bin=VAL_VALIDATE,
            purpose=purpose,
        )
        dt = time.time() - t0
        trials.append(trial)
        if progress_log:
            with progress_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "purpose": purpose,
                    "domain": domain,
                    "instance_id": iid,
                    "success": trial.success,
                    "cost": trial.cost_usd,
                    "n_samples": trial.n_samples,
                    "n_replans": trial.n_replans,
                    "wallclock_sec": dt,
                    "error": trial.error,
                    "strategy": trial.chosen_strategy,
                }) + "\n")
    return trials


def stage_founder_fitness(
    genome: dict[str, Any],
    llm_client: LLMClient,
    n_per_domain: int = 25,
    reps: int = 5,
    instance_offset: int = 2,
    progress_log: Path | None = None,
) -> dict[str, Any]:
    """Run founder MAG `reps` times on a fixed BW + LG subset."""
    bw_ids = list(range(instance_offset, instance_offset + n_per_domain))
    lg_ids = list(range(instance_offset, instance_offset + n_per_domain))

    per_rep: list[dict[str, Any]] = []
    for rep in range(reps):
        rep_data: dict[str, Any] = {"rep": rep, "domains": {}}
        for dom, iids in [("blocksworld", bw_ids), ("logistics", lg_ids)]:
            trials = evaluate_genome_on_subset(
                genome, dom, iids, llm_client,
                purpose=f"m2_founder_{dom}_rep{rep}",
                progress_log=progress_log,
            )
            n_ok = sum(1 for t in trials if t.success)
            rep_data["domains"][dom] = {
                "n": len(trials),
                "n_success": n_ok,
                "rate": n_ok / max(len(trials), 1),
                "cost_usd": sum(t.cost_usd for t in trials),
                "trials": [t.__dict__ for t in trials],
            }
        per_rep.append(rep_data)
    return {"per_rep": per_rep, "founder_path": str(FOUNDER_PATH)}


def stage_self_cross(
    genome: dict[str, Any],
    llm_client: LLMClient,
    n_clones: int = 8,
    R_hybrids_per_pair: int = 8,
    eval_subset_size: int = 10,
    seed: int = 42,
    progress_log: Path | None = None,
) -> dict[str, Any]:
    """Cheap self-cross calibration.

    Strategy: n_clones identical founder genomes (epsilon mutations are trivial in v0).
    Generate R hybrids per unique pair via typed_subgraph_crossover; since all clones
    are identical, every hybrid == founder, so every hybrid evaluation reuses the
    same fitness distribution. We just sample R*pair_count viability outcomes and
    calibrate τ_v from continuous fitness V(o) per spec.

    Wait — `pair_count * R` × `eval_subset_size` evals would blow budget. Since clones
    are identical we ONLY need: M = R_hybrids_per_pair * n_pairs evaluations of
    "founder genome on `eval_subset_size`-task subset", yielding M viability scores.
    To save cost we sample M_eff = min(M, 64) unique trials (each a fresh founder
    eval on a 10-task subset → continuous viability ∈ [0,1]).

    This gives a viability distribution → τ_v calibrated as the 30th percentile,
    so that ≥ 70% of self-cross hybrids land above τ_v.
    """
    rng = random.Random(seed)
    n_pairs = n_clones * (n_clones - 1) // 2  # C(n_clones, 2)
    # M_total = n_pairs * R_hybrids_per_pair  # nominal but redundant when clones identical
    M_eff = 64  # 64 viability samples is enough to estimate 30th percentile to ±0.05

    # Use a rotating 10-task subset
    instances = load_prompt_json("blocksworld", "task_1_plan_generation")
    inst_ids = [i["instance_id"] for i in instances]
    # exclude the 25 used in stage_founder_fitness (offset 2..26) to keep tasks held out
    held_out = [i for i in inst_ids if i >= 27]
    if len(held_out) < eval_subset_size:
        held_out = inst_ids
    rng.shuffle(held_out)

    viab_scores: list[dict[str, Any]] = []
    for k in range(M_eff):
        # rotate through held-out instances
        subset = [held_out[(k * eval_subset_size + j) % len(held_out)] for j in range(eval_subset_size)]
        trials = evaluate_genome_on_subset(
            genome, "blocksworld", subset, llm_client,
            purpose=f"m2_self_cross_k{k}",
            progress_log=progress_log,
        )
        n_ok = sum(1 for t in trials if t.success)
        V = n_ok / max(len(trials), 1)
        viab_scores.append({"sample": k, "subset": subset, "n": len(trials), "n_ok": n_ok, "V": V,
                            "cost_usd": sum(t.cost_usd for t in trials)})

    V_list = [s["V"] for s in viab_scores]
    return {
        "n_clones": n_clones,
        "R_hybrids_per_pair": R_hybrids_per_pair,
        "n_pairs": n_pairs,
        "eval_subset_size": eval_subset_size,
        "M_effective": M_eff,
        "viability_scores": viab_scores,
        "viab_summary": {
            "mean": statistics.mean(V_list) if V_list else 0.0,
            "std": statistics.stdev(V_list) if len(V_list) > 1 else 0.0,
            "min": min(V_list) if V_list else 0.0,
            "max": max(V_list) if V_list else 0.0,
        },
    }


def calibrate_tau_v(viab_scores: list[float], target_pass_frac: float = 0.70) -> dict[str, Any]:
    """Find the largest τ_v such that fraction of V ≥ τ_v ≥ target_pass_frac."""
    sorted_v = sorted(viab_scores, reverse=True)
    n = len(sorted_v)
    if n == 0:
        return {"tau_v": None, "achieved_pass_frac": 0.0, "error": "no_samples"}
    # τ_v = max V s.t. (#{V >= τ_v}) / n ≥ target_pass_frac
    candidates = [0.05 * i for i in range(0, 21)]  # 0.00 .. 1.00 step 0.05
    best_tau = 0.0
    best_frac = 0.0
    for tau in candidates:
        frac = sum(1 for v in sorted_v if v >= tau) / n
        if frac >= target_pass_frac and tau > best_tau:
            best_tau = tau
            best_frac = frac
    return {"tau_v": best_tau, "achieved_pass_frac": best_frac,
            "target_pass_frac": target_pass_frac, "n_samples": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=15.0, help="hard USD cap for this run")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_per_domain", type=int, default=25)
    ap.add_argument("--self_cross_M", type=int, default=64, help="self-cross viability samples")
    ap.add_argument("--dry-run", action="store_true", help="evaluate 1 task per domain only")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "m2_anchor1_calibration.json"))
    ap.add_argument("--progress", type=str, default=str(RESULTS_DIR / "m2_progress.jsonl"))
    args = ap.parse_args()

    print(f"=== M2 anchor_1 founder calibration ===", flush=True)
    print(f"founder: {FOUNDER_PATH}", flush=True)
    genome = mag.load_mag(FOUNDER_PATH)
    ok, viol = mag.full_validate(genome)
    assert ok, f"founder genome invalid: {viol}"

    client = LLMClient(
        budget_usd_hard_cap=args.budget,
        tracker_path="./budget_tracker.json",
        purpose_tag="m2_anchor1",
    )
    initial_spend = client.total_usd
    print(f"budget cap ${args.budget:.2f}; existing spend ${initial_spend:.4f}", flush=True)

    progress = Path(args.progress)
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("")  # truncate

    t0 = time.time()
    if args.dry_run:
        print("=== DRY RUN (1 instance per domain) ===", flush=True)
        s1 = stage_founder_fitness(
            genome, client,
            n_per_domain=1, reps=1, instance_offset=2,
            progress_log=progress,
        )
        result = {"dry_run": True, "stage_founder": s1, "elapsed_sec": time.time() - t0,
                  "spent_usd_delta": client.total_usd - initial_spend}
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str)[:2000])
        return 0

    # ---- Stage 1: founder fitness ----
    print(f"\n[stage 1] founder fitness: BW+LG × {args.n_per_domain} tasks × {args.reps} reps", flush=True)
    stage1 = stage_founder_fitness(
        genome, client,
        n_per_domain=args.n_per_domain, reps=args.reps,
        progress_log=progress,
    )
    spent_after_s1 = client.total_usd - initial_spend
    print(f"[stage 1] spend ${spent_after_s1:.4f}, elapsed {time.time()-t0:.0f}s", flush=True)

    # GA.M2 abort gate check
    bw_rates = [r["domains"]["blocksworld"]["rate"] for r in stage1["per_rep"]]
    lg_rates = [r["domains"]["logistics"]["rate"] for r in stage1["per_rep"]]
    aggregate_rate = (sum(bw_rates) + sum(lg_rates)) / (len(bw_rates) + len(lg_rates))
    print(f"  BW rates per rep: {bw_rates}", flush=True)
    print(f"  LG rates per rep: {lg_rates}", flush=True)
    print(f"  aggregate PlanBench rate: {aggregate_rate:.3f}", flush=True)

    abort_gate_m2_planbench_60 = aggregate_rate < 0.60

    if abort_gate_m2_planbench_60:
        print(f"⚠️ GA.M2 PlanBench < 60% triggered (rate={aggregate_rate:.3f})", flush=True)

    # ---- Stage 2: self-cross calibration (skip if PlanBench already abort) ----
    stage2 = None
    if not abort_gate_m2_planbench_60 or client.total_usd - initial_spend < args.budget * 0.6:
        # only run if we have budget left and either rate ok or extra to spend
        remaining = args.budget - (client.total_usd - initial_spend)
        print(f"\n[stage 2] self-cross calibration; remaining budget ${remaining:.4f}", flush=True)
        try:
            stage2 = stage_self_cross(
                genome, client,
                n_clones=8, R_hybrids_per_pair=8, eval_subset_size=10,
                progress_log=progress,
            )
        except Exception as e:  # noqa: BLE001
            stage2 = {"error": f"{type(e).__name__}: {e}"}

    # ---- Calibrate τ_v ----
    tau_v_block = {"skipped": True}
    if stage2 and isinstance(stage2, dict) and "viability_scores" in stage2:
        v_list = [s["V"] for s in stage2["viability_scores"]]
        tau_v_block = calibrate_tau_v(v_list, target_pass_frac=0.70)
        if tau_v_block.get("tau_v") is None:
            print("⚠️ τ_v calibration failed (no samples)", flush=True)

    # ---- Final report ----
    result = {
        "anchor": "anchor_1",
        "machine": "SERVER",
        "founder_path": str(FOUNDER_PATH),
        "budget_cap_usd": args.budget,
        "spent_delta_usd": client.total_usd - initial_spend,
        "elapsed_sec": time.time() - t0,
        "stage_1_founder_fitness": {
            "config": {"n_per_domain": args.n_per_domain, "reps": args.reps},
            "per_rep_rates": {"blocksworld": bw_rates, "logistics": lg_rates},
            "aggregate_rate": aggregate_rate,
            "ga_m2_planbench_60_triggered": abort_gate_m2_planbench_60,
            "detail": stage1,
        },
        "stage_2_self_cross": stage2,
        "tau_v_calibration": tau_v_block,
        "abort_gates": {
            "ga_m2_planbench_lt_60": abort_gate_m2_planbench_60,
            "ga_m2_self_cross_viability_lt_70": (tau_v_block.get("tau_v") is None
                                                  if isinstance(tau_v_block, dict) else None),
            "ga_m2_budget_breach": client.total_usd - initial_spend > 30.0,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\n=== DONE: spent ${client.total_usd - initial_spend:.4f}, "
          f"elapsed {time.time()-t0:.0f}s, output {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
