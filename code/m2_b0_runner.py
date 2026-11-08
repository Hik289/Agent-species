"""Measure the starting agent before population evolution.

The stages evaluate planning on Blocksworld and Logistics, test crosses of
founder copies, and evaluate memory questions on LoCoMo. Command-line options
set sample sizes, repetitions, and the spending limit."""
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from core.llm_client import LLMClient, BudgetExceeded  # noqa: E402
from core.agent_runner import run_founder_on_instance  # noqa: E402
from core import mag  # noqa: E402
from niches.planbench_eval import (  # noqa: E402
    load_prompt_json, load_domain_config, get_problem_actions, get_problem_pddl,
    DOMAIN_PDDL, VAL_VALIDATE, _DOMAIN_TO_FN,
)
from niches.locomo_eval import evaluate_one as locomo_eval_one, sample_locomo_subset  # noqa: E402

FOUNDER_PATH = Path("./data/founder_genome_v0.json")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------

def _planbench_evaluate(genome, dom, instance_ids, llm_client, purpose, progress_log):
    instances = load_prompt_json(dom, "task_1_plan_generation")
    by_id = {i["instance_id"]: i for i in instances}
    data = load_domain_config(dom)
    trials = []
    for iid in instance_ids:
        if iid not in by_id:
            continue
        action_set = get_problem_actions(dom, iid)
        problem_pddl = get_problem_pddl(dom, iid)
        t0 = time.time()
        trial = run_founder_on_instance(
            genome=genome,
            domain=dom,
            instance_id=iid,
            instance_payload=by_id[iid],
            llm_client=llm_client,
            domain_data=data,
            action_set=action_set,
            text_to_plan_fn=_DOMAIN_TO_FN[dom],
            domain_pddl=DOMAIN_PDDL[dom],
            problem_pddl=problem_pddl,
            val_bin=VAL_VALIDATE,
            purpose=purpose,
        )
        dt = time.time() - t0
        trials.append(trial)
        if progress_log:
            with progress_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "purpose": purpose, "domain": dom, "instance_id": iid,
                    "success": trial.success, "cost": trial.cost_usd,
                    "n_samples": trial.n_samples, "n_replans": trial.n_replans,
                    "wallclock_sec": dt, "strategy": trial.chosen_strategy,
                    "prompt_tokens": trial.prompt_tokens, "completion_tokens": trial.completion_tokens,
                    "error": trial.error,
                }) + "\n")
    return trials


def _checkpoint_save(out_path: Path, state: dict[str, Any]):
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(out_path)


# ---------- Stage 1: founder PlanBench BW+LG 50×3 ----------

def stage_1_founder_fitness(
    genome, llm_client, n_per_domain=25, reps=3, instance_offset=2,
    progress_log=None, partial_out=None,
) -> dict[str, Any]:
    """50 task (25 BW + 25 LG) × reps. Save partial after every rep."""
    bw_ids = list(range(instance_offset, instance_offset + n_per_domain))
    lg_ids = list(range(instance_offset, instance_offset + n_per_domain))

    state: dict[str, Any] = {"per_rep": []}
    for rep in range(reps):
        rep_data: dict[str, Any] = {"rep": rep, "domains": {}}
        for dom, iids in [("blocksworld", bw_ids), ("logistics", lg_ids)]:
            trials = _planbench_evaluate(
                genome, dom, iids, llm_client,
                purpose=f"m2b0_stage1_{dom}_rep{rep}",
                progress_log=progress_log,
            )
            n_ok = sum(1 for t in trials if t.success)
            rep_data["domains"][dom] = {
                "n": len(trials), "n_success": n_ok,
                "rate": n_ok / max(len(trials), 1),
                "cost_usd": sum(t.cost_usd for t in trials),
                "tokens": {
                    "prompt": sum(t.prompt_tokens for t in trials),
                    "completion": sum(t.completion_tokens for t in trials),
                },
                "trials": [t.__dict__ for t in trials],
            }
        state["per_rep"].append(rep_data)
        if partial_out:
            _checkpoint_save(partial_out, state)
    return state


# ---------- Stage 2: self-cross τ_v calibration ----------

def stage_2_self_cross(
    genome, llm_client, n_clones=16, R=8, eval_subset_size=10, seed=42,
    progress_log=None, partial_out=None,
) -> dict[str, Any]:
    """16 founder clones × R=8 hybrids per pair (we treat hybrids as founder
    since v0 mutations are trivial; so M independent founder evaluations each
    on a 10-task subset).

    M = min(n_clones * R / 2, 64). Defaults: 16*8/2 = 64.
    """
    rng = random.Random(seed)
    M_total = (n_clones * R) // 2
    M_eff = min(M_total, 64)
    # Use BW instance 27..150 as held-out (stage 1 used 2..26)
    held_out = list(range(27, 151))
    rng.shuffle(held_out)

    viab_scores: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "n_clones": n_clones, "R": R, "M_total": M_total, "M_effective": M_eff,
        "eval_subset_size": eval_subset_size, "held_out_range": [27, 150],
        "viability_scores": viab_scores,
    }
    for k in range(M_eff):
        subset = [held_out[(k * eval_subset_size + j) % len(held_out)]
                  for j in range(eval_subset_size)]
        trials = _planbench_evaluate(
            genome, "blocksworld", subset, llm_client,
            purpose=f"m2b0_stage2_k{k}", progress_log=progress_log,
        )
        n_ok = sum(1 for t in trials if t.success)
        V = n_ok / max(len(trials), 1)
        viab_scores.append({
            "sample": k, "subset": subset, "n": len(trials), "n_ok": n_ok, "V": V,
            "cost_usd": sum(t.cost_usd for t in trials),
        })
        if partial_out:
            _checkpoint_save(partial_out, state)
    return state


def calibrate_tau_v(viab_scores: list[float], target_pass_frac: float = 0.70):
    """Largest τ_v such that fraction(V ≥ τ_v) ≥ target_pass_frac."""
    if not viab_scores:
        return {"tau_v": None, "error": "no_samples"}
    n = len(viab_scores)
    # Sweep candidate thresholds at 0.05 steps
    best_tau = 0.0
    best_frac = 0.0
    for k in range(0, 21):
        tau = k * 0.05
        frac = sum(1 for v in viab_scores if v >= tau) / n
        if frac >= target_pass_frac and tau > best_tau:
            best_tau = tau
            best_frac = frac
    return {"tau_v": best_tau, "achieved_pass_frac": best_frac,
            "target_pass_frac": target_pass_frac, "n_samples": n,
            "viab_quartiles": {
                "q25": statistics.quantiles(viab_scores, n=4)[0] if n >= 4 else None,
                "q50": statistics.median(viab_scores),
                "q75": statistics.quantiles(viab_scores, n=4)[2] if n >= 4 else None,
                "mean": statistics.mean(viab_scores),
                "std": statistics.stdev(viab_scores) if n > 1 else 0.0,
            }}


# ---------- Stage 3: LoCoMo sanity ----------

def stage_3_locomo(
    genome, llm_client, n_tasks=50, progress_log=None, partial_out=None,
) -> dict[str, Any]:
    subset = sample_locomo_subset(n=n_tasks, seed=42)
    trials = []
    state: dict[str, Any] = {"n": n_tasks, "trials": []}
    for i, item in enumerate(subset):
        t0 = time.time()
        trial = locomo_eval_one(
            conv_payload=item["conv_payload"],
            question=item["question"],
            gold_answer=item["gold_answer"],
            category=item["category"],
            conversation_id=item["conversation_id"],
            llm_client=llm_client,
            genome=genome,  # use verifier.samples
            purpose="m2b0_stage3_locomo",
        )
        dt = time.time() - t0
        trials.append(trial.__dict__)
        state["trials"] = trials
        if progress_log:
            with progress_log.open("a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "purpose": "m2b0_stage3_locomo",
                    "domain": "locomo", "instance_id": i,
                    "success": trial.success, "category": trial.category,
                    "cost": trial.cost_usd, "n_samples": trial.n_samples,
                    "prompt_tokens": trial.prompt_tokens,
                    "completion_tokens": trial.completion_tokens,
                    "wallclock_sec": dt, "error": trial.error,
                }) + "\n")
        if partial_out:
            _checkpoint_save(partial_out, state)

    # Aggregate
    n_ok = sum(1 for t in trials if t["success"])
    by_cat: dict[int, dict] = {}
    for t in trials:
        c = t["category"]
        if c not in by_cat:
            by_cat[c] = {"n": 0, "n_ok": 0}
        by_cat[c]["n"] += 1
        by_cat[c]["n_ok"] += int(t["success"])
    state["aggregate"] = {
        "n": len(trials), "n_success": n_ok,
        "rate": n_ok / max(len(trials), 1),
        "cost_usd": sum(t["cost_usd"] for t in trials),
        "tokens": {
            "prompt": sum(t["prompt_tokens"] for t in trials),
            "completion": sum(t["completion_tokens"] for t in trials),
        },
        "by_category": by_cat,
    }
    return state


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--stage1-n-per-domain", type=int, default=25)
    ap.add_argument("--stage1-reps", type=int, default=3)
    ap.add_argument("--stage2-M", type=int, default=64)
    ap.add_argument("--stage3-n", type=int, default=50)
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "m2_b0_calibration.json"))
    ap.add_argument("--progress", type=str, default=str(RESULTS_DIR / "m2_b0_progress.jsonl"))
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--skip-stage2", action="store_true")
    ap.add_argument("--skip-stage3", action="store_true")
    args = ap.parse_args()

    print(f"=== M2 B-0 runner ===", flush=True)
    genome = mag.load_mag(FOUNDER_PATH)
    ok, viol = mag.full_validate(genome)
    assert ok, f"founder invalid: {viol}"

    client = LLMClient(
        budget_usd_hard_cap=args.budget,
        tracker_path="./budget_tracker.json",
        purpose_tag="m2_b0",
    )
    initial = client.total_usd
    print(f"budget cap ${args.budget:.2f}; existing spend ${initial:.4f}", flush=True)

    progress = Path(args.progress)
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("")

    out = Path(args.out)
    result: dict[str, Any] = {"phase": "starting", "started_at": time.time()}
    _checkpoint_save(out, result)

    t0 = time.time()

    # ---- Stage 1 ----
    if not args.skip_stage1:
        print(f"\n[stage 1] PlanBench BW+LG {args.stage1_n_per_domain} task × {args.stage1_reps} reps", flush=True)
        try:
            s1 = stage_1_founder_fitness(
                genome, client,
                n_per_domain=args.stage1_n_per_domain,
                reps=args.stage1_reps,
                progress_log=progress,
                partial_out=RESULTS_DIR / "m2_b0_stage1_partial.json",
            )
            bw_rates = [r["domains"]["blocksworld"]["rate"] for r in s1["per_rep"]]
            lg_rates = [r["domains"]["logistics"]["rate"] for r in s1["per_rep"]]
            agg_rate = (sum(bw_rates) + sum(lg_rates)) / (len(bw_rates) + len(lg_rates))
            s1["aggregate"] = {
                "bw_rates_per_rep": bw_rates, "lg_rates_per_rep": lg_rates,
                "bw_mean": statistics.mean(bw_rates) if bw_rates else 0,
                "lg_mean": statistics.mean(lg_rates) if lg_rates else 0,
                "bw_std": statistics.stdev(bw_rates) if len(bw_rates) > 1 else 0,
                "lg_std": statistics.stdev(lg_rates) if len(lg_rates) > 1 else 0,
                "aggregate_rate": agg_rate,
            }
            result["stage_1"] = s1
            print(f"  BW rates: {bw_rates}, mean={s1['aggregate']['bw_mean']:.3f} ± {s1['aggregate']['bw_std']:.3f}", flush=True)
            print(f"  LG rates: {lg_rates}, mean={s1['aggregate']['lg_mean']:.3f} ± {s1['aggregate']['lg_std']:.3f}", flush=True)
            print(f"  aggregate: {agg_rate:.3f}, spent so far ${client.total_usd-initial:.4f}", flush=True)
        except BudgetExceeded as e:
            result["stage_1_error"] = f"BudgetExceeded: {e}"
            print(f"⚠️ {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            result["stage_1_error"] = f"{type(e).__name__}: {e}"
            print(f"⚠️ stage 1 failed: {e}", flush=True)
        _checkpoint_save(out, result)

    # ---- Stage 2 ----
    if not args.skip_stage2 and (client.total_usd - initial) < args.budget * 0.9:
        print(f"\n[stage 2] self-cross calibration M={args.stage2_M}", flush=True)
        try:
            s2 = stage_2_self_cross(
                genome, client,
                n_clones=16, R=8, eval_subset_size=10,
                progress_log=progress,
                partial_out=RESULTS_DIR / "m2_b0_stage2_partial.json",
            )
            v_list = [s["V"] for s in s2["viability_scores"]]
            s2["tau_v_calibration"] = calibrate_tau_v(v_list, target_pass_frac=0.70)
            result["stage_2"] = s2
            print(f"  M_eff={s2['M_effective']}, V mean={s2['tau_v_calibration']['viab_quartiles']['mean']:.3f}, "
                  f"τ_v={s2['tau_v_calibration']['tau_v']:.2f} (achieved {s2['tau_v_calibration']['achieved_pass_frac']:.2%})", flush=True)
        except BudgetExceeded as e:
            result["stage_2_error"] = f"BudgetExceeded: {e}"
        except Exception as e:  # noqa: BLE001
            result["stage_2_error"] = f"{type(e).__name__}: {e}"
            print(f"⚠️ stage 2 failed: {e}", flush=True)
        _checkpoint_save(out, result)

    # ---- Stage 3 ----
    if not args.skip_stage3 and (client.total_usd - initial) < args.budget * 0.95:
        print(f"\n[stage 3] LoCoMo sanity (50 conv-QA × 1 rep, with verifier.samples=3)", flush=True)
        try:
            s3 = stage_3_locomo(
                genome, client,
                n_tasks=args.stage3_n, progress_log=progress,
                partial_out=RESULTS_DIR / "m2_b0_stage3_partial.json",
            )
            result["stage_3"] = s3
            agg = s3["aggregate"]
            print(f"  LoCoMo: {agg['n_success']}/{agg['n']} = {agg['rate']:.3f}, spent ${client.total_usd-initial:.4f}", flush=True)
            for cat, d in agg["by_category"].items():
                cat_name = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain", 5: "adversarial"}.get(cat, str(cat))
                print(f"    cat={cat} ({cat_name}): {d['n_ok']}/{d['n']}", flush=True)
        except BudgetExceeded as e:
            result["stage_3_error"] = f"BudgetExceeded: {e}"
        except Exception as e:  # noqa: BLE001
            result["stage_3_error"] = f"{type(e).__name__}: {e}"
            print(f"⚠️ stage 3 failed: {e}", flush=True)
        _checkpoint_save(out, result)

    # ---- Final summary ----
    result["finished_at"] = time.time()
    result["elapsed_sec"] = time.time() - t0
    result["spent_delta_usd"] = client.total_usd - initial
    result["budget_cap_usd"] = args.budget
    result["machine"] = "SERVER"
    result["founder_path"] = str(FOUNDER_PATH)
    _checkpoint_save(out, result)

    print(f"\n=== DONE: spent ${result['spent_delta_usd']:.4f}, elapsed {result['elapsed_sec']:.0f}s ===", flush=True)
    print(f"output: {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
