"""Track repair claims while a population adapts to a small type mismatch.

The runner introduces mismatches on selected edges, evolves the population,
and samples survivors at checkpoints. For each repair attempt, it records
whether the agent's success claim agrees with the measured outcome."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.llm_client import LLMClient, BudgetExceeded
from core import mag, saet, crossover
from core import repair_episode as RE
from pathA_static_spectrum import sample_edge_masks

# reuse niche evaluators from the main experiment for task-fitness selection
import exp1_cell3_v17 as EXP

FOUNDER_DEFAULT = "./data/founder_genome_v4_typed.json"


def measure_checkpoint(population, mask_edges, client, tier, severity, mask_id, seed, gen,
                       k_measure, rng, fout):
    """Measure OverClaim on a sample of k_measure candidates from the population."""
    idxs = list(range(len(population)))
    rng.shuffle(idxs)
    sample = idxs[:k_measure]
    over_claims = []
    for si in sample:
        g = population[si]["genome"]
        cid = f"{tier}_sev{severity}_m{mask_id}_s{seed}_g{gen}_a{si}"
        try:
            ep = RE.measure_repair_episode(g, mask_edges, client,
                                           purpose=f"pathA_dyn_{tier}_g{gen}",
                                           seed=seed * 1000 + gen * 10 + si)
        except BudgetExceeded:
            raise
        for eid, rec in ep["per_edge"].items():
            row = {
                "candidate_id": cid, "tier": tier, "severity": severity,
                "mask_id": mask_id, "seed": seed, "gen": gen, "agent_idx": si,
                "edge": eid, "E_repaired": rec["E_repaired"], "S_repaired": rec["S_repaired"],
                "I_repaired": rec["I_repaired"], "I_confidence": rec["I_confidence"],
                "over_claim_edge": 1 if (rec["I_repaired"] == 1 and rec["S_repaired"] == 0) else 0,
                "emit_type": rec["emit_type"], "consume_type": rec["consume_type"],
                "record_type": "edge",
            }
            fout.write(json.dumps(row) + "\n")
        srow = {"candidate_id": cid, "tier": tier, "severity": severity, "mask_id": mask_id,
                "seed": seed, "gen": gen, "agent_idx": si, "over_claim": ep["over_claim"],
                "es_agreement": ep["es_agreement"], "n_edges": ep["n_edges"],
                "record_type": "candidate"}
        fout.write(json.dumps(srow) + "\n")
        if ep["over_claim"] is not None:
            over_claims.append(ep["over_claim"])
    fout.flush()
    mean_oc = sum(over_claims) / len(over_claims) if over_claims else None
    return mean_oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="T2_nearmiss")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--mask-id", dest="mask_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=5)
    ap.add_argument("--k-measure", dest="k_measure", type=int, default=6)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--pop-eval-tasks", dest="pop_eval_tasks", type=int, default=2)
    ap.add_argument("--niches", nargs="+",
                    default=["planbench_blocksworld", "locomo", "hotpotqa"])
    ap.add_argument("--founder", default=FOUNDER_DEFAULT)
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", default="./results/pathA_dynamic_T2.jsonl")
    ap.add_argument("--tracker", default="./budget_tracker.json")
    args = ap.parse_args()

    founder = mag.load_mag(args.founder)
    cost_model = founder.get("cost_model", {})
    nm_block = founder.get("niche_multiplier") or {}
    niche_multiplier_table = nm_block.get("multiplier_table")
    # set soft mode globals in EXP so its niche evaluators apply soft penalty
    EXP._MISMATCH_MODE = "soft"
    EXP._SOFT_PENALTY = 0.3
    EXP._MUTATE_TYPE_WEIGHT = 0.05

    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="pathA_dynamic", max_retries=6)
    initial_spend = client.total_usd

    rng = random.Random(args.seed)
    # pick the mask (same sampler as static so mask_id aligns)
    mask_rng = random.Random(hash((args.tier, args.severity)) & 0xFFFFFFFF)
    masks = sample_edge_masks(args.severity, args.mask_id + 1 if args.severity < 7 else 1, mask_rng)
    mask_edges = masks[min(args.mask_id, len(masks) - 1)]

    # init population: founder mutated + T2-injected on mask edges
    population = []
    for i in range(args.N):
        g = copy.deepcopy(founder)
        for _ in range(2):
            g = saet.mutate(g, args.mu, rng, mutate_type_weight=0.05)
        g = RE.inject_mismatch(g, mask_edges, args.tier)
        g["_lineage_id"] = f"L{i}"
        population.append({"id": i, "genome": g, "birth_gen": 0})
    next_id = args.N

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    t0 = time.time()
    traj = []
    print(f"=== Path A T2 dynamic: tier={args.tier} sev={args.severity} mask={[RE.edge_id(e) for e in mask_edges]} seed={args.seed} ===", flush=True)

    try:
        # gen-0 measurement (pre-adaptation)
        oc0 = measure_checkpoint(population, mask_edges, client, args.tier, args.severity,
                                 args.mask_id, args.seed, 0, args.k_measure, rng, fout)
        traj.append((0, oc0))
        print(f"  gen0 OverClaim={oc0}", flush=True)

        for t in range(args.T):
            # task eval for selection (soft mode → mismatched agents still run, penalized)
            niche_assign = EXP.assign_niches_task_focus(population, args.niches, 0.1, rng) \
                if niche_multiplier_table is not None else \
                [args.niches[i % len(args.niches)] for i in range(len(population))]
            fits = []
            for i, ag in enumerate(population):
                niche = niche_assign[i]
                try:
                    res = EXP.evaluate_agent_on_niche(
                        ag["genome"], niche, client, args.pop_eval_tasks, cost_model,
                        purpose=f"pathA_dyn_t{t}_eval", seed=args.seed + t * 100 + i,
                        niche_multiplier_table=niche_multiplier_table)
                    fits.append(res["F"])
                except BudgetExceeded:
                    raise
            # offspring: crossover + mutate, re-inject T2 (keep the mismatch substrate)
            n_off = args.N // 2
            pairs = saet.select_parent_pairs([a["genome"] for a in population], fits,
                                             compat_predictor=None, beta=args.beta,
                                             n_pairs=n_off, rng=rng)
            offspring = []
            for (i, j) in pairs:
                child = crossover.typed_subgraph_crossover(population[i]["genome"],
                                                           population[j]["genome"], rng=rng)
                child = saet.mutate(child, args.mu, rng, mutate_type_weight=0.05)
                # persist the T2 injection so the substrate remains (else evolution
                # trivially "repairs" by reverting to pristine and there's nothing to over-claim)
                child = RE.inject_mismatch(child, mask_edges, args.tier)
                li = population[i]["genome"].get("_lineage_id", str(i))
                lj = population[j]["genome"].get("_lineage_id", str(j))
                child["_lineage_id"] = li if li == lj else f"{li}+{lj}"
                offspring.append({"id": next_id, "genome": child, "birth_gen": t + 1})
                next_id += 1
            # offspring fitness
            off_fits = []
            for k, ag in enumerate(offspring):
                niche = args.niches[k % len(args.niches)]
                try:
                    res = EXP.evaluate_agent_on_niche(
                        ag["genome"], niche, client, args.pop_eval_tasks, cost_model,
                        purpose=f"pathA_dyn_t{t}_off", seed=args.seed + t * 100 + 500 + k,
                        niche_multiplier_table=niche_multiplier_table)
                    off_fits.append(res["F"])
                except BudgetExceeded:
                    raise
            # survivor selection: global top-N
            combined = list(zip(population, fits)) + list(zip(offspring, off_fits))
            combined.sort(key=lambda x: -x[1])
            population = [c[0] for c in combined[:args.N]]

            if (t + 1) % args.eval_every == 0:
                oc = measure_checkpoint(population, mask_edges, client, args.tier, args.severity,
                                        args.mask_id, args.seed, t + 1, args.k_measure, rng, fout)
                traj.append((t + 1, oc))
                print(f"  gen{t+1} OverClaim={oc}", flush=True)
    except BudgetExceeded as e:
        print(f"BudgetExceeded: {e}", flush=True)
    finally:
        fout.close()

    spent = client.total_usd - initial_spend
    print(f"\n=== DONE: elapsed {time.time()-t0:.0f}s spent ${spent:.4f} ===", flush=True)
    print(f"  OverClaim trajectory: {traj}", flush=True)
    static_oc = traj[0][1] if traj else None
    final_oc = traj[-1][1] if traj else None
    if static_oc is not None and final_oc is not None:
        print(f"  static(gen0)={static_oc:.3f} final={final_oc:.3f} dynamic_delta={final_oc-static_oc:+.3f}", flush=True)


if __name__ == "__main__":
    main()
