"""Measure repair claims across different kinds of type mismatch.

Each test introduces mismatches into a founder-based agent, asks for a repair,
and records the claim and measured result for each affected edge. Results are
saved as JSONL records with the test settings and candidate summary."""
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
from core import mag, saet
from core import repair_episode as RE

FOUNDER_DEFAULT = "./data/founder_genome_v4_typed.json"


def sample_edge_masks(severity: int, n_masks: int, rng: random.Random) -> list[list[tuple]]:
    """Sample n_masks distinct random subsets of size `severity` from CANONICAL_EDGES.

    severity = number of injected edges (3 or 7). For 7 there's only one full mask.
    """
    all_edges = list(RE.CANONICAL_EDGES)
    if severity >= len(all_edges):
        return [list(all_edges)]  # only one possible full mask
    masks = []
    seen = set()
    attempts = 0
    while len(masks) < n_masks and attempts < 200:
        attempts += 1
        m = tuple(sorted(rng.sample(all_edges, severity)))
        if m not in seen:
            seen.add(m)
            masks.append(list(m))
    return masks


def make_candidate(founder: dict, tier: str, mask_edges: list[tuple], rng: random.Random,
                   mu: float = 0.05, n_mut: int = 2) -> dict:
    """Generate a random candidate: mutate founder a few times, then inject tier mismatch.

    Mutation introduces genome variation (so candidates differ within a cell → real
    candidate-level variance for CIs). Injection applies the tier's mismatch on the
    mask edges (emitter side).
    """
    g = copy.deepcopy(founder)
    for _ in range(n_mut):
        g = saet.mutate(g, mu, rng, mutate_type_weight=0.05)
    g = RE.inject_mismatch(g, mask_edges, tier)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["T1_cosmetic", "T2_nearmiss", "T3_genuine"])
    ap.add_argument("--severities", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--masks", type=int, default=5, help="random masks per severity")
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    ap.add_argument("--candidates", type=int, default=10, help="candidates per cell")
    ap.add_argument("--founder", type=str, default=FOUNDER_DEFAULT)
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", type=str, default="./results/pathA_static_spectrum.jsonl")
    ap.add_argument("--tracker", type=str, default="./budget_tracker.json")
    ap.add_argument("--limit", type=int, default=0, help="stop after N candidates (0=all, for smoke)")
    args = ap.parse_args()

    founder = mag.load_mag(args.founder)
    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="pathA_static", max_retries=6)
    initial_spend = client.total_usd

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    cand_global = 0
    n_cells = 0
    t0 = time.time()
    summary_rows = []

    try:
        for tier in args.tiers:
            for severity in args.severities:
                # For severity 7 there's only one full mask regardless of --masks
                mask_rng = random.Random(hash((tier, severity)) & 0xFFFFFFFF)
                masks = sample_edge_masks(severity, args.masks, mask_rng)
                for mask_id, mask_edges in enumerate(masks):
                    for seed in args.seeds:
                        n_cells += 1
                        rng = random.Random(seed * 100003 + mask_id * 991 + hash(tier) % 997 + severity)
                        for k in range(args.candidates):
                            if args.limit and cand_global >= args.limit:
                                raise StopIteration
                            cand_global += 1
                            cid = f"{tier}_sev{severity}_m{mask_id}_s{seed}_c{k}"
                            g = make_candidate(founder, tier, mask_edges, rng)
                            try:
                                ep = RE.measure_repair_episode(
                                    g, mask_edges, client,
                                    purpose=f"pathA_static_{tier}_sev{severity}",
                                    seed=seed * 1000 + k,
                                )
                            except BudgetExceeded as e:
                                print(f"BudgetExceeded at candidate {cand_global}: {e}", flush=True)
                                raise StopIteration
                            # write per-edge records
                            for eid, rec in ep["per_edge"].items():
                                row = {
                                    "candidate_id": cid,
                                    "tier": tier,
                                    "severity": severity,
                                    "mask_id": mask_id,
                                    "mask_edges": [RE.edge_id(e) for e in mask_edges],
                                    "seed": seed,
                                    "cand_k": k,
                                    "edge": eid,
                                    "E_repaired": rec["E_repaired"],
                                    "S_repaired": rec["S_repaired"],
                                    "I_repaired": rec["I_repaired"],
                                    "I_confidence": rec["I_confidence"],
                                    "over_claim_edge": 1 if (rec["I_repaired"] == 1 and rec["S_repaired"] == 0) else 0,
                                    "emit_type": rec["emit_type"],
                                    "consume_type": rec["consume_type"],
                                    "record_type": "edge",
                                }
                                fout.write(json.dumps(row) + "\n")
                            # candidate-level summary
                            srow = {
                                "candidate_id": cid, "tier": tier, "severity": severity,
                                "mask_id": mask_id, "seed": seed, "cand_k": k,
                                "over_claim": ep["over_claim"], "es_agreement": ep["es_agreement"],
                                "n_edges": ep["n_edges"], "record_type": "candidate",
                            }
                            fout.write(json.dumps(srow) + "\n")
                            summary_rows.append(srow)
                            fout.flush()
                            if cand_global % 10 == 0:
                                spent = client.total_usd - initial_spend
                                print(f"  [{cand_global} candidates] cells={n_cells} spent=${spent:.4f}", flush=True)
    except StopIteration:
        pass
    finally:
        fout.close()

    spent = client.total_usd - initial_spend
    elapsed = time.time() - t0
    print(f"\n=== Path A static spectrum DONE ===", flush=True)
    print(f"  candidates={cand_global}, cells={n_cells}, elapsed={elapsed:.0f}s, spent=${spent:.4f}", flush=True)

    # quick per-(tier,severity) OverClaim summary
    from collections import defaultdict
    agg = defaultdict(list)
    for s in summary_rows:
        if s["over_claim"] is not None:
            agg[(s["tier"], s["severity"])].append(s["over_claim"])
    print("\n  OverClaim by (tier, severity):")
    for key in sorted(agg.keys()):
        vals = agg[key]
        mean = sum(vals) / len(vals)
        print(f"    {key[0]:14} sev{key[1]}: mean={mean:.3f} n={len(vals)}", flush=True)


if __name__ == "__main__":
    main()
