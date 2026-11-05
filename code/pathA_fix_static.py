"""Path A-fix static applied-repair spectrum (orchestrator 2026-07-24).

Headline confirmatory run using the 3-phase applied-repair episode (repair_episode_v2).
OverClaim = Pr(I=1, S=0) on the REPAIRED state, S deterministic.

Matrix: 3 tier × 2 severity × M masks × S seeds × {blinded, labeled} × K candidates.
Each candidate = one repair episode over the mask's injected edges (the "candidate"
variation comes from LLM sampling + seed; genome is founder-based but the repair is what
varies). Records per candidate × edge: op, transform, E, S, I, conf, over_claim.

Cluster-robust note: resampling unit for CI is seed×severity (n≤10 per tier) — done in
analysis, not here. Here we just emit all records with seed/mask/severity tags.

Output JSONL. Run:
  python pathA_fix_static.py --tiers T1_cosmetic T2_nearmiss T3_genuine \
    --severities 3 7 --masks 3 --seeds 101 102 103 104 105 --candidates 6 \
    --modes blinded labeled --out .../pathA_fix_static.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.llm_client import LLMClient, BudgetExceeded
from core import repair_episode_v2 as RE2


def sample_edge_masks(severity, n_masks, rng):
    all_edges = list(RE2.CANONICAL_EDGES)
    if severity >= len(all_edges):
        return [list(all_edges)]
    masks, seen, attempts = [], set(), 0
    while len(masks) < n_masks and attempts < 200:
        attempts += 1
        m = tuple(sorted(rng.sample(all_edges, severity)))
        if m not in seen:
            seen.add(m); masks.append(list(m))
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["T1_cosmetic", "T2_nearmiss", "T3_genuine"])
    ap.add_argument("--severities", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--masks", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--modes", nargs="+", default=["blinded", "labeled"])
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", default="./results/pathA_fix_static.jsonl")
    ap.add_argument("--tracker", default="./budget_tracker.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="pathAfix_static", max_retries=6)
    initial = client.total_usd
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    n = 0; t0 = time.time()
    summary = []
    try:
        for tier in args.tiers:
            for severity in args.severities:
                mask_rng = random.Random(hash((tier, severity)) & 0xFFFFFFFF)
                masks = sample_edge_masks(severity, args.masks, mask_rng)
                for mask_id, mask_edges in enumerate(masks):
                    for mode in args.modes:
                        blinded = (mode == "blinded")
                        for seed in args.seeds:
                            for k in range(args.candidates):
                                if args.limit and n >= args.limit:
                                    raise StopIteration
                                n += 1
                                cid = f"{tier}_sev{severity}_m{mask_id}_{mode}_s{seed}_c{k}"
                                try:
                                    ep = RE2.measure_repair_episode_v2(
                                        mask_edges, client,
                                        purpose=f"pathAfix_{tier}_sev{severity}_{mode}",
                                        tier=tier, blinded=blinded, seed=seed * 1000 + k)
                                except BudgetExceeded as e:
                                    print(f"BudgetExceeded at {n}: {e}", flush=True)
                                    raise StopIteration
                                for eid, pe in ep["per_edge"].items():
                                    row = {"candidate_id": cid, "tier": tier, "severity": severity,
                                           "mask_id": mask_id, "mask_edges": [RE2.edge_id(e) for e in mask_edges],
                                           "mode": mode, "blinded": blinded, "seed": seed, "cand_k": k,
                                           "edge": eid, "op": pe["op"], "transform": pe["transform"],
                                           "E_repaired": pe["E_repaired"], "S_repaired": pe["S_repaired"],
                                           "S_detail": pe["S_detail"], "I_repaired": pe["I_repaired"],
                                           "I_confidence": pe["I_confidence"], "over_claim_edge": pe["over_claim_edge"],
                                           "consumer_type": pe["consumer_type"], "record_type": "edge"}
                                    fout.write(json.dumps(row) + "\n")
                                srow = {"candidate_id": cid, "tier": tier, "severity": severity,
                                        "mask_id": mask_id, "mode": mode, "blinded": blinded,
                                        "seed": seed, "cand_k": k, "over_claim": ep["over_claim"],
                                        "es_agreement": ep["es_agreement"], "confusion": ep["confusion"],
                                        "n_edges": ep["n_edges"], "record_type": "candidate"}
                                fout.write(json.dumps(srow) + "\n"); fout.flush()
                                summary.append(srow)
                                if n % 20 == 0:
                                    print(f"  [{n}] spent=${client.total_usd-initial:.4f}", flush=True)
    except StopIteration:
        pass
    finally:
        fout.close()

    spent = client.total_usd - initial
    print(f"\n=== Path A-fix static DONE: n={n} elapsed={time.time()-t0:.0f}s spent=${spent:.4f} ===", flush=True)
    from collections import defaultdict
    agg = defaultdict(list)
    for s in summary:
        if s["over_claim"] is not None:
            agg[(s["tier"], s["mode"])].append(s["over_claim"])
    print("  OverClaim by (tier, mode):")
    for key in sorted(agg.keys()):
        v = agg[key]
        print(f"    {key[0]:14} {key[1]:8}: {sum(v)/len(v):.3f} (n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
