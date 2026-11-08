"""Repeat repair tests with a fixed set of mismatched edges.

The mismatch is introduced at the start. Each later repair attempt is tested
as an independent episode. The output records claimed and measured success at each checkpoint."""
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
from pathA_fix_static import sample_edge_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="T2_nearmiss")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--mask-id", dest="mask_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=5)
    ap.add_argument("--blinded", action="store_true", default=False)
    ap.add_argument("--labeled", dest="blinded", action="store_false")
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", default="./results/pathA_fix_dynamic.jsonl")
    ap.add_argument("--tracker", default="./budget_tracker.json")
    args = ap.parse_args()

    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="pathAfix_dyn", max_retries=6)
    initial = client.total_usd
    rng = random.Random(args.seed)
    mask_rng = random.Random(hash((args.tier, args.severity)) & 0xFFFFFFFF)
    masks = sample_edge_masks(args.severity, args.mask_id + 1 if args.severity < 7 else 1, mask_rng)
    mask_edges = masks[min(args.mask_id, len(masks) - 1)]
    mode = "blinded" if args.blinded else "labeled"

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    def checkpoint(gen):
        """Measure OverClaim over N independent repair episodes at this gen."""
        ocs = []
        for a in range(args.N):
            try:
                ep = RE2.measure_repair_episode_v2(
                    mask_edges, client, purpose=f"pathAfix_dyn_{args.tier}_{mode}_g{gen}",
                    tier=args.tier, blinded=args.blinded, seed=args.seed * 1000 + gen * 10 + a)
            except BudgetExceeded:
                raise
            cid = f"{args.tier}_sev{args.severity}_m{args.mask_id}_{mode}_s{args.seed}_g{gen}_a{a}"
            for eid, pe in ep["per_edge"].items():
                fout.write(json.dumps({
                    "candidate_id": cid, "tier": args.tier, "severity": args.severity,
                    "mask_id": args.mask_id, "mode": mode, "blinded": args.blinded,
                    "seed": args.seed, "gen": gen, "agent_idx": a, "edge": eid,
                    "op": pe["op"], "transform": pe["transform"], "E_repaired": pe["E_repaired"],
                    "S_repaired": pe["S_repaired"], "I_repaired": pe["I_repaired"],
                    "I_confidence": pe["I_confidence"], "over_claim_edge": pe["over_claim_edge"],
                    "record_type": "edge"}) + "\n")
            fout.write(json.dumps({
                "candidate_id": cid, "tier": args.tier, "severity": args.severity,
                "mask_id": args.mask_id, "mode": mode, "blinded": args.blinded,
                "seed": args.seed, "gen": gen, "agent_idx": a, "over_claim": ep["over_claim"],
                "es_agreement": ep["es_agreement"], "confusion": ep["confusion"],
                "record_type": "candidate"}) + "\n")
            if ep["over_claim"] is not None:
                ocs.append(ep["over_claim"])
        fout.flush()
        return sum(ocs) / len(ocs) if ocs else None

    t0 = time.time()
    traj = []
    print(f"=== Path A-fix dynamic: {args.tier} sev{args.severity} {mode} seed{args.seed} "
          f"mask={[RE2.edge_id(e) for e in mask_edges]} ===", flush=True)
    try:
        for gen in range(0, args.T + 1, args.eval_every):
            oc = checkpoint(gen)
            traj.append((gen, oc))
            print(f"  gen{gen} OverClaim={oc}", flush=True)
    except BudgetExceeded as e:
        print(f"BudgetExceeded: {e}", flush=True)
    finally:
        fout.close()
    spent = client.total_usd - initial
    print(f"\n=== DONE: elapsed {time.time()-t0:.0f}s spent ${spent:.4f} ===", flush=True)
    print(f"  trajectory: {traj}", flush=True)


if __name__ == "__main__":
    main()
