"""Search for agent setups with useful combinations of success and cost.

The search changes role models, execution methods, and workflow structure.
It tracks candidates for which no other evaluated candidate is both at least
as successful and at most as costly, with a strict improvement in one measure.
All candidates use the shared task protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from core.llm_client import LLMClient, BudgetExceeded
from phase2 import genotype as GT
from phase2.samex_protocol import SameXProtocol, SameXEnforcer
from phase2.agent_exec import run_genotype_on_task
from phase0.eval_harness import get_tasks, make_task  # benchmark-pluggable task specs


def _atomic_write_json(path, obj):
    """Atomic JSON write: write to tmp then os.replace (never leaves a 0-byte/corrupt file
    if the process is killed or json.dumps raises mid-write). 2026-10-27 fix."""
    import os as _os, tempfile as _tf
    d = _os.path.dirname(path) or "."
    try:
        payload = json.dumps(obj, indent=2, default=str)  # serialize FIRST (if this raises, old file intact)
    except Exception as _e:
        print(f"[atomic_write] json.dumps failed, keeping old file: {_e}", flush=True)
        return
    fd, tmp = _tf.mkstemp(dir=d, suffix=".tmp")
    try:
        with _os.fdopen(fd, "w") as _fh:
            _fh.write(payload)
        _os.replace(tmp, path)  # atomic on same filesystem
    except Exception as _e:
        print(f"[atomic_write] write/replace failed: {_e}", flush=True)
        try: _os.unlink(tmp)
        except Exception: pass


def evaluate_composition(genotype, bench_specs, client, protocol, enforcer, purpose,
                         prefilter_benchmarks=None, prefilter_threshold=0.0,
                         tau_haiku_only=False):
    """Evaluate an agent setup across the selected benchmarks.

    Optional screening tasks run first. If their success rate is below the
    configured threshold, later benchmarks are skipped. The result records
    which benchmarks ran and the success and cost on each."""
    enforcer.audit(genotype.get("_lineage_id", "?") + "|" + GT.genotype_signature(genotype))
    prefilter_set = set(prefilter_benchmarks or [])
    # order: prefilter benchmarks first, then main-signal
    ordered = ([bs for bs in bench_specs if bs[0] in prefilter_set] +
               [bs for bs in bench_specs if bs[0] not in prefilter_set])

    total_pass = total_tasks = 0
    total_cost = 0.0
    per_bench = {}
    pre_pass = pre_tasks = 0
    prefilter_done = False

    _bench_errors = []
    def _run_bench(bname, specs):
        nonlocal total_cost
        p = n = 0
        cost = 0.0
        for i, spec in enumerate(specs):
            task = make_task(spec)
            try:
                r = run_genotype_on_task(genotype, task, client, protocol=protocol,
                                         enforcer=enforcer, purpose=f"{purpose}_{bname}_t{i}")
            except BudgetExceeded:
                raise
            except Exception as _te:
                r = {"success": 0, "cost_usd": 0.0}
                _bench_errors.append(f"{type(_te).__name__}")
            p += r["success"]; n += 1; cost += r["cost_usd"]
        total_cost += cost
        return p, n, cost

    # τ³ quota gate (orchestrator 2026-08-01): opus/sonnet comps can't complete τ³ episodes
    # under the proxy's time-varying 429 quota → skip τ³ for non-haiku-only comps and tag,
    # so opus-τ³ 429-zeros don't corrupt the frontier. haiku-only comps get real τ³ signal.
    _tau_skip_quota = tau_haiku_only and not GT.is_haiku_only(genotype)

    for bname, specs in ordered:
        is_pre = bname in prefilter_set
        # τ³ quota skip for non-haiku-only comps
        if bname == "taubench" and _tau_skip_quota:
            per_bench[bname] = {"SR": None, "skipped_tau_quota": True, "n": len(specs),
                                "note": "opus/sonnet τ³ skipped: proxy quota (time-varying 429)"}
            continue
        # gate: if prefilter finished and composition failed it, skip main-signal benches
        if (not is_pre) and prefilter_set and prefilter_done:
            pre_sr = pre_pass / pre_tasks if pre_tasks else 0.0
            if pre_sr < prefilter_threshold:
                per_bench[bname] = {"SR": None, "skipped_prefilter": True, "n": len(specs)}
                continue
        p, n, cost = _run_bench(bname, specs)
        per_bench[bname] = {"SR": round(p / n, 4) if n else 0, "passes": p, "n": n,
                            "cost": round(cost, 6), "tier": "prefilter" if is_pre else "main"}
        total_pass += p; total_tasks += n
        if is_pre:
            pre_pass += p; pre_tasks += n
        # mark prefilter complete after the last prefilter benchmark
        if is_pre and bname == ordered[len([b for b in ordered if b[0] in prefilter_set]) - 1][0]:
            prefilter_done = True

    sr = total_pass / total_tasks if total_tasks else 0.0
    cop = total_cost / total_pass if total_pass > 0 else None
    pre_sr = pre_pass / pre_tasks if pre_tasks else None
    passed = (pre_sr is None or pre_sr >= prefilter_threshold)
    # which benchmarks actually ran for THIS comp (main-signal skipped comps ran fewer)
    benches_run = [b for b, rec in per_bench.items()
                   if not rec.get("skipped_prefilter") and not rec.get("skipped_tau_quota")]
    main_evaluated = passed and any(b not in (prefilter_benchmarks or []) for b in benches_run)
    tau_quota_skipped = any(rec.get("skipped_tau_quota") for rec in per_bench.values())
    return {"success_rate": round(sr, 4), "total_passes": total_pass, "total_tasks": total_tasks,
            "n_task_errors": len(_bench_errors), "task_error_types": _bench_errors[:10],
            "total_cost": round(total_cost, 6), "cost_of_pass": round(cop, 6) if cop else None,
            "prefilter_sr": round(pre_sr, 4) if pre_sr is not None else None,
            "passed_prefilter": passed,
            # ⚠️ METHODOLOGY (orchestrator 2026-07-31): SR denominators differ between
            # main-evaluated comps (all benchmarks) and prefilter-only comps (prefilter
            # benchmarks only). main_signal_evaluated flags which, so the frontier is
            # interpretable + pruned comps stay on the front (at prefilter capability +
            # saved τ³ cost) rather than vanishing.
            "main_signal_evaluated": main_evaluated,
            "benchmarks_scored": benches_run,
            "tau_quota_skipped": tau_quota_skipped,
            "tau_haiku_only_comp": GT.is_haiku_only(genotype),
            "cross_vendor": GT.is_cross_vendor(genotype), "hetero_model": GT.is_hetero_model(genotype),
            "opus_role_pattern": GT.opus_role_pattern(genotype),
            "expensive_planner_cheap_executor": GT.expensive_planner_cheap_executor(genotype),
            "per_bench": per_bench}


def pareto_front(pop):
    front = []
    for i, a in enumerate(pop):
        dom = False
        for j, b in enumerate(pop):
            if i == j:
                continue
            fa, fb = a["fit"], b["fit"]
            if (fb["success_rate"] >= fa["success_rate"] and fb["total_cost"] <= fa["total_cost"]
                    and (fb["success_rate"] > fa["success_rate"] or fb["total_cost"] < fa["total_cost"])):
                dom = True
                break
        if not dom:
            front.append(a)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "claude-sonnet-4-6", "claude-opus-4-8"])
    ap.add_argument("--benchmarks", nargs="+", default=["alfworld", "humaneval"])
    ap.add_argument("--alfworld-n", dest="alf_n", type=int, default=8)
    ap.add_argument("--humaneval-n", dest="he_n", type=int, default=6)
    ap.add_argument("--planbench-n", dest="pb_n", type=int, default=6)
    ap.add_argument("--apibank-n", dest="ab_n", type=int, default=6)
    ap.add_argument("--taubench-n", dest="tau_n", type=int, default=0)
    ap.add_argument("--tau-domain", dest="tau_domain", default="retail")
    ap.add_argument("--tau-user-model", dest="tau_user_model", default="gpt-5.4-mini",
                    help='Model used to simulate the retail customer.')
    ap.add_argument("--prefilter", nargs="+", default=[],
                    help='Benchmarks to run first. Agents below the screening threshold skip the remaining benchmarks.')
    ap.add_argument("--prefilter-threshold", dest="prefilter_threshold", type=float, default=0.0)
    ap.add_argument("--tau-haiku-only", dest="tau_haiku_only", action="store_true",
                    help='Run retail tasks only for agents whose roles all use Haiku.')
    ap.add_argument("--comp-timeout", dest="comp_timeout", type=int, default=1200,
                    help='Time limit in seconds for each agent evaluation. A timeout records zero success.')
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--warm-start-from", type=str, default=None,
                    help='Start a new search using the best setups from a saved run. The generation counter starts over.')
    ap.add_argument("--comp-quota", dest="comp_quota", type=float, default=0.0,
                    help='Minimum share of the starting population that must use multiple roles.')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--max-turns", dest="max_turns", type=int, default=30)
    ap.add_argument("--token-budget", dest="token_budget", type=int, default=100000)
    # type-adaptive same-X: coding fixes repository+tool_set (vs agentic env+tool_api)
    ap.add_argument("--benchmark-type", dest="benchmark_type", default="agentic",
                    choices=["agentic", "coding"])
    ap.add_argument("--repository-id", dest="repository_id", default="N/A",
                    help='Task-set identifier used for coding benchmark comparisons.')
    ap.add_argument("--tool-set-id", dest="tool_set_id", default="N/A",
                    help="coding same-X: fixed tool set id (e.g. swe-agent-tools-fixed)")
    ap.add_argument("--budget", type=float, default=1000.0)
    ap.add_argument("--out", default="./results/phase2_samex_search.json")
    ap.add_argument("--tracker", default="./budget_tracker.json")
    args = ap.parse_args()

    GT.MODELS = list(args.models)
    # type-adaptive same-X (author 2026-08-02 coding activation): agentic fixes env+tool_api;
    # coding fixes repository+tool_set. harness stays the evolved dimension either way.
    protocol = SameXProtocol(max_turns=args.max_turns, token_budget=args.token_budget,
                             benchmark_type=args.benchmark_type,
                             repository_id=args.repository_id, tool_set_id=args.tool_set_id)
    enforcer = SameXEnforcer(protocol)
    client = LLMClient(budget_usd_hard_cap=args.budget, tracker_path=args.tracker,
                       purpose_tag="phase2_samex", max_retries=6)
    initial = client.total_usd
    rng = random.Random(args.seed)

    bench_specs = []
    if "alfworld" in args.benchmarks:
        bench_specs.append(("alfworld", get_tasks("alfworld", "valid_seen", args.alf_n, seed=args.seed, max_steps=args.max_turns)))
    if "humaneval" in args.benchmarks:
        bench_specs.append(("humaneval", get_tasks("humaneval", "", args.he_n, seed=args.seed)))
    if "planbench" in args.benchmarks:
        bench_specs.append(("planbench", get_tasks("planbench", "", args.pb_n, seed=args.seed)))
    if "apibank" in args.benchmarks:
        bench_specs.append(("apibank", get_tasks("apibank", "", args.ab_n, seed=args.seed)))
    if "taubench" in args.benchmarks:
        # set the τ³ user-sim model (provenance: gpt-mini=current, claude-haiku=authoritative)
        from phase2 import taubench_task as _TB
        _TB.TAU_USER_MODEL = args.tau_user_model
        bench_specs.append(("taubench", get_tasks("taubench", args.tau_domain, args.tau_n, seed=args.seed, max_steps=args.max_turns)))
    # provenance: record which benchmarks this frontier used
    log_benchmarks_used = [b for b, _ in bench_specs]
    fully_claude = all("claude" in m for m in args.models) and \
        ("taubench" not in log_benchmarks_used or "claude" in args.tau_user_model)

    # Initial population (orchestrator 2026-08-01 comp-exploration fix): keep the single-model
    # anchors (fair single baselines, one per model) BUT guarantee a minimum quota of
    # multi-role COMPOSITIONS so every seed adequately samples comps (avoids the seed1
    # under-exploration artifact where only 8/47 comps were evaluated). Applied identically
    # across all seeds (documented search-config).
    pop = []
    warm_from = getattr(args, "warm_start_from", None)
    if warm_from:
        # WARM-RESTART (orchestrator 2026-08-08, author PI request "extend exploration"):
        # seed the initial population from a PRIOR run's final Pareto front genotypes so
        # evolution CONTINUES from prior elites rather than random anchors. This is NOT a
        # pure checkpoint-resume: the generation counter resets and fitness is re-evaluated;
        # it is honestly a warm-restart that INHERITS the prior run's elite genotypes.
        import json as _json
        prior = _json.load(open(warm_from))
        prior_gens = prior.get("generations", [])
        seed_sigs = []
        if prior_gens:
            for f in prior_gens[-1].get("front", []):
                seed_sigs.append(f["sig"])
        # parse sigs → genotypes (reuse the high-n parser semantics inline)
        _MMAP = {"c-opus-5": "claude-opus-5", "c-opus-4-8": "claude-opus-4-8",
                 "c-sonnet-5": "claude-sonnet-5", "c-sonnet-4-6": "claude-sonnet-4-6",
                 "c-haiku-4-5-20251001": "claude-haiku-4-5-20251001"}
        def _parse(sig, i):
            parts = sig.split(); topo = parts[0]; roles = {}
            for p in parts[1:]:
                role, spec = p.split("=", 1); m, h = spec.rsplit("/", 1)
                roles[role] = {"model": _MMAP.get(m, m), "harness": h, "params": {}}
            return {"roles": roles, "topology": topo, "topology_params": {},
                    "_lineage_id": f"warm{i}"}
        for i, sig in enumerate(seed_sigs):
            if len(pop) >= args.pop:
                break
            try:
                pop.append({"g": _parse(sig, i), "fit": None})
            except Exception as _e:
                print(f"  [warm] skip unparseable sig: {sig[:40]} ({_e})", flush=True)
        n_warm = len(pop)
        # fill remaining slots with random compositions (keep exploring)
        while len(pop) < args.pop:
            pop.append({"g": GT.random_composition_genotype(rng, f"wcmp{len(pop)}"), "fit": None})
        pop = pop[:args.pop]
        n_singles = sum(1 for m in pop if m["g"].get("topology") == "single")
        n_comp_seeded = n_warm - n_singles
        print(f"  WARM-RESTART from {warm_from}: {n_warm} prior-elite genotypes + "
              f"{args.pop - n_warm} random comps (NOT pure resume; gen counter reset)", flush=True)
    else:
        for m in args.models:
            pop.append({"g": GT.make_genotype("single", m, "react", f"seed_{m}"), "fit": None})
        n_singles = len(pop)
        # guarantee at least comp_quota fraction of the pop are compositions
        n_comp_min = int(round(args.comp_quota * args.pop))
        n_comp_seeded = 0
        while len(pop) < args.pop and n_comp_seeded < n_comp_min:
            pop.append({"g": GT.random_composition_genotype(rng, f"cmp{n_comp_seeded}"), "fit": None})
            n_comp_seeded += 1
        # fill any remaining slots with fully-random genotypes
        while len(pop) < args.pop:
            pop.append({"g": GT.random_genotype(rng, f"rnd{len(pop)}"), "fit": None})
        pop = pop[:args.pop]
        print(f"  init pop: {n_singles} single anchors + {n_comp_seeded} guaranteed comps + "
              f"{args.pop - n_singles - n_comp_seeded} random (comp_quota={args.comp_quota})", flush=True)

    log = {"config": vars(args), "protocol": protocol.as_dict(),
           "benchmarks_used": log_benchmarks_used,  # PROVENANCE (orchestrator guardrail)
           "is_headline": ("taubench" in log_benchmarks_used),  # τ³ = headline main signal
           "tau_user_sim_model": args.tau_user_model,  # PROVENANCE (orchestrator 2026-07-31)
           "fully_claude": fully_claude,  # True iff agent pool + tau3 user-sim all Claude
           "run_label": ("fully-Claude (Claude-haiku user-sim)" if fully_claude
                         else "Claude-agent + " + args.tau_user_model + "-user-sim"),
           "generations": [], "all_evaluated": []}
    t0 = time.time()
    nid = 0

    import signal as _signal

    class _CompTimeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _CompTimeout()

    def eval_pop(members):
        nonlocal nid
        for mem in members:
            if mem["fit"] is not None:
                continue
            # ROBUSTNESS (orchestrator 2026-07-27): per-composition wall-clock watchdog so a
            # hung/429-storm comp can NEVER wedge the whole run. On timeout OR any error →
            # score the comp 0 + continue. Per-gen checkpoint already persists progress.
            _signal.signal(_signal.SIGALRM, _alarm)
            _signal.alarm(args.comp_timeout)
            try:
                mem["fit"] = evaluate_composition(mem["g"], bench_specs, client, protocol, enforcer, f"p2_{nid}",
                                                  prefilter_benchmarks=args.prefilter,
                                                  prefilter_threshold=args.prefilter_threshold,
                                                  tau_haiku_only=args.tau_haiku_only)
            except BudgetExceeded as e:
                mem["fit"] = {"success_rate": 0, "total_cost": 0, "cost_of_pass": None,
                              "cross_vendor": GT.is_cross_vendor(mem["g"]), "error": str(e)}
            except _CompTimeout:
                mem["fit"] = {"success_rate": 0, "total_cost": 0, "cost_of_pass": None,
                              "cross_vendor": GT.is_cross_vendor(mem["g"]),
                              "opus_role_pattern": GT.opus_role_pattern(mem["g"]),
                              "error": f"comp_timeout_{args.comp_timeout}s"}
                print(f"  [comp {nid}] TIMEOUT >{args.comp_timeout}s → scored 0, continuing", flush=True)
            except Exception as e:  # noqa: BLE001 — never let one comp kill the run
                mem["fit"] = {"success_rate": 0, "total_cost": 0, "cost_of_pass": None,
                              "cross_vendor": GT.is_cross_vendor(mem["g"]),
                              "opus_role_pattern": GT.opus_role_pattern(mem["g"]),
                              "error": f"comp_error:{type(e).__name__}"}
                print(f"  [comp {nid}] ERROR {type(e).__name__} → scored 0, continuing", flush=True)
            finally:
                _signal.alarm(0)
            log["all_evaluated"].append({"sig": GT.genotype_signature(mem["g"]),
                                         "cross_vendor": mem["fit"].get("cross_vendor"),
                                         "fit": mem["fit"]})
            nid += 1

    print(f"=== Phase 2 same-X search: models={args.models} pop={args.pop} gens={args.gens} ===", flush=True)
    print(f"    protocol: {protocol.signature()}", flush=True)
    print(f"    benchmarks: {[(b, len(s)) for b, s in bench_specs]}", flush=True)
    try:
        eval_pop(pop)
        for gen in range(args.gens):
            front = pareto_front(pop)
            # per-gen FULL-POP composition/single census (orchestrator 2026-08-02 method-
            # validation: prove exploration adequacy THROUGH generations, not just front).
            _pop_single = sum(1 for m in pop if m["g"].get("topology") == "single")
            _pop_comp = len(pop) - _pop_single
            gr = {"gen": gen,
                  "pop_comp": _pop_comp, "pop_single": _pop_single,  # full-pop census
                  "front": [{"sig": GT.genotype_signature(m["g"]),
                                         "cross_vendor": m["fit"]["cross_vendor"],
                                         "opus_role_pattern": m["fit"].get("opus_role_pattern"),
                                         "expensive_planner_cheap_executor": m["fit"].get("expensive_planner_cheap_executor"),
                                         "main_signal_evaluated": m["fit"].get("main_signal_evaluated"),
                                         "benchmarks_scored": m["fit"].get("benchmarks_scored"),
                                         "SR": m["fit"]["success_rate"], "cost": m["fit"]["total_cost"],
                                         "CoP": m["fit"]["cost_of_pass"]} for m in front]}
            log["generations"].append(gr)
            print(f"\n  gen{gen} pop[{_pop_comp}c/{_pop_single}s] Pareto ({len(front)}):", flush=True)
            for f in gr["front"]:
                tag = "XVND" if f["cross_vendor"] else "1vnd"
                print(f"    [{tag}] {f['sig']:52} SR={f['SR']:.3f} cost=${f['cost']:.4f} CoP={f['CoP']}", flush=True)
            log["samex_audit"] = enforcer.audit_summary()
            _atomic_write_json(args.out, log)
            if gen == args.gens - 1:
                break
            # POP-SIZING FIX (2026-10-20): cap elite carryover to pop//2 so the child-creation
            # loop ALWAYS runs (mutation never freezes when front grows to pop size). Prevents the
            # gptonly-style freeze (front==pop → new_pop=list(front) full → 0 children).
            _elite_keep = max(1, args.pop // 2)
            new_pop = list(front)[:_elite_keep]
            parents = front if len(front) >= 2 else pop
            while len(new_pop) < args.pop:
                pa, pb = rng.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
                child = GT.mutate(GT.crossover(pa["g"], pb["g"], rng), rng, mu=args.mu)
                new_pop.append({"g": child, "fit": None})
            pop = new_pop[:args.pop]
            eval_pop(pop)
    except BudgetExceeded as e:
        print(f"BudgetExceeded: {e}", flush=True)
    finally:
        log["spent_usd"] = round(client.total_usd - initial, 4)
        log["elapsed_sec"] = round(time.time() - t0, 1)
        # machine-checked fairness
        try:
            enforcer.assert_constant()
            log["samex_constant"] = True
        except AssertionError as e:
            log["samex_constant"] = False
            log["samex_error"] = str(e)
        log["samex_audit"] = enforcer.audit_summary()
        # two-tier provenance (orchestrator 2026-07-26): N evaluated, M passed prefilter → main signal
        ev = log.get("all_evaluated", [])
        n_eval = len(ev)
        n_passed = sum(1 for e in ev if e.get("fit", {}).get("passed_prefilter"))
        main_benches = [b for b, _ in bench_specs if b not in set(args.prefilter)]
        # front composition: how many front comps are main-evaluated vs prefilter-only
        front_now = pareto_front(pop)
        n_front_main = sum(1 for m in front_now if m["fit"].get("main_signal_evaluated"))
        log["twotier_provenance"] = {
            "prefilter_benchmarks": list(args.prefilter),
            "prefilter_threshold": args.prefilter_threshold,
            "main_signal_benchmarks": main_benches,
            "n_compositions_evaluated": n_eval,
            "n_passed_prefilter_to_main": n_passed,
            "n_skipped_prefilter": n_eval - n_passed,
            # front still represents full population (pruned comps appear at prefilter
            # capability, tagged main_signal_evaluated=false):
            "front_size": len(front_now),
            "n_front_main_evaluated": n_front_main,
            "n_front_prefilter_only": len(front_now) - n_front_main,
            "note": "SR denominators differ: main-evaluated=all benchmarks, "
                    "prefilter-only=prefilter benchmarks. See per-comp main_signal_evaluated.",
        }
        _atomic_write_json(args.out, log)
        if args.prefilter:
            print(f"  two-tier: {n_passed}/{n_eval} compositions passed prefilter (>={args.prefilter_threshold}) "
                  f"→ main signal {main_benches}; {n_eval-n_passed} skipped", flush=True)

    front = pareto_front(pop)
    # frontier pattern summary (orchestrator 2026-07-26): which opus-role patterns dominate?
    from collections import Counter
    fpat = Counter(m["fit"].get("opus_role_pattern") for m in front)
    n_epce = sum(1 for m in front if m["fit"].get("expensive_planner_cheap_executor"))
    log["frontier_pattern_summary"] = {
        "opus_role_pattern_counts": dict(fpat),
        "n_expensive_planner_cheap_executor_on_front": n_epce,
        "front_size": len(front),
    }
    _atomic_write_json(args.out, log)
    print(f"\n=== DONE spent ${client.total_usd-initial:.4f} elapsed {time.time()-t0:.0f}s ===", flush=True)
    print(f"  same-X constant: {log.get('samex_constant')}", flush=True)
    print(f"  frontier opus-role patterns: {dict(fpat)} | expensive-planner+cheap-executor on front: {n_epce}/{len(front)}", flush=True)
    print("  FINAL Pareto (XVND=cross-vendor, 1vnd=single-vendor):", flush=True)
    for m in sorted(front, key=lambda x: -x["fit"]["success_rate"]):
        tag = "XVND" if m["fit"]["cross_vendor"] else "1vnd"
        pat = m["fit"].get("opus_role_pattern", "?")
        print(f"    [{tag}/{pat}] {GT.genotype_signature(m['g']):52} SR={m['fit']['success_rate']:.3f} "
              f"cost=${m['fit']['total_cost']:.4f} CoP={m['fit']['cost_of_pass']}", flush=True)


if __name__ == "__main__":
    main()
