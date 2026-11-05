"""Phase 2 genotype: role→model + HARNESS + topology (author terminology, 2026-07-26).

Terminology fix: the field formerly called "scaffold" is the **HARNESS** — the key
evolvable dimension (react/cot/direct/reflexion/...). Industry fixes harness to compare
models; our thesis is harness must CO-EVOLVE with model (cost-aware model×harness).

Genotype:
{
  "roles": {"<role>": {"model": ..., "harness": ..., "params": {...}}, ...},
  "topology": "single|pipeline|debate|router",
  "topology_params": {...},
  "_lineage_id": str,
}

Same-X fair-comparison fixed set (NOT in genotype — enforced by protocol layer):
  same tools / same prompt template / same repository / same max_turns / same token_budget.
Harness is DELIBERATELY EXCLUDED from the fixed set (that exclusion is the novelty).
"""
from __future__ import annotations

import copy
import random

# Claude-only gene pool (author 2026-07-31): spans capability-cost gradient
# haiku (cheap) → sonnet → opus-5 (flagship). Controls for vendor → cleaner model-tier
# × harness isolation.
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-sonnet-5",
          "claude-opus-4-8", "claude-opus-5"]
HARNESSES = ["direct", "cot", "react", "reflexion"]

# expensive-tier (flagship) vs cheap-tier — for the "flagship-plans / cheap-executes" pattern
_EXPENSIVE_TIER = ("opus-5", "opus-4-8", "gpt-5.5")   # top capability/cost
_CHEAP_TIER = ("haiku", "gpt-5.4-mini")               # low cost
TOPOLOGIES = ["single", "pipeline", "debate", "router"]

TOPOLOGY_ROLES = {
    "single": ["solver"],
    "pipeline": ["planner", "executor"],
    "debate": ["solver_a", "solver_b", "judge"],
    "router": ["router", "coder", "planner"],
}


def default_role(model="gpt-5.4-mini", harness="react"):
    # note: params carry only genotype-level knobs; same-X fixed items (max_turns,
    # token_budget) are injected by the protocol layer, NOT stored here.
    return {"model": model, "harness": harness, "params": {}}


def make_genotype(topology="single", model="gpt-5.4-mini", harness="react", lineage_id="G0"):
    roles = {r: default_role(model, harness) for r in TOPOLOGY_ROLES[topology]}
    return {"roles": roles, "topology": topology, "topology_params": {}, "_lineage_id": lineage_id}


def random_genotype(rng: random.Random, lineage_id="Grnd"):
    topo = rng.choice(TOPOLOGIES)
    g = {"roles": {}, "topology": topo, "topology_params": {}, "_lineage_id": lineage_id}
    for r in TOPOLOGY_ROLES[topo]:
        g["roles"][r] = {"model": rng.choice(MODELS), "harness": rng.choice(HARNESSES), "params": {}}
    return g


# multi-role (composition) topologies only — used to GUARANTEE composition exploration in
# the initial population (orchestrator 2026-08-01: single-anchor seeding + small pop caused some
# seeds to under-explore comps → unfair comp-vs-single Pareto). "single" excluded.
_COMPOSITION_TOPOLOGIES = [t for t in TOPOLOGIES if t != "single"]


def random_composition_genotype(rng: random.Random, lineage_id="Gcmp"):
    """Random genotype restricted to MULTI-ROLE (composition) topologies."""
    topo = rng.choice(_COMPOSITION_TOPOLOGIES)
    g = {"roles": {}, "topology": topo, "topology_params": {}, "_lineage_id": lineage_id}
    for r in TOPOLOGY_ROLES[topo]:
        g["roles"][r] = {"model": rng.choice(MODELS), "harness": rng.choice(HARNESSES), "params": {}}
    return g


def mutate(genotype, rng, mu=0.3):
    g = copy.deepcopy(genotype)
    if rng.random() < mu * 0.2:
        # STRICT-NEUTRAL two-step topology mutation (orchestrator 2026-08-02): naive
        # rng.choice(other topologies) drifts toward compositions because 3 of 4
        # topologies are multi-role (single->comp 100% vs comp->single 33% of changes).
        # Two-step removes that: step1 pick target CATEGORY single-vs-composition 50/50,
        # step2 if composition pick which one uniformly. This makes single<->comp mutation
        # symmetric (50% each direction) so exploration is neutral for the comp-vs-single
        # fairness test.
        want_single = rng.random() < 0.5  # step1: category 50/50
        if want_single:
            new_topo = "single"
        else:
            new_topo = rng.choice(_COMPOSITION_TOPOLOGIES)  # step2: which composition
        if new_topo != g["topology"]:
            old = list(g["roles"].values())
            g["topology"] = new_topo
            g["roles"] = {}
            for i, r in enumerate(TOPOLOGY_ROLES[new_topo]):
                src = old[i % len(old)] if old else default_role()
                g["roles"][r] = copy.deepcopy(src)
            return g
        # else: same topology → no-op, fall through to param mutation
    for r, spec in g["roles"].items():
        if rng.random() < mu:
            spec["model"] = rng.choice(MODELS)
        if rng.random() < mu:
            spec["harness"] = rng.choice(HARNESSES)
    return g


def crossover(g_a, g_b, rng):
    """Generalized cross-vendor role-wise crossover (harness + model recombine)."""
    if g_a["topology"] == g_b["topology"]:
        child = {"roles": {}, "topology": g_a["topology"],
                 "topology_params": copy.deepcopy(g_a["topology_params"]),
                 "_lineage_id": _mix(g_a, g_b)}
        for r in TOPOLOGY_ROLES[g_a["topology"]]:
            src = g_a if rng.random() < 0.5 else g_b
            child["roles"][r] = copy.deepcopy(src["roles"].get(r, default_role()))
        return child
    base = g_a if rng.random() < 0.5 else g_b
    other = g_b if base is g_a else g_a
    child = {"roles": {}, "topology": base["topology"],
             "topology_params": copy.deepcopy(base["topology_params"]), "_lineage_id": _mix(g_a, g_b)}
    other_specs = list(other["roles"].values())
    for i, r in enumerate(TOPOLOGY_ROLES[base["topology"]]):
        spec = copy.deepcopy(base["roles"].get(r, default_role()))
        if other_specs:
            borrow = other_specs[i % len(other_specs)]
            if rng.random() < 0.5:
                spec["model"] = borrow["model"]
            if rng.random() < 0.5:
                spec["harness"] = borrow["harness"]
        child["roles"][r] = spec
    return child


def _mix(a, b):
    la, lb = a.get("_lineage_id", "?"), b.get("_lineage_id", "?")
    return la if la == lb else f"{la}+{lb}"


def _vendor(model):
    return "claude" if "claude" in model else "gpt"


def is_cross_vendor(g):
    vendors = set(_vendor(r["model"]) for r in g["roles"].values())
    return len(vendors) > 1


# roles considered "planner-like" (reasoning/orchestration) vs "executor-like" (action)
_PLANNER_ROLES = {"planner", "router", "solver", "judge", "solver_a", "solver_b"}
_EXECUTOR_ROLES = {"executor", "coder"}


def _is_expensive(model):
    """Expensive/flagship tier (opus-5, opus-4-8, gpt-5.5)."""
    m = model.lower()
    return any(t in m for t in _EXPENSIVE_TIER)


def _is_cheap(model):
    """Cheap tier (haiku, gpt-5.4-mini)."""
    m = model.lower()
    return any(t in m for t in _CHEAP_TIER)


def models_in_genotype(g):
    """Set of all model strings used across the genotype's roles."""
    return {spec["model"] for spec in g.get("roles", {}).values() if spec.get("model")}


def is_haiku_only(g):
    """True iff EVERY role uses a haiku model (no opus/sonnet).

    Used to gate τ³ under the proxy quota wall: opus/sonnet comps cannot complete τ³
    episodes (time-varying 429), so only haiku-only comps get real τ³ signal; others
    have τ³ skipped+tagged (tau_quota_skipped) to avoid 429-corrupted zeros.
    """
    ms = models_in_genotype(g)
    return bool(ms) and all("haiku" in m.lower() for m in ms)


def opus_role_pattern(g):
    """Tag the composition's EXPENSIVE-TIER (flagship) role usage — generalized for
    Claude-only (author 2026-07-31). Surfaces the headline pattern:
    'flagship-tier-as-planner + cheap-tier-as-executor'. Name kept for back-compat.
    Returns: no_flagship / flagship_as_planner / flagship_as_executor /
             flagship_everywhere / flagship_mixed.
    """
    roles = g["roles"]
    exp_roles = {r for r, spec in roles.items() if _is_expensive(spec["model"])}
    if not exp_roles:
        return "no_flagship"
    if exp_roles == set(roles.keys()):
        return "flagship_everywhere"
    in_planner = any(r in _PLANNER_ROLES for r in exp_roles)
    in_executor = any(r in _EXECUTOR_ROLES for r in exp_roles)
    if in_planner and not in_executor:
        return "flagship_as_planner"
    if in_executor and not in_planner:
        return "flagship_as_executor"
    return "flagship_mixed"


def expensive_planner_cheap_executor(g):
    """True iff a flagship-tier model is in a planner-like role AND a cheap-tier model in
    an executor-like role (the discovered optimal cost-aware pattern:
    flagship-plans / cheap-executes)."""
    roles = g["roles"]
    planner_expensive = any(_is_expensive(spec["model"]) for r, spec in roles.items()
                            if r in _PLANNER_ROLES)
    executor_cheap = any(_is_cheap(spec["model"]) for r, spec in roles.items()
                         if r in _EXECUTOR_ROLES)
    has_executor_role = any(r in _EXECUTOR_ROLES for r in roles)
    return bool(planner_expensive and has_executor_role and executor_cheap)


def is_hetero_model(g):
    return len(set(r["model"] for r in g["roles"].values())) > 1


def genotype_signature(g):
    parts = [g["topology"]]
    for r in TOPOLOGY_ROLES.get(g["topology"], sorted(g["roles"].keys())):
        s = g["roles"].get(r, {})
        m = s.get("model", "?").replace("claude-", "c-").replace("gpt-5.4", "g")
        parts.append(f"{r}={m}/{s.get('harness', '?')}")
    return " ".join(parts)


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(3):
        g = random_genotype(rng)
        print(genotype_signature(g), "| cross_vendor:", is_cross_vendor(g))
