"""Store and modify an agent setup for model and workflow search.

A setup records the model and execution method for each role, how the roles
connect, and a lineage ID. Mutation changes these choices. Crossover combines
choices from two parents. The task protocol sets tools, prompts, turn limits,
and token budgets."""
from __future__ import annotations

import copy
import random

# Claude model choices used by the search.
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


def mutate(genotype, rng, mu=0.3):
    g = copy.deepcopy(genotype)
    if rng.random() < mu * 0.2:
        new_topo = rng.choice([t for t in TOPOLOGIES if t != g["topology"]])
        old = list(g["roles"].values())
        g["topology"] = new_topo
        g["roles"] = {}
        for i, r in enumerate(TOPOLOGY_ROLES[new_topo]):
            src = old[i % len(old)] if old else default_role()
            g["roles"][r] = copy.deepcopy(src)
        return g
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
    """Return whether an expensive model plans and a cheaper model executes.

    Model groups and planner/executor roles are defined by this module."""
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
