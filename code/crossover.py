"""Build a new agent by exchanging a connected part between two parents.

The exchange reconnects ports and applies syntax and type checks. Semantic
compatibility is measured by the experiment after the exchange."""
from __future__ import annotations

import copy
import random
from typing import Any

from . import mag


def typed_subgraph_crossover(
    G_A: dict[str, Any],
    G_B: dict[str, Any],
    rng: random.Random | None = None,
    module_choice: str | None = None,
) -> dict[str, Any]:
    """Replace one module of G_B with the corresponding subgraph from G_A.

    Args:
        G_A, G_B: parent MAGs
        rng: random number generator (default new random.Random())
        module_choice: optional fixed module to swap (else uniform random over MODULE_KEYS)

    Returns:
        new child genome (deep copied)
    """
    rng = rng or random.Random()
    if module_choice is None:
        module_choice = rng.choice(mag.MODULE_KEYS)

    Q = mag.extract_typed_subgraph(G_A, module_choice)
    child = mag.replace_subgraph(G_B, module_choice, Q)
    child = mag.reconnect_typed_ports(child)
    child = mag.syntax_fix(child)
    child = mag.type_check_fix(child)
    # Semantic compatibility is assessed after the exchange.
    # Choose the child's task_focus from either parent with equal probability.
    if "task_focus" in G_A or "task_focus" in G_B:
        a_focus = G_A.get("task_focus", "none")
        b_focus = G_B.get("task_focus", "none")
        child["task_focus"] = rng.choice([a_focus, b_focus])
    return child


def uniform_genome_crossover(
    G_A: dict[str, Any],
    G_B: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Choose each module from either parent with equal probability.

    Each choice copies the whole module, including its data types and parameters.
    The resulting connections are checked later by the population evaluator."""
    import copy
    rng = rng or random.Random()
    # Start from a deep copy of G_A as scaffold (preserves top-level fields)
    child = copy.deepcopy(G_A)
    if "modules" not in G_A or "modules" not in G_B:
        return child
    all_modules = set(G_A["modules"].keys()) | set(G_B["modules"].keys())
    for mod_name in all_modules:
        # 50/50 choose source
        if rng.random() < 0.5:
            if mod_name in G_A["modules"]:
                child["modules"][mod_name] = copy.deepcopy(G_A["modules"][mod_name])
            elif mod_name in G_B["modules"]:
                child["modules"][mod_name] = copy.deepcopy(G_B["modules"][mod_name])
        else:
            if mod_name in G_B["modules"]:
                child["modules"][mod_name] = copy.deepcopy(G_B["modules"][mod_name])
            elif mod_name in G_A["modules"]:
                child["modules"][mod_name] = copy.deepcopy(G_A["modules"][mod_name])
    # task_focus uniform 50/50 same as typed_subgraph
    if "task_focus" in G_A or "task_focus" in G_B:
        a_focus = G_A.get("task_focus", "none")
        b_focus = G_B.get("task_focus", "none")
        child["task_focus"] = rng.choice([a_focus, b_focus])
    # No semantic_repair, no reconnect_typed_ports — uniform may produce mismatches by design
    return child


def sanity_self_crossover_identity(genome: dict[str, Any]) -> bool:
    """G x G with any module_choice must equal G (founder self-cross identity)."""
    for mk in mag.MODULE_KEYS:
        child = typed_subgraph_crossover(genome, genome, module_choice=mk)
        # compare modules sub-dict (validation_rules / metadata may diff if we ever mutate; here equal)
        if child["modules"] != genome["modules"]:
            return False
    return True


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../../data/founder_genome_v0.json"
    g = mag.load_mag(path)
    ok = sanity_self_crossover_identity(g)
    print(f"self-crossover identity ({path}): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
