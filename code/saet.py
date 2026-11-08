"""Evolve a population of agents and test how well their parts combine.

The loop evaluates agents on assigned tasks, selects parents, creates new
agents by mutation or exchange, and saves checkpoints. At evaluation points
it measures new agents from parent pairs and groups parents by those scores."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core import mag, crossover


# Module connections used by the rigid interface check.
RIGID_TYPE_EDGES = [
    ('planner', 'workflow'),
    ('verifier', 'communication'),
    ('verifier', 'update_policy'),
]

# Strict checks cover adjacent module pairs with
# compatible types in founder_v2_typed. Used when mismatch_mode="rigid" AND agent is
# cross-lineage hybrid. Only edges with matching pristine types are checked — the
# others are "structural" (e.g., tools.out=tool_result NOT routed through verifier
# directly) and would always fail.
#
# Verified against pristine founder_v2_typed (all 7 edges below pass):
STRICT_TYPE_EDGES = [
    ('planner', 'workflow'),                # plan_v_linear
    ('workflow', 'memory'),                 # action_seq
    ('memory', 'memory_compactor'),         # memory_chunk
    ('tools', 'tool_chain_compiler'),       # tool_result
    ('verifier', 'communication'),          # verify_result
    ('verifier', 'update_policy'),          # verify_result
    ('verifier', 'error_log'),              # verify_result
]


def check_rigid_interface(genome):
    """Return (ok, fail_reason). If genome has output_type/input_type fields and they
    mismatch on any RIGID_TYPE_EDGES, return False with a reason string."""
    modules = genome.get('modules', {})
    for src, dst in RIGID_TYPE_EDGES:
        s_out = modules.get(src, {}).get('output_type')
        d_in = modules.get(dst, {}).get('input_type')
        if s_out is not None and d_in is not None and s_out != d_in:
            return False, f'TYPE_MISMATCH {src}.{s_out}->{dst}.{d_in}'
    return True, None


def count_rigid_mismatches(genome):
    """Return (count, reasons_list) of rigid interface mismatches (RIGID_TYPE_EDGES only)."""
    modules = genome.get('modules', {})
    count = 0
    reasons = []
    for src, dst in RIGID_TYPE_EDGES:
        s_out = modules.get(src, {}).get('output_type')
        d_in = modules.get(dst, {}).get('input_type')
        if s_out is not None and d_in is not None and s_out != d_in:
            count += 1
            reasons.append(f'{src}.{s_out}->{dst}.{d_in}')
    return count, reasons


def check_strict_interface(genome):
    """Check the required module connections for matching data types.

    Return whether the check passed and the first failure reason, if any."""
    modules = genome.get('modules', {})
    for src, dst in STRICT_TYPE_EDGES:
        s_out = modules.get(src, {}).get('output_type')
        d_in = modules.get(dst, {}).get('input_type')
        # Both must exist for edge to be checkable; otherwise skip
        if s_out is None or d_in is None:
            continue
        if s_out != d_in:
            return False, f'STRICT_TYPE_MISMATCH {src}.{s_out}->{dst}.{d_in}'
    return True, None


def count_strict_mismatches(genome):
    """E1: count of all-edge interface mismatches."""
    modules = genome.get('modules', {})
    count = 0
    reasons = []
    for src, dst in STRICT_TYPE_EDGES:
        s_out = modules.get(src, {}).get('output_type')
        d_in = modules.get(dst, {}).get('input_type')
        if s_out is None or d_in is None:
            continue
        if s_out != d_in:
            count += 1
            reasons.append(f'{src}.{s_out}->{dst}.{d_in}')
    return count, reasons


def is_hybrid(genome):
    """E1: agent is a cross-lineage hybrid iff _lineage_id contains '+' (combined parents)."""
    lid = genome.get('_lineage_id', '')
    return '+' in str(lid)


# ---------------- mutation operators (founder_v0 §3) ----------------

def _bump_version(schema_id: str) -> str:
    """Bump schema version v1 → v2; if no version, add .v2."""
    import re
    m = re.search(r'\.v(\d+)$', schema_id)
    if m:
        n = int(m.group(1))
        return schema_id[:m.start()] + f".v{n+1}"
    return schema_id + ".v2"


def mutate(genome: dict[str, Any], mu: float, rng: random.Random,
            mutate_type_weight: float | None = None) -> dict[str, Any]:
    """Apply one mutation op chosen from update_policy.mutation_distribution.

    If mutate_type_weight is given, override the weight of mutate_output_type and
    mutate_input_type ops to that value and renormalize the rest.
    """
    new = copy.deepcopy(genome)
    up = new["modules"]["update_policy"]
    ops = list(up["mutation_distribution"].keys())
    weights = list(up["mutation_distribution"].values())
    if mutate_type_weight is not None:
        new_weights = []
        for op, w in zip(ops, weights):
            if "type" in op and op in ("mutate_output_type", "mutate_input_type"):
                new_weights.append(mutate_type_weight)
            else:
                new_weights.append(w)
        s = sum(new_weights)
        weights = [w / s for w in new_weights] if s > 0 else weights
    op = rng.choices(ops, weights=weights)[0]
    try:
        _apply_op(new, op, rng)
    except Exception:
        # silent skip if op fails; keep parent genome
        pass
    return new


def _apply_op(g: dict, op: str, rng: random.Random):
    mods = g["modules"]
    rules = g.get("validation_rules", {})
    ranges = rules.get("field_value_ranges", {})
    enums = rules.get("field_enums", {})

    if op == "modify_param":
        # pick a random numeric field within ranges, gaussian perturb
        path = rng.choice(list(ranges.keys()))
        spec = ranges[path]
        parts = path.split(".")
        cur = mods
        for p in parts[:-1]:
            cur = cur[p]
        old = cur[parts[-1]]
        if spec["type"] == "int":
            delta = max(1, int(round(rng.gauss(0, 0.5 * (spec["max"] - spec["min"]) * 0.1))))
            new_v = max(spec["min"], min(spec["max"], old + rng.choice([-1, 1]) * delta))
        else:
            delta = rng.gauss(0, 0.05 * (spec["max"] - spec["min"]))
            new_v = max(spec["min"], min(spec["max"], old + delta))
        cur[parts[-1]] = new_v
    elif op == "replace_module":
        path = rng.choice(list(enums.keys()))
        parts = path.split(".")
        cur = mods
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = rng.choice(enums[path])
    elif op == "modify_tool_interface":
        tools = mods.get("tools", {}).get("tool_specs", {})
        if not tools:
            return
        tname = rng.choice(list(tools.keys()))
        old_id = tools[tname]["schema_id"]
        tools[tname]["schema_id"] = _bump_version(old_id)
    elif op == "modify_memory_schema":
        m = mods.get("memory", {})
        if "schema" in m and "fields" in m["schema"]:
            # randomly add/remove/rename a field
            fields = list(m["schema"]["fields"])
            choice = rng.choice(["add", "remove", "rename"])
            if choice == "add":
                fields.append(f"new_field_{rng.randint(0, 999)}")
            elif choice == "remove" and len(fields) > 1:
                fields.pop(rng.randrange(len(fields)))
            elif choice == "rename" and fields:
                idx = rng.randrange(len(fields))
                fields[idx] = f"{fields[idx]}_v{rng.randint(2, 9)}"
            m["schema"]["fields"] = fields
            # bump output port schema version
            ports = m.get("output_ports", {})
            for k, v in ports.items():
                if "schema_ref" in v:
                    v["schema_ref"] = _bump_version(v["schema_ref"])
    elif op == "modify_communication_schema":
        c = mods.get("communication", {})
        if "schema" in c and enums.get("communication.schema"):
            c["schema"] = rng.choice(enums["communication.schema"])
    elif op == "modify_verifier_or_stopping":
        v = mods.get("verifier", {})
        sr = v.get("stopping_rule", {})
        if "score_threshold" in sr:
            sr["score_threshold"] = max(0.0, min(1.0, sr["score_threshold"] + rng.gauss(0, 0.05)))
    elif op == "mutate_output_type":
        type_opts = ["plan_v_linear", "plan_v_dag", "plan_v_heuristic"]
        candidates = ["planner", "workflow", "memory"]
        eligible = [c for c in candidates if c in mods]
        if eligible:
            m = rng.choice(eligible)
            mods[m]["output_type"] = rng.choice(type_opts)
    elif op == "mutate_input_type":
        type_opts = ["plan_v_linear", "plan_v_dag", "plan_v_heuristic"]
        candidates = ["verifier", "communication", "update_policy"]
        eligible = [c for c in candidates if c in mods]
        if eligible:
            m = rng.choice(eligible)
            mods[m]["input_type"] = rng.choice(type_opts)
    elif op == "mutate_task_focus":
        # E2 (orchestrator 2026-06-26): change genome top-level task_focus tag
        focus_opts = ["none", "planning", "memory", "retrieval"]
        cur = g.get("task_focus", "none")
        choices = [f for f in focus_opts if f != cur]
        if choices:
            g["task_focus"] = rng.choice(choices)
    elif op == "add_workflow_node":
        wf = mods.get("workflow", {})
        nodes = wf.get("nodes", [])
        node_id = f"hidden_{rng.randint(0, 999)}"
        # insert as a passthrough after retriever
        new_node = {"id": node_id, "module_ref": "M"}  # reuse memory module
        nodes.append(new_node)
        wf["nodes"] = nodes
    elif op == "delete_workflow_node":
        wf = mods.get("workflow", {})
        nodes = wf.get("nodes", [])
        # don't delete entry/exit
        deletable = [i for i, n in enumerate(nodes)
                     if isinstance(n, dict) and n.get("id") not in ("planner", "verifier")
                     and n.get("id", "").startswith("hidden_")]
        if deletable:
            nodes.pop(rng.choice(deletable))
    elif op == "rewire_workflow_edge":
        wf = mods.get("workflow", {})
        edges = wf.get("edges", [])
        if len(edges) >= 2:
            # swap two edges' dst
            i, j = rng.sample(range(len(edges)), 2)
            if isinstance(edges[i], dict) and isinstance(edges[j], dict):
                edges[i]["dst"], edges[j]["dst"] = edges[j]["dst"], edges[i]["dst"]


# ---------------- assortative parent selection ----------------

def select_parent_pairs(
    population: list[dict],
    fitness: list[float],
    compat_predictor: Callable[[dict, dict], float] | None,
    beta: float,
    n_pairs: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Softmax over (alpha_K * compat_hat + alpha_F * fitness) — assortative on compatibility.
    For β=0 → random uniform (Thm 3 control)."""
    N = len(population)
    pairs = []
    for _ in range(n_pairs):
        # pick i proportional to fitness
        f_arr = np.array(fitness, dtype=float)
        # ensure positive weights
        weights = f_arr - f_arr.min() + 1e-6
        weights = weights / weights.sum()
        i = int(np.random.choice(N, p=weights))
        if beta <= 0:
            # uniform random j
            j = rng.randrange(N)
            while j == i:
                j = rng.randrange(N)
        else:
            # compat-aware softmax for j given i
            if compat_predictor is None:
                scores = np.array([1.0 if k != i else -1e9 for k in range(N)])
            else:
                scores = np.array([
                    compat_predictor(population[i], population[k]) if k != i else -1e9
                    for k in range(N)
                ])
            scores = scores + 0.1 * np.array(fitness)
            log_w = beta * scores
            log_w = log_w - log_w.max()  # stabilize
            w = np.exp(log_w)
            w = w / w.sum()
            j = int(np.random.choice(N, p=w))
        pairs.append((i, j))
    return pairs


# ---------------- RCM + RCC ----------------

def estimate_rcm(
    population: list[dict],
    eval_pair_hybrid_viability: Callable[[dict, dict, int], float],
    R: int = 8,
    rng_seed: int = 42,
) -> np.ndarray:
    """Pairwise mean hybrid viability matrix. For M5 single-niche control, R can be small."""
    N = len(population)
    rcm = np.zeros((N, N))
    for i in range(N):
        # self-cross diag
        rcm[i, i] = eval_pair_hybrid_viability(population[i], population[i], rng_seed + i * 1000)
        for j in range(i + 1, N):
            v = eval_pair_hybrid_viability(population[i], population[j], rng_seed + i * 1000 + j)
            rcm[i, j] = v
            rcm[j, i] = v
    return rcm


def rcc_partition(rcm: np.ndarray, n_clusters_max: int = 4,
                  tau_in: float = 0.30, tau_out: float = 0.60,
                  n_min: int = 3) -> dict[str, Any]:
    """Spectral clustering with eigengap-based K selection. tau_in/tau_out adjusted for v1 low-fitness regime."""
    from sklearn.cluster import SpectralClustering
    N = rcm.shape[0]
    sym = 0.5 * (rcm + rcm.T)
    sym = np.clip(sym, 0.0, None)
    # eigengap
    try:
        from scipy.linalg import eigh
        D = np.diag(sym.sum(axis=1))
        # avoid degenerate degree
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(sym.sum(axis=1), 1e-9)))
        L_norm = np.eye(N) - d_inv_sqrt @ sym @ d_inv_sqrt
        eigvals = np.sort(np.real(eigh(L_norm, eigvals_only=True)))
    except Exception:
        eigvals = np.array([])

    best_k = 1
    if len(eigvals) >= 3:
        gaps = np.diff(eigvals[: min(n_clusters_max + 1, len(eigvals))])
        if len(gaps) > 0:
            best_k = max(1, int(np.argmax(gaps)) + 1)
            best_k = min(best_k, n_clusters_max)

    if best_k <= 1 or N < n_min * 2:
        return {"n_clusters": 1, "labels": [0] * N, "valid_clusters": [], "eigvals": eigvals.tolist() if len(eigvals) else []}

    try:
        sc = SpectralClustering(n_clusters=best_k, affinity="precomputed",
                                assign_labels="kmeans", random_state=42)
        labels = sc.fit_predict(sym).tolist()
    except Exception:
        labels = [0] * N
        best_k = 1

    # Compute per-cluster K_within / K_between, filter valid
    arr_labels = np.array(labels)
    cluster_ids = sorted(set(labels))
    valid_clusters = []
    for s in cluster_ids:
        idx = np.where(arr_labels == s)[0]
        if len(idx) < n_min:
            continue
        sub = sym[idx][:, idx]
        if sub.shape[0] > 1:
            mask = ~np.eye(sub.shape[0], dtype=bool)
            K_within = float(sub[mask].mean())
        else:
            K_within = float(sub[0, 0])
        K_between = 0.0
        for s2 in cluster_ids:
            if s2 == s:
                continue
            idx2 = np.where(arr_labels == s2)[0]
            if len(idx2) == 0:
                continue
            K_between = max(K_between, float(sym[idx][:, idx2].mean()))
        valid_clusters.append({"id": int(s), "members": idx.tolist(),
                                "K_within": K_within, "K_between": K_between,
                                "size": len(idx)})
    return {"n_clusters": best_k, "labels": labels, "valid_clusters": valid_clusters,
            "eigvals_first8": eigvals[:8].tolist() if len(eigvals) else []}


def compute_rii(rcm: np.ndarray, labels: list[int]) -> dict[str, Any]:
    """RII = 1 - K_AB / sqrt(K_AA * K_BB). Returns mean RII across all inter-species pairs."""
    arr_labels = np.array(labels)
    species_ids = sorted(set(labels))
    if len(species_ids) < 2:
        return {"rii_mean": 0.0, "K_within_mean": float(np.diag(rcm).mean()),
                "K_between_mean": None, "rii_pairs": []}
    K_within = {}
    for s in species_ids:
        idx = np.where(arr_labels == s)[0]
        if len(idx) >= 2:
            sub = rcm[idx][:, idx]
            mask = ~np.eye(sub.shape[0], dtype=bool)
            K_within[s] = float(sub[mask].mean())
        elif len(idx) == 1:
            K_within[s] = float(rcm[idx[0], idx[0]])
        else:
            K_within[s] = 0.0
    rii_pairs = []
    for i, s1 in enumerate(species_ids):
        for s2 in species_ids[i + 1:]:
            idx1 = np.where(arr_labels == s1)[0]
            idx2 = np.where(arr_labels == s2)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue
            K_AB = float(rcm[idx1][:, idx2].mean())
            denom = math.sqrt(K_within[s1] * K_within[s2]) + 1e-9
            rii_val = 1.0 - K_AB / denom
            rii_pairs.append({"pair": (int(s1), int(s2)),
                              "K_AA": K_within[s1], "K_BB": K_within[s2],
                              "K_AB": K_AB, "rii": rii_val})
    return {
        "rii_mean": float(np.mean([p["rii"] for p in rii_pairs])) if rii_pairs else 0.0,
        "K_within_mean": float(np.mean(list(K_within.values()))),
        "K_between_mean": float(np.mean([p["K_AB"] for p in rii_pairs])) if rii_pairs else None,
        "rii_pairs": rii_pairs,
    }


# ---------------- fitness with λ_c ----------------

# Map task group names to the keys used by the fitness multiplier.
# Maps the long niche names used in exp1_cell3 to v3 schema short keys.
NICHE_SHORT_KEY = {
    "planbench_blocksworld": "planbench",
    "planbench_logistics": "planbench",
    "locomo": "locomo",
    "hotpotqa": "hotpotqa",
}


def fitness_with_cost(q: float, cost_usd: float, niche_name: str,
                       cost_model: dict[str, Any],
                       genome: dict[str, Any] | None = None,
                       niche_multiplier_table: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """F = mult(task_focus, niche) * q - λ_c · c_norm - λ_f · f + λ_r · r.

    If genome.task_focus exists and niche_multiplier_table provided (v3+), apply multiplier.
    Otherwise mult=1.0 (backward compat with v1/v2).
    """
    lc = cost_model["lambda_c"]
    c_max_table = cost_model["c_normalization"]["c_max_usd"]
    c_max = c_max_table.get(niche_name, 0.001)
    c_norm = min(max(cost_usd / c_max, 0.0), 2.0)
    mult = 1.0
    task_focus = None
    if genome is not None and niche_multiplier_table is not None:
        task_focus = genome.get("task_focus", "none")
        short_key = NICHE_SHORT_KEY.get(niche_name, niche_name)
        mult = float(niche_multiplier_table.get(task_focus, {}).get(short_key, 1.0))
    q_eff = mult * q
    F = q_eff - lc * c_norm
    return {"F": F, "q": q, "q_eff": q_eff, "niche_multiplier": mult,
             "task_focus": task_focus, "c_norm": c_norm, "c_usd": cost_usd, "lambda_c": lc}
