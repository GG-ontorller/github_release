#!/usr/bin/env python3
"""
Planted modular multistable random Boolean networks.

This script generates structured random Boolean networks assembled from weakly
coupled modules. Randomness is kept in the local truth tables and in basin
geometry. Only a small number of global fixed points are planted by constraining
the corresponding truth-table entries; all remaining truth-table entries are
unbiased random values, subject to essential-variable checks.

The generated networks are accepted by rejection filtering only if they have a
nontrivial multistable attractor landscape and a target basin in a prescribed
range. Sparse basin recovery is then computed by reverse reachability in the
feasible transition graph for K=0,1,2.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

NodeSet = Tuple[int, ...]


@dataclass
class BooleanFunction:
    inputs: Tuple[int, ...]
    table: np.ndarray


@dataclass
class PlantedNetwork:
    n: int
    module_count: int
    module_size: int
    modules: List[Tuple[int, ...]]
    functions: List[BooleanFunction]
    transition: np.ndarray
    planted_states: Tuple[int, ...]
    target_state: int


@dataclass
class Attractor:
    states: Tuple[int, ...]
    basin_size: int

    @property
    def period(self) -> int:
        return len(self.states)


@dataclass
class ExperimentResult:
    module_count: int
    module_size: int
    n: int
    sample_id: int
    attempts_used: int
    attractor_count: int
    target_period: int
    target_state: int
    target_basin_size: int
    B0: float
    B1: float
    B2: float
    delta1: float
    delta2: float
    eta: float
    gamma_max: float
    gamma_argmax: Tuple[int, int]
    gamma_pair_cross_module: bool
    best_sets: Dict[int, NodeSet]
    singleton_values: np.ndarray
    pair_values: np.ndarray
    synergy_matrix: np.ndarray
    network: PlantedNetwork


def bit_at(state: int, node: int) -> int:
    return (state >> node) & 1


def set_bit(state: int, node: int, value: int) -> int:
    if value:
        return state | (1 << node)
    return state & ~(1 << node)


def truth_index(state: int, inputs: Sequence[int]) -> int:
    idx = 0
    for pos, node in enumerate(inputs):
        idx |= bit_at(state, node) << pos
    return idx


def hamming_distance(a: int, b: int, nodes: Sequence[int]) -> int:
    return sum(bit_at(a, node) != bit_at(b, node) for node in nodes)


def depends_on_all_inputs(table: np.ndarray, k: int) -> bool:
    for var in range(k):
        step = 1 << var
        period = 1 << (var + 1)
        essential = False
        for base in range(0, 1 << k, period):
            for offset in range(step):
                if table[base + offset] != table[base + offset + step]:
                    essential = True
                    break
            if essential:
                break
        if not essential:
            return False
    return True


def choose_inputs(
    rng: np.random.Generator,
    node: int,
    modules: Sequence[Tuple[int, ...]],
    node_to_module: Sequence[int],
    k_in: int,
    k_inter: int,
) -> Tuple[int, ...]:
    own_module = node_to_module[node]
    own_nodes = list(modules[own_module])
    other_nodes = [v for m, nodes in enumerate(modules) if m != own_module for v in nodes]

    inter = min(k_inter, k_in, len(other_nodes))
    intra = k_in - inter
    chosen: List[int] = []
    chosen.extend(int(x) for x in rng.choice(own_nodes, size=intra, replace=False))
    if inter > 0:
        chosen.extend(int(x) for x in rng.choice(other_nodes, size=inter, replace=False))
    rng.shuffle(chosen)
    return tuple(chosen)


def random_planted_states(
    rng: np.random.Generator,
    modules: Sequence[Tuple[int, ...]],
    min_module_distance: int,
) -> Tuple[int, ...]:
    """Create 2^M global planted states from two random regimes per module."""

    module_patterns: List[Tuple[int, int]] = []
    for nodes in modules:
        while True:
            a = 0
            b = 0
            for node in nodes:
                a = set_bit(a, node, int(rng.integers(0, 2)))
                b = set_bit(b, node, int(rng.integers(0, 2)))
            if hamming_distance(a, b, nodes) >= min_module_distance:
                module_patterns.append((a, b))
                break

    planted: List[int] = []
    for selector in itertools.product([0, 1], repeat=len(modules)):
        state = 0
        for module_id, choice in enumerate(selector):
            pattern = module_patterns[module_id][choice]
            for node in modules[module_id]:
                state = set_bit(state, node, bit_at(pattern, node))
        planted.append(state)
    return tuple(planted)


def generate_function_with_constraints(
    rng: np.random.Generator,
    inputs: Tuple[int, ...],
    node: int,
    planted_states: Sequence[int],
    k_in: int,
    max_tries: int = 5000,
) -> BooleanFunction:
    """Generate an unbiased essential truth table with planted fixed-point rows."""

    constraints: Dict[int, int] = {}
    for state in planted_states:
        idx = truth_index(state, inputs)
        value = bit_at(state, node)
        old = constraints.get(idx)
        if old is not None and old != value:
            raise ValueError("Conflicting planted constraints for one truth-table row.")
        constraints[idx] = value

    for _ in range(max_tries):
        table = rng.integers(0, 2, size=1 << k_in, dtype=np.uint8)
        for idx, value in constraints.items():
            table[idx] = value
        if depends_on_all_inputs(table, k_in):
            return BooleanFunction(inputs=inputs, table=table)

    raise RuntimeError("Could not generate an essential constrained truth table.")


def build_transition(n: int, functions: Sequence[BooleanFunction]) -> np.ndarray:
    transition = np.zeros(1 << n, dtype=np.uint32)
    for state in range(1 << n):
        nxt = 0
        for node, func in enumerate(functions):
            value = int(func.table[truth_index(state, func.inputs)])
            nxt = set_bit(nxt, node, value)
        transition[state] = nxt
    return transition


def generate_planted_network(
    rng: np.random.Generator,
    module_count: int,
    module_size: int,
    k_in: int,
    k_inter: int,
    min_module_distance: int,
) -> PlantedNetwork:
    n = module_count * module_size
    modules = [
        tuple(range(m * module_size, (m + 1) * module_size))
        for m in range(module_count)
    ]
    node_to_module = [0] * n
    for m, nodes in enumerate(modules):
        for node in nodes:
            node_to_module[node] = m

    planted_states = random_planted_states(rng, modules, min_module_distance)
    functions: List[BooleanFunction] = []
    for node in range(n):
        inputs = choose_inputs(rng, node, modules, node_to_module, k_in, k_inter)
        func = generate_function_with_constraints(
            rng, inputs, node, planted_states, k_in
        )
        functions.append(func)

    transition = build_transition(n, functions)
    for state in planted_states:
        if int(transition[state]) != state:
            raise RuntimeError("Planted state was not preserved.")

    target_state = int(rng.choice(planted_states))
    return PlantedNetwork(
        n=n,
        module_count=module_count,
        module_size=module_size,
        modules=modules,
        functions=functions,
        transition=transition,
        planted_states=planted_states,
        target_state=target_state,
    )


def find_attractors_and_basins(transition: np.ndarray) -> List[Attractor]:
    assigned = np.full(len(transition), -1, dtype=np.int32)
    cycles: List[Tuple[int, ...]] = []
    basin_sizes: List[int] = []
    for start in range(len(transition)):
        if assigned[start] >= 0:
            continue
        path: List[int] = []
        local_pos: Dict[int, int] = {}
        cur = start
        while cur not in local_pos and assigned[cur] < 0:
            local_pos[cur] = len(path)
            path.append(cur)
            cur = int(transition[cur])
        if cur in local_pos:
            cycle = tuple(path[local_pos[cur] :])
            attr_id = len(cycles)
            cycles.append(cycle)
            basin_sizes.append(0)
        else:
            attr_id = int(assigned[cur])
        for state in path:
            assigned[state] = attr_id
            basin_sizes[attr_id] += 1
    return [Attractor(c, b) for c, b in zip(cycles, basin_sizes)]


def target_attractor_info(
    attractors: Sequence[Attractor], target_state: int
) -> Optional[Attractor]:
    for attr in attractors:
        if target_state in attr.states:
            return attr
    return None


def reverse_reachable_size(
    transition: np.ndarray,
    n: int,
    target_states: Sequence[int],
    controlled_set: Sequence[int],
) -> int:
    all_mask = (1 << n) - 1
    controlled_mask = 0
    for node in controlled_set:
        controlled_mask |= 1 << node
    uncontrolled_mask = all_mask ^ controlled_mask

    buckets: Dict[int, List[int]] = defaultdict(list)
    for state, successor in enumerate(transition):
        buckets[int(successor) & uncontrolled_mask].append(state)

    reached = np.zeros(1 << n, dtype=bool)
    queue: deque[int] = deque()
    for target in target_states:
        target = int(target)
        reached[target] = True
        queue.append(target)

    while queue:
        y = queue.popleft()
        signature = y & uncontrolled_mask
        for pred in buckets.get(signature, []):
            if not reached[pred]:
                reached[pred] = True
                queue.append(pred)
    return int(reached.sum())


def compute_recovery(
    transition: np.ndarray, n: int, target_states: Sequence[int], k_max: int
) -> Tuple[Dict[NodeSet, float], Dict[int, float], Dict[int, NodeSet]]:
    values: Dict[NodeSet, float] = {}
    total = 1 << n
    for size in range(k_max + 1):
        for node_set in itertools.combinations(range(n), size):
            size_rr = reverse_reachable_size(transition, n, target_states, node_set)
            values[tuple(node_set)] = size_rr / total

    best_values: Dict[int, float] = {}
    best_sets: Dict[int, NodeSet] = {}
    for k in range(k_max + 1):
        candidates = [(v, s) for s, v in values.items() if len(s) <= k]
        value, node_set = max(candidates, key=lambda item: (item[0], -len(item[1])))
        best_values[k] = value
        best_sets[k] = node_set
    return values, best_values, best_sets


def compute_synergy(values: Dict[NodeSet, float], n: int) -> Tuple[np.ndarray, np.ndarray]:
    pair_values = np.zeros((n, n), dtype=float)
    gamma = np.zeros((n, n), dtype=float)
    base = values[()]
    for i, j in itertools.combinations(range(n), 2):
        pair_value = values[(i, j)]
        pair_values[i, j] = pair_values[j, i] = pair_value
        gamma_ij = pair_value - values[(i,)] - values[(j,)] + base
        gamma[i, j] = gamma[j, i] = gamma_ij
    np.fill_diagonal(gamma, np.nan)
    return pair_values, gamma


def module_of_node(network: PlantedNetwork, node: int) -> int:
    for m, nodes in enumerate(network.modules):
        if node in nodes:
            return m
    raise ValueError(node)


def accept_network(
    network: PlantedNetwork,
    min_attractors: int,
    min_basin: float,
    max_basin: float,
    max_target_period: int,
) -> Tuple[bool, Optional[Attractor], List[Attractor]]:
    attractors = find_attractors_and_basins(network.transition)
    target = target_attractor_info(attractors, network.target_state)
    if target is None:
        return False, None, attractors
    target_fraction = target.basin_size / (1 << network.n)
    if len(attractors) < min_attractors:
        return False, target, attractors
    if target.period > max_target_period:
        return False, target, attractors
    if not (min_basin <= target_fraction <= max_basin):
        return False, target, attractors
    return True, target, attractors


def run_one_valid_network(
    rng: np.random.Generator,
    module_count: int,
    module_size: int,
    k_in: int,
    k_inter: int,
    min_module_distance: int,
    sample_id: int,
    k_max: int,
    max_attempts: int,
    min_attractors: int,
    min_basin: float,
    max_basin: float,
    max_target_period: int,
    max_B1: float,
) -> ExperimentResult:
    for attempt in range(1, max_attempts + 1):
        try:
            network = generate_planted_network(
                rng,
                module_count,
                module_size,
                k_in,
                k_inter,
                min_module_distance,
            )
        except (RuntimeError, ValueError):
            continue

        ok, target, attractors = accept_network(
            network,
            min_attractors,
            min_basin,
            max_basin,
            max_target_period,
        )
        if not ok or target is None:
            continue

        values, best_values, best_sets = compute_recovery(
            network.transition, network.n, target.states, k_max
        )
        if best_values[1] > max_B1:
            continue
        pair_values, gamma = compute_synergy(values, network.n)
        gamma_max = float(np.nanmax(gamma))
        max_pos = np.argwhere(np.isclose(gamma, gamma_max, equal_nan=False))[0]
        gamma_argmax = tuple(sorted((int(max_pos[0]), int(max_pos[1]))))
        cross = module_of_node(network, gamma_argmax[0]) != module_of_node(
            network, gamma_argmax[1]
        )
        singleton_values = np.array([values[(i,)] for i in range(network.n)])
        B0, B1, B2 = best_values[0], best_values[1], best_values[2]

        return ExperimentResult(
            module_count=module_count,
            module_size=module_size,
            n=network.n,
            sample_id=sample_id,
            attempts_used=attempt,
            attractor_count=len(attractors),
            target_period=target.period,
            target_state=network.target_state,
            target_basin_size=target.basin_size,
            B0=B0,
            B1=B1,
            B2=B2,
            delta1=B1 - B0,
            delta2=B2 - B1,
            eta=B2 - 2 * B1 + B0,
            gamma_max=gamma_max,
            gamma_argmax=gamma_argmax,
            gamma_pair_cross_module=cross,
            best_sets=best_sets,
            singleton_values=singleton_values,
            pair_values=pair_values,
            synergy_matrix=gamma,
            network=network,
        )
    raise RuntimeError(
        f"No valid network found for modules={module_count}, size={module_size}."
    )


def result_to_row(result: ExperimentResult) -> Dict[str, object]:
    return {
        "module_count": result.module_count,
        "module_size": result.module_size,
        "n": result.n,
        "sample_id": result.sample_id,
        "attempts_used": result.attempts_used,
        "attractor_count": result.attractor_count,
        "target_period": result.target_period,
        "target_state": result.target_state,
        "target_basin_size": result.target_basin_size,
        "B0": result.B0,
        "B1": result.B1,
        "B2": result.B2,
        "delta1": result.delta1,
        "delta2": result.delta2,
        "eta": result.eta,
        "cooperative_jump": result.eta > 0,
        "gamma_max": result.gamma_max,
        "gamma_i": result.gamma_argmax[0],
        "gamma_j": result.gamma_argmax[1],
        "gamma_pair_cross_module": result.gamma_pair_cross_module,
        "best_S0": " ".join(map(str, result.best_sets[0])),
        "best_S1": " ".join(map(str, result.best_sets[1])),
        "best_S2": " ".join(map(str, result.best_sets[2])),
        "planted_states": " ".join(map(str, result.network.planted_states)),
    }


def mean_std(values: Sequence[float], digits: int = 4) -> str:
    arr = np.asarray(values, dtype=float)
    ddof = 1 if arr.size > 1 else 0
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=ddof):.{digits}f}"


def build_summary(results: Sequence[ExperimentResult]) -> pd.DataFrame:
    rows = []
    grouped = itertools.groupby(
        sorted(results, key=lambda r: (r.module_count, r.module_size)),
        key=lambda r: (r.module_count, r.module_size),
    )
    for (module_count, module_size), group_iter in grouped:
        group = list(group_iter)
        rows.append(
            {
                "modules": module_count,
                "module_size": module_size,
                "n": module_count * module_size,
                "B0": mean_std([r.B0 for r in group]),
                "B1": mean_std([r.B1 for r in group]),
                "B2": mean_std([r.B2 for r in group]),
                "Delta1": mean_std([r.delta1 for r in group]),
                "Delta2": mean_std([r.delta2 for r in group]),
                "eta": mean_std([r.eta for r in group]),
                "p_coop": float(np.mean([r.eta > 0 for r in group])),
                "Gamma_max": mean_std([r.gamma_max for r in group]),
                "p_cross_max_pair": float(np.mean([r.gamma_pair_cross_module for r in group])),
                "attempts": mean_std([r.attempts_used for r in group], digits=2),
                "valid_networks": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_representative(results: Sequence[ExperimentResult]) -> ExperimentResult:
    positive = [r for r in results if r.eta > 0 and r.gamma_max > 0]
    pool = positive if positive else list(results)
    eta_values = np.array([r.eta for r in pool])
    median_eta = float(np.median(eta_values))
    return min(pool, key=lambda r: abs(r.eta - median_eta))


def plot_recovery_curve(result: ExperimentResult, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    y = [result.B0, result.B1, result.B2]
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.bar([0, 1, 2], y, color="#D8E6F2", edgecolor="#2F6B9A", linewidth=0.9)
    ax.plot([0, 1, 2], y, color="#2F6B9A", marker="o", linewidth=2.0)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Allowed number of controlled nodes K")
    ax.set_ylabel(r"$B_T^\star(K)$")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        rf"Representative network: M={result.module_count}, m={result.module_size}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "representative_recovery_curve.png", dpi=300)
    fig.savefig(output_dir / "representative_recovery_curve.pdf")
    plt.close(fig)


def plot_synergy_heatmap(result: ExperimentResult, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    gamma = result.synergy_matrix.copy()
    vmax = max(float(np.nanmax(np.abs(gamma))), 1e-12)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(gamma, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(result.n))
    ax.set_yticks(np.arange(result.n))
    ax.set_xticklabels(np.arange(1, result.n + 1), fontsize=8)
    ax.set_yticklabels(np.arange(1, result.n + 1), fontsize=8)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    boundary = result.module_size
    while boundary < result.n:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.0)
        boundary += result.module_size
    i, j = result.gamma_argmax
    ax.scatter([j], [i], s=90, facecolors="none", edgecolors="black", linewidths=1.7)
    ax.scatter([i], [j], s=90, facecolors="none", edgecolors="black", linewidths=1.7)
    ax.set_title(r"Pairwise synergy $\Gamma_{ij}$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Gamma_{ij}$")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_synergy_heatmap.png", dpi=300)
    fig.savefig(output_dir / "representative_synergy_heatmap.pdf")
    plt.close(fig)


def plot_ensemble_stats(raw_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = (
        raw_df.groupby(["module_count", "module_size"], as_index=False)
        .agg(
            p_coop=("cooperative_jump", "mean"),
            gamma_mean=("gamma_max", "mean"),
            gamma_std=("gamma_max", "std"),
            eta_mean=("eta", "mean"),
            eta_std=("eta", "std"),
        )
        .sort_values(["module_count", "module_size"])
    )
    labels = [f"M={int(r.module_count)}, m={int(r.module_size)}" for r in grouped.itertuples()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].bar(x, grouped["p_coop"], color="#4C78A8", edgecolor="black", linewidth=0.6)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel(r"$p_{\rm coop}$")
    axes[0].set_title("Cooperative jump frequency")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x,
        grouped["gamma_mean"],
        yerr=grouped["gamma_std"],
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel(r"$\Gamma_{\max}$")
    axes[1].set_title("Maximum pairwise synergy")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_cooperation_stats.png", dpi=300)
    fig.savefig(output_dir / "ensemble_cooperation_stats.pdf")
    plt.close(fig)


def save_representative(result: ExperimentResult, output_dir: Path) -> None:
    details = {
        "module_count": result.module_count,
        "module_size": result.module_size,
        "n": result.n,
        "target_state": result.target_state,
        "target_basin_size": result.target_basin_size,
        "B0": result.B0,
        "B1": result.B1,
        "B2": result.B2,
        "delta1": result.delta1,
        "delta2": result.delta2,
        "eta": result.eta,
        "gamma_max": result.gamma_max,
        "gamma_argmax": result.gamma_argmax,
        "best_sets": result.best_sets,
        "planted_states": result.network.planted_states,
        "modules": result.network.modules,
        "inputs": [f.inputs for f in result.network.functions],
        "truth_tables": [f.table.astype(int).tolist() for f in result.network.functions],
        "singleton_values": result.singleton_values,
        "pair_values": result.pair_values,
        "synergy_matrix": result.synergy_matrix,
    }
    with open(output_dir / "representative_network.pkl", "wb") as f:
        pickle.dump(details, f)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    groups = [(2, 4), (2, 5), (3, 4)]
    if args.groups == "three":
        groups = [(3, 4), (3, 5)]
    elif args.groups == "three-small":
        groups = [(3, 4)]
    elif args.groups == "five":
        groups = [(5, 3)]
    elif args.groups == "all":
        groups = [(2, 4), (2, 5), (3, 4), (3, 5), (5, 3)]
    results: List[ExperimentResult] = []
    for module_count, module_size in groups:
        print(f"\n=== modules={module_count}, module_size={module_size} ===", flush=True)
        for sample_id in range(args.networks_per_group):
            result = run_one_valid_network(
                rng=rng,
                module_count=module_count,
                module_size=module_size,
                k_in=args.k_in,
                k_inter=args.k_inter,
                min_module_distance=args.min_module_distance,
                sample_id=sample_id,
                k_max=args.k_max,
                max_attempts=args.max_attempts_per_network,
                min_attractors=args.min_attractors,
                min_basin=args.min_basin,
                max_basin=args.max_basin,
                max_target_period=args.max_target_period,
                max_B1=args.max_B1,
            )
            results.append(result)
            print(
                f"  {sample_id + 1:02d}/{args.networks_per_group}: "
                f"attempts={result.attempts_used}, attractors={result.attractor_count}, "
                f"B=({result.B0:.3f},{result.B1:.3f},{result.B2:.3f}), "
                f"eta={result.eta:.3f}, Gamma_max={result.gamma_max:.3f}",
                flush=True,
            )

    raw_df = pd.DataFrame(result_to_row(r) for r in results)
    raw_df.to_csv(output_dir / "raw_modular_results.csv", index=False)
    with open(output_dir / "raw_modular_results.pkl", "wb") as f:
        pickle.dump(results, f)
    summary = build_summary(results)
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    try:
        summary.to_excel(output_dir / "summary_table.xlsx", index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}", flush=True)

    representative = select_representative(results)
    plot_recovery_curve(representative, output_dir)
    plot_synergy_heatmap(representative, output_dir)
    plot_ensemble_stats(raw_df, output_dir)
    save_representative(representative, output_dir)
    print(f"\nResults saved to: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planted modular multistable random BN recovery experiment."
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--networks-per-group", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=2)
    parser.add_argument("--k-in", type=int, default=3)
    parser.add_argument("--k-inter", type=int, default=1)
    parser.add_argument("--min-module-distance", type=int, default=2)
    parser.add_argument("--min-attractors", type=int, default=3)
    parser.add_argument("--min-basin", type=float, default=0.05)
    parser.add_argument("--max-basin", type=float, default=0.50)
    parser.add_argument("--max-target-period", type=int, default=8)
    parser.add_argument(
        "--max-B1",
        type=float,
        default=1.0,
        help="Reject networks whose best one-node recovery exceeds this value.",
    )
    parser.add_argument("--max-attempts-per-network", type=int, default=2000)
    parser.add_argument(
        "--groups",
        choices=["three", "three-small", "five", "all"],
        default="three",
        help="Which modular groups to run.",
    )
    parser.add_argument("--output-dir", type=str, default="results_modular_multistable")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
