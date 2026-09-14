#!/usr/bin/env python3
"""
Random Boolean network simulations for sparse attractor basin recovery.

The experiment is designed to illustrate that sparse basin recovery is not
merely an additive accumulation of single-node effects. In many random Boolean
networks, two controlled nodes can jointly open recovery paths that are not
available when either node is used alone.

Main outputs:
  1. grouped_bar_marginal_gains.{png,pdf}
  2. representative_synergy_heatmap.{png,pdf}
  3. summary_table.{csv,xlsx}
  4. raw_network_results.csv
  5. raw_network_results.pkl
  6. representative_network.pkl

Dependencies:
  numpy, pandas, matplotlib
  openpyxl is optional and only needed for Excel export.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))


NodeSet = Tuple[int, ...]


@dataclass
class BooleanFunction:
    """A local Boolean function represented by inputs and a truth table."""

    inputs: Tuple[int, ...]
    table: np.ndarray


@dataclass
class RandomBooleanNetwork:
    """A synchronous Boolean network with unbiased essential local functions."""

    n: int
    k_in: int
    functions: List[BooleanFunction]
    transition: np.ndarray


@dataclass
class Attractor:
    """A cycle attractor and its basin size."""

    states: Tuple[int, ...]
    basin_size: int

    @property
    def period(self) -> int:
        return len(self.states)

    def basin_fraction(self, state_count: int) -> float:
        return self.basin_size / state_count


@dataclass
class NetworkResult:
    """Results for one accepted random Boolean network."""

    group_n: int
    group_k_in: int
    sample_id: int
    attempts_used: int
    target_states: Tuple[int, ...]
    target_period: int
    B0: float
    B1: float
    B2: float
    delta1: float
    delta2: float
    eta: float
    gamma_max: float
    gamma_argmax: Tuple[int, int]
    best_sets: Dict[int, NodeSet]
    singleton_values: np.ndarray
    pair_values: np.ndarray
    synergy_matrix: np.ndarray
    network: RandomBooleanNetwork


def bit_at(state: int, node: int) -> int:
    """Return Boolean value of node in an integer-encoded state."""

    return (state >> node) & 1


def truth_index(state: int, inputs: Sequence[int]) -> int:
    """Index the truth table by the selected input node values."""

    idx = 0
    for pos, node in enumerate(inputs):
        idx |= bit_at(state, node) << pos
    return idx


def depends_on_all_inputs(table: np.ndarray, k_in: int) -> bool:
    """Check whether all selected variables are essential variables."""

    for var in range(k_in):
        step = 1 << var
        period = 1 << (var + 1)
        essential = False
        for base in range(0, 1 << k_in, period):
            for offset in range(step):
                a = base + offset
                b = a + step
                if table[a] != table[b]:
                    essential = True
                    break
            if essential:
                break
        if not essential:
            return False
    return True


def generate_essential_function(
    rng: np.random.Generator, n: int, k_in: int
) -> BooleanFunction:
    """Generate one unbiased Boolean function depending on all chosen inputs."""

    inputs = tuple(int(x) for x in rng.choice(n, size=k_in, replace=False))
    while True:
        table = rng.integers(0, 2, size=1 << k_in, dtype=np.uint8)
        if depends_on_all_inputs(table, k_in):
            return BooleanFunction(inputs=inputs, table=table)


def build_transition(n: int, functions: Sequence[BooleanFunction]) -> np.ndarray:
    """Construct the global synchronous transition map."""

    state_count = 1 << n
    transition = np.zeros(state_count, dtype=np.uint32)
    for state in range(state_count):
        nxt = 0
        for node, func in enumerate(functions):
            value = int(func.table[truth_index(state, func.inputs)])
            nxt |= value << node
        transition[state] = nxt
    return transition


def generate_random_bn(
    rng: np.random.Generator, n: int, k_in: int
) -> RandomBooleanNetwork:
    """Generate a random Boolean network with essential local functions."""

    functions = [generate_essential_function(rng, n, k_in) for _ in range(n)]
    transition = build_transition(n, functions)
    return RandomBooleanNetwork(n=n, k_in=k_in, functions=functions, transition=transition)


def find_attractors_and_basins(transition: np.ndarray) -> List[Attractor]:
    """Find all attractors and basin sizes in a deterministic finite system."""

    state_count = len(transition)
    assigned = np.full(state_count, -1, dtype=np.int32)
    cycles: List[Tuple[int, ...]] = []
    basin_sizes: List[int] = []

    for start in range(state_count):
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
            attractor_id = len(cycles)
            cycles.append(cycle)
            basin_sizes.append(0)
        else:
            attractor_id = int(assigned[cur])

        for state in path:
            assigned[state] = attractor_id
            basin_sizes[attractor_id] += 1

    return [
        Attractor(states=cycle, basin_size=basin)
        for cycle, basin in zip(cycles, basin_sizes)
    ]


def choose_target_attractor(
    attractors: Sequence[Attractor],
    state_count: int,
    min_fraction: float = 0.05,
    max_fraction: float = 0.50,
    max_period: int = 8,
    target_fraction: float = 0.25,
) -> Optional[Attractor]:
    """Choose the admissible attractor whose basin fraction is closest to 0.25."""

    admissible = [
        attr
        for attr in attractors
        if min_fraction <= attr.basin_fraction(state_count) <= max_fraction
        and attr.period <= max_period
    ]
    if not admissible:
        return None
    return min(
        admissible,
        key=lambda attr: (
            abs(attr.basin_fraction(state_count) - target_fraction),
            attr.period,
        ),
    )


def reverse_reachable_size(
    transition: np.ndarray,
    n: int,
    target_states: Sequence[int],
    controlled_set: Sequence[int],
) -> int:
    """
    Compute |R_S(A_T)| by reverse reachability in the feasible transition graph.

    For fixed S, x is a predecessor of y if the uncontrolled coordinates of
    f(x) match those of y. Controlled coordinates are free and are not tested.
    """

    state_count = 1 << n
    all_mask = state_count - 1
    controlled_mask = 0
    for node in controlled_set:
        controlled_mask |= 1 << node
    uncontrolled_mask = all_mask ^ controlled_mask

    buckets: Dict[int, List[int]] = defaultdict(list)
    for state, successor in enumerate(transition):
        buckets[int(successor) & uncontrolled_mask].append(state)

    reached = np.zeros(state_count, dtype=bool)
    queue: deque[int] = deque()
    for state in target_states:
        state = int(state)
        if not reached[state]:
            reached[state] = True
            queue.append(state)

    while queue:
        y = queue.popleft()
        signature = y & uncontrolled_mask
        for pred in buckets.get(signature, []):
            if not reached[pred]:
                reached[pred] = True
                queue.append(pred)

    return int(reached.sum())


def enumerate_node_sets(n: int, max_size: int) -> List[NodeSet]:
    """Enumerate node sets of size at most max_size."""

    node_sets: List[NodeSet] = []
    for size in range(max_size + 1):
        node_sets.extend(tuple(c) for c in itertools.combinations(range(n), size))
    return node_sets


def compute_sparse_recovery(
    transition: np.ndarray,
    n: int,
    target_states: Sequence[int],
    k_max: int = 2,
) -> Tuple[Dict[NodeSet, float], Dict[int, float], Dict[int, NodeSet]]:
    """Compute B_T^*(S) for |S|<=k_max and B_T^*(K) for K=0..k_max."""

    state_count = 1 << n
    values_by_set: Dict[NodeSet, float] = {}
    best_by_k: Dict[int, float] = {}
    best_set_by_k: Dict[int, NodeSet] = {}

    for node_set in enumerate_node_sets(n, k_max):
        size = reverse_reachable_size(transition, n, target_states, node_set)
        values_by_set[node_set] = size / state_count

    for k in range(k_max + 1):
        candidates = [(value, node_set) for node_set, value in values_by_set.items() if len(node_set) <= k]
        best_value, best_set = max(candidates, key=lambda item: (item[0], -len(item[1])))
        best_by_k[k] = best_value
        best_set_by_k[k] = best_set

    return values_by_set, best_by_k, best_set_by_k


def compute_synergy_matrix(values_by_set: Dict[NodeSet, float], n: int) -> np.ndarray:
    """Compute pairwise synergy Gamma_ij."""

    gamma = np.zeros((n, n), dtype=float)
    base = values_by_set[()]
    for i, j in itertools.combinations(range(n), 2):
        pair = tuple(sorted((i, j)))
        gamma_ij = values_by_set[pair] - values_by_set[(i,)] - values_by_set[(j,)] + base
        gamma[i, j] = gamma_ij
        gamma[j, i] = gamma_ij
    np.fill_diagonal(gamma, np.nan)
    return gamma


def run_one_valid_network(
    rng: np.random.Generator,
    n: int,
    k_in: int,
    sample_id: int,
    max_attempts: int,
    k_max: int,
) -> NetworkResult:
    """Generate networks until one satisfies the target-attractor rule."""

    for attempt in range(1, max_attempts + 1):
        network = generate_random_bn(rng, n, k_in)
        state_count = 1 << n
        attractors = find_attractors_and_basins(network.transition)
        target = choose_target_attractor(attractors, state_count)
        if target is None:
            continue

        values_by_set, best_by_k, best_set_by_k = compute_sparse_recovery(
            network.transition, n, target.states, k_max=k_max
        )
        synergy = compute_synergy_matrix(values_by_set, n)
        gamma_max = float(np.nanmax(synergy))
        max_pos = np.argwhere(np.isclose(synergy, gamma_max, equal_nan=False))[0]
        gamma_argmax = (int(max_pos[0]), int(max_pos[1]))
        if gamma_argmax[0] > gamma_argmax[1]:
            gamma_argmax = (gamma_argmax[1], gamma_argmax[0])

        B0 = best_by_k[0]
        B1 = best_by_k[1]
        B2 = best_by_k[2]
        singleton_values = np.array([values_by_set[(i,)] for i in range(n)], dtype=float)
        pair_values = np.zeros((n, n), dtype=float)
        for i, j in itertools.combinations(range(n), 2):
            pair_value = values_by_set[tuple(sorted((i, j)))]
            pair_values[i, j] = pair_value
            pair_values[j, i] = pair_value

        return NetworkResult(
            group_n=n,
            group_k_in=k_in,
            sample_id=sample_id,
            attempts_used=attempt,
            target_states=target.states,
            target_period=target.period,
            B0=B0,
            B1=B1,
            B2=B2,
            delta1=B1 - B0,
            delta2=B2 - B1,
            eta=B2 - 2 * B1 + B0,
            gamma_max=gamma_max,
            gamma_argmax=gamma_argmax,
            best_sets=best_set_by_k,
            singleton_values=singleton_values,
            pair_values=pair_values,
            synergy_matrix=synergy,
            network=network,
        )

    raise RuntimeError(
        f"Could not find a valid network for n={n}, k_in={k_in} after {max_attempts} attempts."
    )


def result_to_row(result: NetworkResult) -> Dict[str, object]:
    """Flatten one NetworkResult into a CSV-friendly row."""

    return {
        "n": result.group_n,
        "k_in": result.group_k_in,
        "sample_id": result.sample_id,
        "attempts_used": result.attempts_used,
        "target_period": result.target_period,
        "target_states": " ".join(str(s) for s in result.target_states),
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
        "best_S0": " ".join(map(str, result.best_sets[0])),
        "best_S1": " ".join(map(str, result.best_sets[1])),
        "best_S2": " ".join(map(str, result.best_sets[2])),
    }


def mean_std_text(values: Sequence[float], digits: int = 4) -> str:
    """Return mean ± std as text."""

    arr = np.asarray(values, dtype=float)
    ddof = 1 if arr.size > 1 else 0
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=ddof):.{digits}f}"


def build_summary_table(results: Sequence[NetworkResult]) -> pd.DataFrame:
    """Build the grouped summary table."""

    rows = []
    for (n, k_in), group in itertools.groupby(
        sorted(results, key=lambda r: (r.group_n, r.group_k_in)),
        key=lambda r: (r.group_n, r.group_k_in),
    ):
        group = list(group)
        rows.append(
            {
                "n": n,
                "k_in": k_in,
                "B0": mean_std_text([r.B0 for r in group]),
                "B1": mean_std_text([r.B1 for r in group]),
                "B2": mean_std_text([r.B2 for r in group]),
                "Delta1": mean_std_text([r.delta1 for r in group]),
                "Delta2": mean_std_text([r.delta2 for r in group]),
                "p_coop": np.mean([r.eta > 0 for r in group]),
                "Gamma_max": mean_std_text([r.gamma_max for r in group]),
                "valid_networks": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_representative_network(results: Sequence[NetworkResult]) -> Optional[NetworkResult]:
    """
    Select a representative network with positive pairwise synergy.

    Priority is given to (n=12, k_in=5). Within the candidate set, choose the
    network whose Gamma_max is closest to the median positive Gamma_max.
    """

    positive = [r for r in results if r.gamma_max > 0]
    if not positive:
        return None

    preferred = [r for r in positive if r.group_n == 12 and r.group_k_in == 5]
    pool = preferred if preferred else positive
    gamma_values = np.array([r.gamma_max for r in pool], dtype=float)
    median_gamma = float(np.median(gamma_values))
    return min(pool, key=lambda r: abs(r.gamma_max - median_gamma))


def plot_grouped_bar_chart(raw_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot average marginal gains Delta1 and Delta2 for the six groups."""

    import matplotlib.pyplot as plt

    group_df = (
        raw_df.groupby(["n", "k_in"], as_index=False)
        .agg(
            delta1_mean=("delta1", "mean"),
            delta1_std=("delta1", "std"),
            delta2_mean=("delta2", "mean"),
            delta2_std=("delta2", "std"),
        )
        .sort_values(["n", "k_in"])
    )

    labels = [f"({int(row.n)},{int(row.k_in)})" for row in group_df.itertuples()]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        x - width / 2,
        group_df["delta1_mean"],
        width,
        yerr=group_df["delta1_std"],
        capsize=3,
        label=r"$\Delta_1$",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        group_df["delta2_mean"],
        width,
        yerr=group_df["delta2_std"],
        capsize=3,
        label=r"$\Delta_2$",
        color="#F58518",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel(r"Random BN group $(n,k_{\rm in})$")
    ax.set_ylabel("Average marginal gain")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_bar_marginal_gains.png", dpi=300)
    fig.savefig(output_dir / "grouped_bar_marginal_gains.pdf")
    plt.close(fig)


def plot_synergy_heatmap(result: NetworkResult, output_dir: Path) -> None:
    """Plot pairwise synergy heatmap for the representative network."""

    import matplotlib.pyplot as plt

    gamma = result.synergy_matrix.copy()
    max_i, max_j = result.gamma_argmax

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    vmax = np.nanmax(np.abs(gamma))
    vmax = max(vmax, 1e-12)
    im = ax.imshow(gamma, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(
        rf"Representative network: $n={result.group_n}$, $k_{{in}}={result.group_k_in}$"
    )
    ax.set_xlabel("Node index")
    ax.set_ylabel("Node index")
    ax.set_xticks(np.arange(result.group_n))
    ax.set_yticks(np.arange(result.group_n))
    ax.set_xticklabels(np.arange(1, result.group_n + 1))
    ax.set_yticklabels(np.arange(1, result.group_n + 1))
    ax.scatter([max_j], [max_i], s=90, facecolors="none", edgecolors="black", linewidths=1.7)
    ax.scatter([max_i], [max_j], s=90, facecolors="none", edgecolors="black", linewidths=1.7)
    ax.text(
        0.02,
        -0.12,
        rf"max pair: ({max_i + 1},{max_j + 1}), $\Gamma_{{max}}={result.gamma_max:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Pairwise synergy $\Gamma_{ij}$")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_synergy_heatmap.png", dpi=300)
    fig.savefig(output_dir / "representative_synergy_heatmap.pdf")
    plt.close(fig)


def save_representative_details(result: NetworkResult, output_dir: Path) -> None:
    """Save representative network details for reproducibility."""

    details = {
        "n": result.group_n,
        "k_in": result.group_k_in,
        "sample_id": result.sample_id,
        "target_states": result.target_states,
        "B0": result.B0,
        "B1": result.B1,
        "B2": result.B2,
        "delta1": result.delta1,
        "delta2": result.delta2,
        "eta": result.eta,
        "gamma_max": result.gamma_max,
        "gamma_argmax": result.gamma_argmax,
        "best_sets": result.best_sets,
        "inputs": [func.inputs for func in result.network.functions],
        "truth_tables": [func.table.astype(int).tolist() for func in result.network.functions],
        "singleton_values": result.singleton_values,
        "pair_values": result.pair_values,
        "synergy_matrix": result.synergy_matrix,
    }
    with open(output_dir / "representative_network.pkl", "wb") as f:
        pickle.dump(details, f)


def run_experiment(args: argparse.Namespace) -> None:
    """Run all random network experiments."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    groups = [(8, 2), (8, 5), (10, 2), (10, 5), (12, 2), (12, 5)]
    all_results: List[NetworkResult] = []

    for n, k_in in groups:
        print(f"\n=== Group n={n}, k_in={k_in} ===", flush=True)
        accepted = 0
        while accepted < args.networks_per_group:
            result = run_one_valid_network(
                rng=rng,
                n=n,
                k_in=k_in,
                sample_id=accepted,
                max_attempts=args.max_attempts_per_network,
                k_max=args.k_max,
            )
            all_results.append(result)
            accepted += 1
            print(
                f"  accepted {accepted:02d}/{args.networks_per_group}: "
                f"B=({result.B0:.3f},{result.B1:.3f},{result.B2:.3f}), "
                f"eta={result.eta:.3f}, Gamma_max={result.gamma_max:.3f}, "
                f"attempts={result.attempts_used}",
                flush=True,
            )

    raw_df = pd.DataFrame([result_to_row(r) for r in all_results])
    raw_df.to_csv(output_dir / "raw_network_results.csv", index=False)
    with open(output_dir / "raw_network_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    summary_df = build_summary_table(all_results)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)
    try:
        summary_df.to_excel(output_dir / "summary_table.xlsx", index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}", flush=True)

    plot_grouped_bar_chart(raw_df, output_dir)
    representative = select_representative_network(all_results)
    if representative is not None:
        plot_synergy_heatmap(representative, output_dir)
        save_representative_details(representative, output_dir)
    else:
        print("No positive pairwise synergy found; heatmap was not generated.", flush=True)

    print(f"\nResults saved to: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random Boolean network basin recovery simulations."
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--networks-per-group", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=2)
    parser.add_argument("--max-attempts-per-network", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results_random_networks")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
