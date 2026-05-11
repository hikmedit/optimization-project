"""Exact, MILP-compatible, and equity-aware p-median solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_COVERAGE_THRESHOLDS, DEFAULT_EPSILON
from .utils import weighted_percentile


@dataclass
class Problem:
    """Prepared p-median problem instance."""

    demand: pd.DataFrame
    existing_sites: pd.DataFrame
    candidate_sites: pd.DataFrame
    all_sites: pd.DataFrame
    time_matrix: np.ndarray
    weights: np.ndarray
    existing_indices: np.ndarray
    candidate_indices: np.ndarray
    travel_time_note: str


@dataclass
class Solution:
    """Solver output with assignments and evaluation metrics."""

    method: str
    p: int
    selected_site_indices: tuple[int, ...]
    active_site_indices: tuple[int, ...]
    assignment_site_indices: np.ndarray
    response_times: np.ndarray
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def selected_site_names(self, sites: pd.DataFrame) -> list[str]:
        return sites.iloc[list(self.selected_site_indices)]["site_name"].tolist()


def evaluate_selection(
    problem: Problem,
    selected_site_indices: Iterable[int],
    method: str,
    p: int | None = None,
    runtime_seconds: float = 0.0,
    metadata: dict[str, object] | None = None,
    coverage_thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> Solution:
    """Assign each demand zone to the nearest active site and compute metrics."""

    selected = tuple(sorted(int(idx) for idx in selected_site_indices))
    active = tuple(sorted(set(problem.existing_indices.tolist()).union(selected)))
    active_matrix = problem.time_matrix[:, list(active)]
    nearest_positions = active_matrix.argmin(axis=1)
    assignment_site_indices = np.array([active[pos] for pos in nearest_positions], dtype=int)
    response_times = active_matrix[np.arange(active_matrix.shape[0]), nearest_positions]
    weights = problem.weights
    total = float(np.dot(weights, response_times))
    weighted_average = float(total / weights.sum())
    metrics: dict[str, float | int | str] = {
        "objective_total_weighted_minutes": total,
        "weighted_average_minutes": weighted_average,
        "max_minutes": float(response_times.max()),
        "p90_minutes": weighted_percentile(response_times, weights, 90),
        "unweighted_average_minutes": float(np.mean(response_times)),
        "selected_count": len(selected),
    }
    for threshold in coverage_thresholds:
        covered = response_times <= threshold
        key = str(threshold).replace(".", "_")
        metrics[f"coverage_{key}_minutes"] = float(weights[covered].sum() / weights.sum())
        metrics[f"coverage_count_{key}_minutes"] = int(covered.sum())
    return Solution(
        method=method,
        p=len(selected) if p is None else p,
        selected_site_indices=selected,
        active_site_indices=active,
        assignment_site_indices=assignment_site_indices,
        response_times=response_times,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        metadata=metadata or {},
    )


def solve_p_median_milp(problem: Problem, p: int) -> Solution:
    """Solve p-median with PuLP when available; otherwise exact enumeration.

    The enumeration fallback is exact for this district-candidate setting and is
    used in the notebook environment when PuLP/OR-Tools are not installed.
    """

    try:
        import pulp
    except Exception:
        return solve_p_median_exact(problem, p, method="MILP exact fallback (enumeration)")

    start = time.perf_counter()
    n_demands, n_sites = problem.time_matrix.shape
    existing = set(problem.existing_indices.tolist())
    candidates = set(problem.candidate_indices.tolist())
    model = pulp.LpProblem("fire_station_p_median", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(n_sites)}
    y = {
        (i, j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n_demands)
        for j in range(n_sites)
    }
    model += pulp.lpSum(problem.weights[i] * problem.time_matrix[i, j] * y[i, j] for i in range(n_demands) for j in range(n_sites))
    for i in range(n_demands):
        model += pulp.lpSum(y[i, j] for j in range(n_sites)) == 1
    for i in range(n_demands):
        for j in range(n_sites):
            model += y[i, j] <= x[j]
    for j in existing:
        model += x[j] == 1
    for j in range(n_sites):
        if j not in existing and j not in candidates:
            model += x[j] == 0
    model += pulp.lpSum(x[j] for j in candidates) == p
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = tuple(j for j in candidates if pulp.value(x[j]) and pulp.value(x[j]) > 0.5)
    runtime = time.perf_counter() - start
    solution = evaluate_selection(problem, selected, method="MILP p-median", p=p, runtime_seconds=runtime)
    solution.metadata["solver_status"] = pulp.LpStatus[model.status]
    return solution


def solve_p_median_exact(
    problem: Problem,
    p: int,
    method: str = "Exact p-median enumeration",
    max_combinations: int = 1_000_000,
) -> Solution:
    """Exact p-median enumeration over new candidate districts."""

    start = time.perf_counter()
    selected, metadata = _enumerate_best(problem, p, objective="weighted_average", max_combinations=max_combinations)
    runtime = time.perf_counter() - start
    return evaluate_selection(problem, selected, method=method, p=p, runtime_seconds=runtime, metadata=metadata)


def solve_equity_refinement(
    problem: Problem,
    p: int,
    p_median_solution: Solution | None = None,
    epsilon: float = DEFAULT_EPSILON,
    max_combinations: int = 1_000_000,
) -> Solution:
    """Minimize the worst district response within an average-time tolerance."""

    start = time.perf_counter()
    if p_median_solution is None:
        p_median_solution = solve_p_median_milp(problem, p)
    objective_bound = p_median_solution.metrics["objective_total_weighted_minutes"] * (1.0 + epsilon)
    selected, metadata = _enumerate_best(
        problem,
        p,
        objective="minimax_with_bound",
        objective_bound=float(objective_bound),
        max_combinations=max_combinations,
    )
    metadata.update(
        {
            "epsilon": epsilon,
            "p_median_weighted_average_minutes": p_median_solution.metrics["weighted_average_minutes"],
            "allowed_weighted_average_minutes": p_median_solution.metrics["weighted_average_minutes"] * (1.0 + epsilon),
        }
    )
    runtime = time.perf_counter() - start
    return evaluate_selection(problem, selected, method="Equity-refined p-median", p=p, runtime_seconds=runtime, metadata=metadata)


def bisection_min_budget(
    problem: Problem,
    max_p: int = 8,
    target_coverage: float = 0.90,
    threshold_minutes: float = 8.0,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[int | None, pd.DataFrame]:
    """Find the smallest p that reaches a weighted coverage target."""

    rows = []
    feasible_p: int | None = None
    low, high = 0, max_p
    while low <= high:
        mid = (low + high) // 2
        base = solve_p_median_milp(problem, mid)
        solution = solve_equity_refinement(problem, mid, base, epsilon=epsilon)
        coverage_key = f"coverage_{str(threshold_minutes).replace('.', '_')}_minutes"
        coverage = float(solution.metrics[coverage_key])
        rows.append(
            {
                "p": mid,
                "coverage": coverage,
                "weighted_average_minutes": solution.metrics["weighted_average_minutes"],
                "max_minutes": solution.metrics["max_minutes"],
                "meets_target": coverage >= target_coverage,
            }
        )
        if coverage >= target_coverage:
            feasible_p = mid
            high = mid - 1
        else:
            low = mid + 1
    return feasible_p, pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def equity_sensitivity(
    problem: Problem,
    p: int,
    epsilons: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Show the average-vs-gap tradeoff for different equity tolerances."""

    pure = solve_p_median_milp(problem, p)
    rows = []
    for epsilon in epsilons:
        solution = solve_equity_refinement(problem, p, pure, epsilon=epsilon)
        rows.append(
            {
                "epsilon": epsilon,
                "allowed_average_min": pure.metrics["weighted_average_minutes"] * (1.0 + epsilon),
                "weighted_average_min": solution.metrics["weighted_average_minutes"],
                "max_min": solution.metrics["max_minutes"],
                "p90_min": solution.metrics["p90_minutes"],
                "coverage_8": solution.metrics.get("coverage_8_0_minutes"),
                "opened_sites": "; ".join(solution.selected_site_names(problem.all_sites)),
            }
        )
    return pd.DataFrame(rows)


def solution_table(problem: Problem, solutions: list[Solution]) -> pd.DataFrame:
    """Compact comparison table for notebook and report."""

    rows = []
    for sol in solutions:
        row = {
            "method": sol.method,
            "p": sol.p,
            "weighted_avg_min": sol.metrics["weighted_average_minutes"],
            "max_min": sol.metrics["max_minutes"],
            "p90_min": sol.metrics["p90_minutes"],
            "coverage_5": sol.metrics.get("coverage_5_0_minutes", sol.metrics.get("coverage_5_minutes")),
            "coverage_8": sol.metrics.get("coverage_8_0_minutes", sol.metrics.get("coverage_8_minutes")),
            "runtime_s": sol.runtime_seconds,
            "opened_sites": "; ".join(sol.selected_site_names(problem.all_sites)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def validate_solution(problem: Problem, solution: Solution) -> None:
    """Raise if a solution violates core assignment/opening constraints."""

    if len(solution.assignment_site_indices) != len(problem.demand):
        raise AssertionError("each demand district must have one assignment")
    if len(set(solution.selected_site_indices)) != solution.p:
        raise AssertionError("solution must open exactly p distinct new sites")
    if not set(problem.existing_indices.tolist()).issubset(solution.active_site_indices):
        raise AssertionError("existing stations must remain active")
    if not set(solution.selected_site_indices).issubset(set(problem.candidate_indices.tolist())):
        raise AssertionError("new stations must be selected from candidate district sites")
    if not np.all(np.isfinite(solution.response_times)):
        raise AssertionError("response times must be finite")


def _enumerate_best(
    problem: Problem,
    p: int,
    objective: str,
    objective_bound: float | None = None,
    max_combinations: int = 1_000_000,
) -> tuple[tuple[int, ...], dict[str, object]]:
    candidate_indices = tuple(int(idx) for idx in problem.candidate_indices)
    if p < 0 or p > len(candidate_indices):
        raise ValueError(f"p must be between 0 and {len(candidate_indices)}")
    total_combinations = math.comb(len(candidate_indices), p)
    if total_combinations > max_combinations:
        raise ValueError(
            f"exact search would evaluate {total_combinations:,} combinations; "
            "reduce p or use GA/SA for this scenario"
        )
    baseline = problem.time_matrix[:, problem.existing_indices].min(axis=1)
    candidate_matrix = problem.time_matrix[:, list(candidate_indices)]
    weights = problem.weights
    best_combo_positions: tuple[int, ...] | None = None
    best_score: tuple[float, float, float] | None = None
    feasible_count = 0
    for combo_positions in combinations(range(len(candidate_indices)), p):
        response = baseline if p == 0 else np.minimum(baseline, candidate_matrix[:, combo_positions].min(axis=1))
        total = float(np.dot(weights, response))
        avg = total / weights.sum()
        max_response = float(response.max())
        p90 = weighted_percentile(response, weights, 90)
        if objective == "weighted_average":
            score = (avg, max_response, p90)
        elif objective == "minimax_with_bound":
            if objective_bound is None:
                raise ValueError("objective_bound is required for minimax_with_bound")
            if total > objective_bound + 1e-9:
                continue
            feasible_count += 1
            score = (max_response, p90, avg)
        else:
            raise ValueError(f"unknown objective: {objective}")
        if best_score is None or score < best_score:
            best_score = score
            best_combo_positions = tuple(combo_positions)
    if best_combo_positions is None:
        # Numerical fallback: if no bounded equity solution is found, use the
        # pure p-median optimum rather than returning an invalid selection.
        if objective == "minimax_with_bound":
            return _enumerate_best(problem, p, "weighted_average", max_combinations=max_combinations)
        raise RuntimeError("no feasible candidate combination found")
    selected = tuple(candidate_indices[pos] for pos in best_combo_positions)
    metadata = {
        "objective": objective,
        "evaluated_combinations": total_combinations,
        "feasible_combinations": feasible_count if objective == "minimax_with_bound" else total_combinations,
        "solver": "exact enumeration over candidate districts",
    }
    return selected, metadata
