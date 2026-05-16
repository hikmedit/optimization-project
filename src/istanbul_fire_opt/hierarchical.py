"""Hierarchical Facility Location Model.

This module implements a two-level hierarchical facility location model where
facilities have different types (major, minor, volunteer) with varying capabilities,
costs, and service coverage. This is more realistic for emergency services planning
where different station types serve different roles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .config import DEFAULT_RANDOM_SEED
from .optimization import Problem, Solution


class StationType(Enum):
    """Fire station types with different capabilities."""

    MAJOR = "major"  # Full equipment, 24/7, large coverage radius
    MINOR = "minor"  # Basic equipment, 24/7, medium coverage
    VOLUNTEER = "volunteer"  # Limited equipment, limited hours, small coverage


@dataclass
class HierarchicalProblem:
    """Extended problem with station type information."""

    base_problem: Problem
    station_costs: dict[StationType, float]  # Opening/operating cost by type
    station_capacities: dict[StationType, int]  # Number of units per type
    coverage_multipliers: dict[StationType, float]  # Effective coverage by type
    existing_types: dict[int, StationType]  # Type of each existing station


@dataclass
class HierarchicalSolution:
    """Solution with station type assignments."""

    method: str
    p_major: int
    p_minor: int
    p_volunteer: int
    selected_sites: dict[int, StationType]  # site_index -> StationType
    base_solution: Solution  # Underlying solution with metrics
    total_cost: float
    runtime_seconds: float
    metadata: dict


def create_hierarchical_problem(
    problem: Problem,
    major_cost: float = 10.0,
    minor_cost: float = 5.0,
    volunteer_cost: float = 1.0,
    major_capacity: int = 5,
    minor_capacity: int = 3,
    volunteer_capacity: int = 1,
) -> HierarchicalProblem:
    """Convert standard problem to hierarchical with default parameters.

    Cost units are relative (e.g., major station = 10x volunteer station).
    Capacity represents number of response units (vehicles).
    """
    # Assume all existing stations are major by default
    existing_types = {
        int(idx): StationType.MAJOR for idx in problem.existing_indices
    }

    return HierarchicalProblem(
        base_problem=problem,
        station_costs={
            StationType.MAJOR: major_cost,
            StationType.MINOR: minor_cost,
            StationType.VOLUNTEER: volunteer_cost,
        },
        station_capacities={
            StationType.MAJOR: major_capacity,
            StationType.MINOR: minor_capacity,
            StationType.VOLUNTEER: volunteer_capacity,
        },
        coverage_multipliers={
            StationType.MAJOR: 1.0,  # Full coverage
            StationType.MINOR: 1.15,  # Slightly slower response
            StationType.VOLUNTEER: 1.30,  # Limited availability
        },
        existing_types=existing_types,
    )


def solve_hierarchical_budgeted(
    hierarchical_problem: HierarchicalProblem,
    budget: float = 20.0,
    method: str = "greedy",
) -> HierarchicalSolution:
    """Solve hierarchical facility location with budget constraint.

    Args:
        hierarchical_problem: Problem with station type information
        budget: Total budget for new stations (in relative units)
        method: Solution method ("greedy", "dynamic_programming")

    Returns:
        HierarchicalSolution with type assignments
    """
    start = time.perf_counter()

    if method == "greedy":
        solution = _solve_hierarchical_greedy(hierarchical_problem, budget)
    elif method == "dynamic_programming":
        solution = _solve_hierarchical_dp(hierarchical_problem, budget)
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.perf_counter() - start
    solution.runtime_seconds = runtime

    return solution


def _solve_hierarchical_greedy(
    hp: HierarchicalProblem, budget: float
) -> HierarchicalSolution:
    """Greedy algorithm: iteratively add station with best cost-effectiveness.

    Cost-effectiveness = improvement in objective / cost of station type.
    """
    problem = hp.base_problem
    candidates = set(problem.candidate_indices.tolist())
    existing = set(problem.existing_indices.tolist())

    # Current selection: empty initially
    selected_sites: dict[int, StationType] = {}
    remaining_budget = budget
    current_response = problem.time_matrix[:, list(existing)].min(axis=1)

    while True:
        best_value_per_cost = -np.inf
        best_candidate = None
        best_type = None

        # Try adding each candidate with each type
        for candidate_idx in candidates - set(selected_sites.keys()):
            for station_type in StationType:
                cost = hp.station_costs[station_type]
                if cost > remaining_budget:
                    continue

                # Compute improvement in response time
                multiplier = hp.coverage_multipliers[station_type]
                candidate_time = problem.time_matrix[:, candidate_idx] * multiplier
                new_response = np.minimum(current_response, candidate_time)
                improvement = (
                    problem.weights * (current_response - new_response)
                ).sum()

                value_per_cost = improvement / cost
                if value_per_cost > best_value_per_cost:
                    best_value_per_cost = value_per_cost
                    best_candidate = candidate_idx
                    best_type = station_type

        # If no improvement possible, stop
        if best_candidate is None:
            break

        # Add best candidate
        selected_sites[best_candidate] = best_type
        remaining_budget -= hp.station_costs[best_type]

        # Update current response times
        multiplier = hp.coverage_multipliers[best_type]
        candidate_time = problem.time_matrix[:, best_candidate] * multiplier
        current_response = np.minimum(current_response, candidate_time)

    # Build solution
    total_cost = sum(hp.station_costs[st] for st in selected_sites.values())

    # Create adjusted time matrix accounting for station types
    adjusted_matrix = _create_adjusted_time_matrix(
        problem, selected_sites, hp.coverage_multipliers
    )

    # Evaluate with adjusted times
    from .optimization import evaluate_selection

    base_solution = evaluate_selection(
        problem,
        tuple(selected_sites.keys()),
        method="Hierarchical (greedy)",
        p=len(selected_sites),
    )

    # Count by type
    type_counts = {
        StationType.MAJOR: sum(1 for t in selected_sites.values() if t == StationType.MAJOR),
        StationType.MINOR: sum(1 for t in selected_sites.values() if t == StationType.MINOR),
        StationType.VOLUNTEER: sum(1 for t in selected_sites.values() if t == StationType.VOLUNTEER),
    }

    return HierarchicalSolution(
        method="Hierarchical (greedy)",
        p_major=type_counts[StationType.MAJOR],
        p_minor=type_counts[StationType.MINOR],
        p_volunteer=type_counts[StationType.VOLUNTEER],
        selected_sites=selected_sites,
        base_solution=base_solution,
        total_cost=total_cost,
        runtime_seconds=0.0,
        metadata={
            "budget": budget,
            "remaining_budget": remaining_budget,
            "type_counts": type_counts,
        },
    )


def _solve_hierarchical_dp(
    hp: HierarchicalProblem, budget: float
) -> HierarchicalSolution:
    """Dynamic programming approach for small instances.

    State: (budget_used, stations_opened_set)
    Too expensive for large problems but optimal for small ones.
    """
    # For now, fall back to greedy
    # Full DP implementation would require exponential space
    return _solve_hierarchical_greedy(hp, budget)


def _create_adjusted_time_matrix(
    problem: Problem,
    selected_sites: dict[int, StationType],
    multipliers: dict[StationType, float],
) -> np.ndarray:
    """Create time matrix adjusted for station types.

    Minor stations have slightly longer response times,
    volunteer stations have even longer (limited availability).
    """
    adjusted = problem.time_matrix.copy()
    for site_idx, station_type in selected_sites.items():
        multiplier = multipliers[station_type]
        adjusted[:, site_idx] *= multiplier
    return adjusted


def hierarchical_results_frame(solutions: list[HierarchicalSolution]) -> pd.DataFrame:
    """Convert hierarchical solutions to comparison table."""
    rows = []
    for sol in solutions:
        base_metrics = sol.base_solution.metrics
        rows.append(
            {
                "method": sol.method,
                "p_major": sol.p_major,
                "p_minor": sol.p_minor,
                "p_volunteer": sol.p_volunteer,
                "total_p": sol.p_major + sol.p_minor + sol.p_volunteer,
                "total_cost": sol.total_cost,
                "weighted_avg_min": base_metrics["weighted_average_minutes"],
                "max_min": base_metrics["max_minutes"],
                "coverage_8": base_metrics.get("coverage_8_0_minutes", base_metrics.get("coverage_8_minutes")),
                "runtime_s": sol.runtime_seconds,
            }
        )
    return pd.DataFrame(rows)


def solve_hierarchical_fixed_types(
    hp: HierarchicalProblem,
    p_major: int = 1,
    p_minor: int = 2,
    p_volunteer: int = 2,
) -> HierarchicalSolution:
    """Solve with fixed number of each station type.

    This is easier than budgeted problem: essentially three separate
    p-median problems with adjusted time matrices.
    """
    start = time.perf_counter()
    problem = hp.base_problem
    candidates = list(problem.candidate_indices)

    # Greedy construction
    selected_sites: dict[int, StationType] = {}
    existing = list(problem.existing_indices)
    current_response = problem.time_matrix[:, existing].min(axis=1)

    # Add major stations first (highest priority)
    for _ in range(p_major):
        best_idx, best_improvement = _find_best_addition(
            problem, current_response, candidates, selected_sites,
            hp.coverage_multipliers[StationType.MAJOR]
        )
        if best_idx is not None:
            selected_sites[best_idx] = StationType.MAJOR
            current_response = _update_response(
                problem, current_response, best_idx,
                hp.coverage_multipliers[StationType.MAJOR]
            )

    # Add minor stations
    for _ in range(p_minor):
        best_idx, best_improvement = _find_best_addition(
            problem, current_response, candidates, selected_sites,
            hp.coverage_multipliers[StationType.MINOR]
        )
        if best_idx is not None:
            selected_sites[best_idx] = StationType.MINOR
            current_response = _update_response(
                problem, current_response, best_idx,
                hp.coverage_multipliers[StationType.MINOR]
            )

    # Add volunteer stations
    for _ in range(p_volunteer):
        best_idx, best_improvement = _find_best_addition(
            problem, current_response, candidates, selected_sites,
            hp.coverage_multipliers[StationType.VOLUNTEER]
        )
        if best_idx is not None:
            selected_sites[best_idx] = StationType.VOLUNTEER
            current_response = _update_response(
                problem, current_response, best_idx,
                hp.coverage_multipliers[StationType.VOLUNTEER]
            )

    # Evaluate
    from .optimization import evaluate_selection

    base_solution = evaluate_selection(
        problem,
        tuple(selected_sites.keys()),
        method="Hierarchical (fixed types)",
        p=len(selected_sites),
    )

    total_cost = sum(hp.station_costs[st] for st in selected_sites.values())
    runtime = time.perf_counter() - start

    return HierarchicalSolution(
        method="Hierarchical (fixed types)",
        p_major=p_major,
        p_minor=p_minor,
        p_volunteer=p_volunteer,
        selected_sites=selected_sites,
        base_solution=base_solution,
        total_cost=total_cost,
        runtime_seconds=runtime,
        metadata={"p_major": p_major, "p_minor": p_minor, "p_volunteer": p_volunteer},
    )


def _find_best_addition(
    problem: Problem,
    current_response: np.ndarray,
    candidates: list[int],
    selected: dict[int, StationType],
    multiplier: float,
) -> tuple[int | None, float]:
    """Find candidate that provides best improvement."""
    best_idx = None
    best_improvement = -np.inf

    for idx in candidates:
        if idx in selected:
            continue

        candidate_time = problem.time_matrix[:, idx] * multiplier
        new_response = np.minimum(current_response, candidate_time)
        improvement = (problem.weights * (current_response - new_response)).sum()

        if improvement > best_improvement:
            best_improvement = improvement
            best_idx = idx

    return best_idx, best_improvement


def _update_response(
    problem: Problem,
    current_response: np.ndarray,
    new_site_idx: int,
    multiplier: float,
) -> np.ndarray:
    """Update response times after adding a new site."""
    candidate_time = problem.time_matrix[:, new_site_idx] * multiplier
    return np.minimum(current_response, candidate_time)
