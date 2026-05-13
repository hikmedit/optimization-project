"""High-level project workflow helpers."""

from __future__ import annotations

import pandas as pd

from .config import DEFAULT_BUDGETS, DEFAULT_EPSILON
from .data import ProjectData, load_project_data
from .heuristics import genetic_algorithm, simulated_annealing
from .optimization import (
    Problem,
    Solution,
    evaluate_selection,
    solution_table,
    solve_equity_refinement,
    solve_p_median_milp,
    validate_solution,
)
from .travel_time import TravelTimeModel, try_make_road_network_model


def build_problem(data: ProjectData | None = None, prefer_road_network: bool = False) -> tuple[ProjectData, Problem, TravelTimeModel]:
    """Load data and construct the p-median instance."""

    data = data or load_project_data()
    travel_model = try_make_road_network_model() if prefer_road_network else TravelTimeModel()
    travel_model.fit_to_baseline(
        demand=data.districts,
        existing_sites=data.stations,
        weights=data.districts["weight"].to_numpy(dtype=float),
        target_minutes=data.arrival_minutes,
    )
    candidate_sites = _candidate_sites_from_districts(data.districts)
    existing_sites = data.stations.copy()
    common_columns = ["site_id", "site_name", "district", "district_key", "lon", "lat", "is_existing"]
    existing_sites = existing_sites[common_columns]
    all_sites = pd.concat([existing_sites, candidate_sites[common_columns]], ignore_index=True)
    time_matrix = travel_model.time_matrix_minutes(data.districts, all_sites)
    existing_indices = all_sites.index[all_sites["is_existing"]].to_numpy(dtype=int)
    candidate_indices = all_sites.index[~all_sites["is_existing"]].to_numpy(dtype=int)
    problem = Problem(
        demand=data.districts,
        existing_sites=existing_sites,
        candidate_sites=candidate_sites,
        all_sites=all_sites,
        time_matrix=time_matrix,
        weights=data.districts["weight"].to_numpy(dtype=float),
        existing_indices=existing_indices,
        candidate_indices=candidate_indices,
        travel_time_note=travel_model.note,
    )
    return data, problem, travel_model


def baseline_solution(problem: Problem) -> Solution:
    """Evaluate the current station network before opening new stations."""

    return evaluate_selection(problem, selected_site_indices=(), method="Current network baseline", p=0)


def run_budget_experiment(
    problem: Problem,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    epsilon: float = DEFAULT_EPSILON,
    include_heuristics: bool = True,
) -> tuple[pd.DataFrame, list[Solution]]:
    """Solve each budget with p-median, equity refinement, and heuristics."""

    solutions: list[Solution] = [baseline_solution(problem)]
    for p in budgets:
        pure = solve_p_median_milp(problem, p)
        equity = solve_equity_refinement(problem, p, pure, epsilon=epsilon)
        for solution in (pure, equity):
            validate_solution(problem, solution)
            solutions.append(solution)
        if include_heuristics:
            bound = float(pure.metrics["weighted_average_minutes"]) * (1.0 + epsilon)
            ga = genetic_algorithm(problem, p, equity_bound_minutes=bound, epsilon=epsilon)
            sa = simulated_annealing(problem, p, equity_bound_minutes=bound, epsilon=epsilon)
            for solution in (ga, sa):
                validate_solution(problem, solution)
                solutions.append(solution)
    return solution_table(problem, solutions), solutions


def assignment_frame(problem: Problem, solution: Solution) -> pd.DataFrame:
    """Demand assignments for map and table outputs."""

    demand = problem.demand.copy()
    assigned = problem.all_sites.iloc[solution.assignment_site_indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "district": demand["district"],
            "population": demand["population"],
            "demand_lon": demand["lon"],
            "demand_lat": demand["lat"],
            "assigned_site": assigned["site_name"],
            "assigned_site_district": assigned["district"],
            "response_minutes": solution.response_times,
        }
    )


def selected_sites_frame(problem: Problem, solution: Solution) -> pd.DataFrame:
    """Opened candidate station sites."""

    if not solution.selected_site_indices:
        return pd.DataFrame(columns=problem.all_sites.columns)
    return problem.all_sites.iloc[list(solution.selected_site_indices)].reset_index(drop=True)


def _candidate_sites_from_districts(districts: pd.DataFrame) -> pd.DataFrame:
    candidates = districts.copy()
    return pd.DataFrame(
        {
            "site_id": [f"candidate_{idx:02d}" for idx in range(len(candidates))],
            "site_name": "Candidate station - " + candidates["district"].astype(str),
            "district": candidates["district"],
            "district_key": candidates["district_key"],
            "lon": candidates["lon"],
            "lat": candidates["lat"],
            "is_existing": False,
        }
    )

