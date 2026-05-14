"""Core unit tests for optimization logic."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from istanbul_fire_opt.optimization import (
    Problem,
    solve_equity_refinement,
    solve_p_median_exact,
    validate_solution,
)


def toy_problem() -> Problem:
    demand = pd.DataFrame(
        {
            "district": ["A", "B", "C"],
            "population": [100, 100, 10],
            "weight": [100.0, 100.0, 10.0],
            "lon": [0.0, 1.0, 2.0],
            "lat": [0.0, 0.0, 0.0],
        }
    )
    existing = pd.DataFrame(
        {
            "site_id": ["existing_0"],
            "site_name": ["Existing A"],
            "district": ["A"],
            "district_key": ["a"],
            "lon": [0.0],
            "lat": [0.0],
            "is_existing": [True],
        }
    )
    candidates = pd.DataFrame(
        {
            "site_id": ["candidate_b", "candidate_c"],
            "site_name": ["Candidate B", "Candidate C"],
            "district": ["B", "C"],
            "district_key": ["b", "c"],
            "lon": [1.0, 2.0],
            "lat": [0.0, 0.0],
            "is_existing": [False, False],
        }
    )
    all_sites = pd.concat([existing, candidates], ignore_index=True)
    time_matrix = np.array(
        [
            [0.0, 2.0, 9.0],
            [7.0, 0.0, 4.0],
            [20.0, 8.0, 0.0],
        ]
    )
    return Problem(
        demand=demand,
        existing_sites=existing,
        candidate_sites=candidates,
        all_sites=all_sites,
        time_matrix=time_matrix,
        weights=demand["weight"].to_numpy(),
        existing_indices=np.array([0]),
        candidate_indices=np.array([1, 2]),
        travel_time_note="toy",
    )


class OptimizationTests(unittest.TestCase):
    def test_exact_p_median_opens_weighted_best_candidate(self) -> None:
        problem = toy_problem()
        solution = solve_p_median_exact(problem, p=1)
        validate_solution(problem, solution)
        self.assertEqual(solution.selected_site_indices, (1,))

    def test_equity_can_trade_average_for_lower_worst_case(self) -> None:
        problem = toy_problem()
        pure = solve_p_median_exact(problem, p=1)
        equity = solve_equity_refinement(problem, p=1, p_median_solution=pure, epsilon=5.0)
        validate_solution(problem, equity)
        self.assertEqual(equity.selected_site_indices, (2,))
        self.assertLess(equity.metrics["max_minutes"], pure.metrics["max_minutes"])


if __name__ == "__main__":
    unittest.main()
