"""Tests for advanced optimization methods."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from istanbul_fire_opt import load_project_data, build_problem
from istanbul_fire_opt.lagrangian import (
    solve_lagrangian_relaxation,
    lagrangian_with_local_search,
)
from istanbul_fire_opt.hierarchical import (
    create_hierarchical_problem,
    solve_hierarchical_budgeted,
    solve_hierarchical_fixed_types,
    StationType,
)
from istanbul_fire_opt.ml_optimization import (
    predict_demand_weights,
    ml_guided_genetic_algorithm,
    feature_importance_analysis,
)


@pytest.fixture
def problem():
    """Load test problem."""
    data = load_project_data()
    return build_problem(data, epsilon=0.02)


class TestLagrangianRelaxation:
    """Tests for Lagrangian Relaxation."""

    def test_lagrangian_basic(self, problem):
        """Test basic Lagrangian relaxation."""
        result = solve_lagrangian_relaxation(problem, p=3, max_iterations=50)

        # Check result structure
        assert result.lower_bound > 0
        assert result.upper_bound > 0
        assert result.lower_bound <= result.upper_bound
        assert result.best_solution is not None
        assert len(result.best_solution.selected_site_indices) == 3
        assert result.iterations <= 50
        assert result.runtime_seconds > 0

    def test_lagrangian_bounds_quality(self, problem):
        """Test that Lagrangian bounds are meaningful."""
        result = solve_lagrangian_relaxation(problem, p=3, max_iterations=100)

        # Gap should be reasonable (< 50%)
        assert result.gap_percent < 50.0

        # Lower bound should be less than upper bound
        assert result.lower_bound < result.upper_bound

    def test_lagrangian_with_local_search(self, problem):
        """Test Lagrangian with local search refinement."""
        solution = lagrangian_with_local_search(problem, p=3, max_iterations=50)

        assert solution is not None
        assert len(solution.selected_site_indices) == 3
        assert solution.metrics["weighted_average_minutes"] > 0
        assert "lagrangian_lower_bound" in solution.metadata

    def test_lagrangian_different_p_values(self, problem):
        """Test Lagrangian with different p values."""
        for p in [1, 2, 3]:
            result = solve_lagrangian_relaxation(problem, p=p, max_iterations=30)
            assert len(result.best_solution.selected_site_indices) == p


class TestHierarchicalOptimization:
    """Tests for Hierarchical Facility Location."""

    def test_create_hierarchical_problem(self, problem):
        """Test hierarchical problem creation."""
        hp = create_hierarchical_problem(problem)

        assert hp.base_problem == problem
        assert len(hp.station_costs) == 3
        assert len(hp.station_capacities) == 3
        assert len(hp.coverage_multipliers) == 3
        assert StationType.MAJOR in hp.station_costs

    def test_hierarchical_budgeted(self, problem):
        """Test budget-constrained hierarchical optimization."""
        hp = create_hierarchical_problem(problem)
        solution = solve_hierarchical_budgeted(hp, budget=20.0, method="greedy")

        # Check solution structure
        assert solution.total_cost <= 20.0
        assert solution.p_major >= 0
        assert solution.p_minor >= 0
        assert solution.p_volunteer >= 0
        assert len(solution.selected_sites) > 0
        assert solution.base_solution is not None

    def test_hierarchical_fixed_types(self, problem):
        """Test fixed-type hierarchical optimization."""
        hp = create_hierarchical_problem(problem)
        solution = solve_hierarchical_fixed_types(
            hp, p_major=1, p_minor=2, p_volunteer=2
        )

        # Check counts
        assert solution.p_major == 1
        assert solution.p_minor == 2
        assert solution.p_volunteer == 2
        assert len(solution.selected_sites) == 5

        # Check types are assigned
        types = list(solution.selected_sites.values())
        assert types.count(StationType.MAJOR) == 1
        assert types.count(StationType.MINOR) == 2
        assert types.count(StationType.VOLUNTEER) == 2

    def test_hierarchical_different_budgets(self, problem):
        """Test hierarchical optimization with different budgets."""
        hp = create_hierarchical_problem(problem)

        budgets = [10.0, 20.0, 30.0]
        costs = []

        for budget in budgets:
            solution = solve_hierarchical_budgeted(hp, budget=budget)
            costs.append(solution.total_cost)

        # Higher budget should allow more/better stations
        assert costs[0] <= costs[1] <= costs[2]


class TestMLOptimization:
    """Tests for ML-Enhanced Optimization."""

    def test_predict_demand_population(self, problem):
        """Test population-based demand prediction."""
        result = predict_demand_weights(problem, method="population_density")

        assert result.predicted_demand is not None
        assert len(result.predicted_demand) == len(problem.demand)
        assert np.all(result.predicted_demand >= 0)
        assert result.method == "population_density"

    def test_predict_demand_random_forest(self, problem):
        """Test Random Forest demand prediction."""
        try:
            result = predict_demand_weights(problem, method="random_forest")

            assert result.predicted_demand is not None
            assert len(result.predicted_demand) == len(problem.demand)
            assert result.method == "random_forest"
            assert "population" in result.feature_importance

        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_feature_importance_analysis(self, problem):
        """Test feature importance analysis."""
        try:
            importance_df = feature_importance_analysis(problem)

            assert len(importance_df) > 0
            assert "feature" in importance_df.columns
            assert "importance" in importance_df.columns
            assert "importance_percent" in importance_df.columns

        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_ml_guided_ga(self, problem):
        """Test ML-guided genetic algorithm."""
        try:
            result = ml_guided_genetic_algorithm(
                problem, p=3, ml_method="population_density", generations=30
            )

            assert result.solution is not None
            assert len(result.solution.selected_site_indices) == 3
            assert result.warmstart_time >= 0
            assert result.optimization_time > 0
            assert result.ml_prediction is not None

        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_compare_all_methods(self, problem):
        """Test that all methods produce valid solutions for same p."""
        p = 3

        # Lagrangian
        lagr_result = solve_lagrangian_relaxation(problem, p, max_iterations=30)
        lagr_value = lagr_result.best_solution.metrics["weighted_average_minutes"]

        # Hierarchical (with fixed types matching p)
        hp = create_hierarchical_problem(problem)
        hier_solution = solve_hierarchical_fixed_types(hp, p_major=1, p_minor=1, p_volunteer=1)
        hier_value = hier_solution.base_solution.metrics["weighted_average_minutes"]

        # ML-guided (fallback to population if sklearn not available)
        try:
            ml_result = ml_guided_genetic_algorithm(
                problem, p, ml_method="population_density", generations=30
            )
            ml_value = ml_result.solution.metrics["weighted_average_minutes"]
        except ImportError:
            ml_value = None

        # All should produce reasonable values
        assert lagr_value > 0
        assert hier_value > 0
        if ml_value is not None:
            assert ml_value > 0

        print(f"\nMethod Comparison (p={p}):")
        print(f"  Lagrangian: {lagr_value:.2f} min")
        print(f"  Hierarchical: {hier_value:.2f} min")
        if ml_value is not None:
            print(f"  ML-Guided: {ml_value:.2f} min")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
