"""Demonstration script for advanced optimization methods.

This script demonstrates the three new optimization techniques:
1. Lagrangian Relaxation
2. Hierarchical Facility Location
3. Machine Learning Enhanced Optimization
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from istanbul_fire_opt import load_project_data, build_problem
from istanbul_fire_opt.lagrangian import (
    solve_lagrangian_relaxation,
    lagrangian_with_local_search,
)
from istanbul_fire_opt.hierarchical import (
    create_hierarchical_problem,
    solve_hierarchical_budgeted,
    solve_hierarchical_fixed_types,
    hierarchical_results_frame,
    StationType,
)
from istanbul_fire_opt.ml_optimization import (
    predict_demand_weights,
    ml_guided_genetic_algorithm,
    feature_importance_analysis,
    ml_comparison_experiment,
)
from istanbul_fire_opt.optimization import solve_p_median_milp, solution_table


def demo_lagrangian_relaxation(problem, p=3):
    """Demonstrate Lagrangian Relaxation."""
    print("=" * 70)
    print("LAGRANGIAN RELAXATION DEMO")
    print("=" * 70)

    # Solve with Lagrangian relaxation
    print(f"\n1. Solving p-median with p={p} using Lagrangian Relaxation...")
    lagr_result = solve_lagrangian_relaxation(problem, p, max_iterations=150)

    print(f"\nResults:")
    print(f"  Lower Bound: {lagr_result.lower_bound:.2f} minutes")
    print(f"  Upper Bound (Best Feasible): {lagr_result.upper_bound:.2f} minutes")
    print(f"  Optimality Gap: {lagr_result.gap_percent:.2f}%")
    print(f"  Iterations: {lagr_result.iterations}")
    print(f"  Runtime: {lagr_result.runtime_seconds:.2f}s")

    solution = lagr_result.best_solution
    print(f"\nBest Solution:")
    print(f"  Weighted Average: {solution.metrics['weighted_average_minutes']:.2f} min")
    print(f"  Max Response Time: {solution.metrics['max_minutes']:.2f} min")
    print(f"  Coverage (8 min): {solution.metrics.get('coverage_8_0_minutes', 0):.2%}")

    # Compare with standard MILP
    print(f"\n2. Comparing with standard MILP...")
    milp_solution = solve_p_median_milp(problem, p)

    comparison = solution_table(problem, [lagr_result.best_solution, milp_solution])
    print("\nComparison Table:")
    print(comparison.to_string(index=False))

    # Lagrangian with local search
    print(f"\n3. Lagrangian with Local Search Refinement...")
    lagr_ls_solution = lagrangian_with_local_search(problem, p, max_iterations=150)
    print(f"  Weighted Average: {lagr_ls_solution.metrics['weighted_average_minutes']:.2f} min")
    print(f"  Runtime: {lagr_ls_solution.runtime_seconds:.2f}s")

    return lagr_result, milp_solution, lagr_ls_solution


def demo_hierarchical_optimization(problem, budget=20.0):
    """Demonstrate Hierarchical Facility Location."""
    print("\n" + "=" * 70)
    print("HIERARCHICAL FACILITY LOCATION DEMO")
    print("=" * 70)

    # Create hierarchical problem
    print(f"\n1. Creating hierarchical problem...")
    print(f"  Station types: Major (cost=10), Minor (cost=5), Volunteer (cost=1)")
    print(f"  Total budget: {budget} units")

    hp = create_hierarchical_problem(
        problem,
        major_cost=10.0,
        minor_cost=5.0,
        volunteer_cost=1.0,
    )

    # Solve with budget constraint
    print(f"\n2. Solving with budget constraint...")
    budget_solution = solve_hierarchical_budgeted(hp, budget=budget, method="greedy")

    print(f"\nBudget-Constrained Solution:")
    print(f"  Major stations: {budget_solution.p_major}")
    print(f"  Minor stations: {budget_solution.p_minor}")
    print(f"  Volunteer stations: {budget_solution.p_volunteer}")
    print(f"  Total cost: {budget_solution.total_cost:.1f}")
    print(f"  Weighted Average: {budget_solution.base_solution.metrics['weighted_average_minutes']:.2f} min")
    print(f"  Runtime: {budget_solution.runtime_seconds:.2f}s")

    # Solve with fixed types
    print(f"\n3. Solving with fixed station types (1 major, 2 minor, 2 volunteer)...")
    fixed_solution = solve_hierarchical_fixed_types(hp, p_major=1, p_minor=2, p_volunteer=2)

    print(f"\nFixed-Type Solution:")
    print(f"  Total cost: {fixed_solution.total_cost:.1f}")
    print(f"  Weighted Average: {fixed_solution.base_solution.metrics['weighted_average_minutes']:.2f} min")

    # Show station assignments
    print(f"\n4. Station Type Assignments:")
    for site_idx, station_type in list(budget_solution.selected_sites.items())[:5]:
        site_name = problem.all_sites.iloc[site_idx]["site_name"]
        print(f"  {site_name}: {station_type.value}")

    # Compare multiple budget levels
    print(f"\n5. Comparing different budget levels...")
    solutions = []
    for b in [10, 15, 20, 25]:
        sol = solve_hierarchical_budgeted(hp, budget=b, method="greedy")
        solutions.append(sol)

    comparison = hierarchical_results_frame(solutions)
    print("\nBudget Comparison:")
    print(comparison.to_string(index=False))

    return budget_solution, fixed_solution


def demo_ml_optimization(problem, p=3):
    """Demonstrate ML-Enhanced Optimization."""
    print("\n" + "=" * 70)
    print("MACHINE LEARNING ENHANCED OPTIMIZATION DEMO")
    print("=" * 70)

    # Feature importance analysis
    print(f"\n1. Analyzing feature importance for demand prediction...")
    try:
        importance_df = feature_importance_analysis(problem)
        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))
    except Exception as e:
        print(f"  Note: Feature importance analysis requires scikit-learn: {e}")
        importance_df = None

    # Demand prediction
    print(f"\n2. Predicting demand weights with ML...")
    try:
        ml_pred = predict_demand_weights(problem, method="random_forest")
        print(f"  Method: {ml_pred.method}")
        print(f"  Model Score (R²): {ml_pred.model_score:.3f}")
        print(f"  Original total demand: {problem.weights.sum():.0f}")
        print(f"  Predicted total demand: {ml_pred.predicted_demand.sum():.0f}")

        # Show top 5 districts by prediction change
        demand_df = problem.demand.copy()
        demand_df["original_weight"] = problem.weights
        demand_df["predicted_weight"] = ml_pred.predicted_demand
        demand_df["change_percent"] = (
            100 * (demand_df["predicted_weight"] - demand_df["original_weight"])
            / demand_df["original_weight"]
        )
        demand_df = demand_df.sort_values("change_percent", ascending=False)

        print("\nTop 5 districts by prediction change:")
        print(demand_df[["district", "original_weight", "predicted_weight", "change_percent"]].head().to_string(index=False))

    except Exception as e:
        print(f"  Note: ML prediction requires scikit-learn: {e}")
        ml_pred = None

    # ML-guided optimization
    print(f"\n3. Running ML-guided Genetic Algorithm...")
    try:
        ml_ga_solution = ml_guided_genetic_algorithm(
            problem, p, ml_method="random_forest", generations=100
        )

        print(f"\nML-Guided GA Results:")
        print(f"  Warmstart Time: {ml_ga_solution.warmstart_time:.2f}s")
        print(f"  Optimization Time: {ml_ga_solution.optimization_time:.2f}s")
        print(f"  Total Time: {ml_ga_solution.warmstart_time + ml_ga_solution.optimization_time:.2f}s")
        print(f"  Weighted Average: {ml_ga_solution.solution.metrics['weighted_average_minutes']:.2f} min")
        print(f"  Max Response: {ml_ga_solution.solution.metrics['max_minutes']:.2f} min")

        # Compare with standard GA
        print(f"\n4. Comparing ML-guided vs Standard methods...")
        comparison_df = ml_comparison_experiment(problem, p=p)
        print("\nMethod Comparison:")
        print(comparison_df.to_string(index=False))

        return ml_ga_solution, comparison_df

    except Exception as e:
        print(f"  Note: ML-guided optimization requires scikit-learn: {e}")
        return None, None


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("ADVANCED OPTIMIZATION METHODS DEMONSTRATION")
    print("Istanbul Fire Station Expansion Project")
    print("=" * 70)

    # Load data
    print("\nLoading project data...")
    data = load_project_data()
    print(f"  Existing stations: {len(data['existing_stations'])}")
    print(f"  Districts: {len(data['districts'])}")

    # Build problem
    print("\nBuilding optimization problem...")
    problem = build_problem(data, epsilon=0.02)
    print(f"  Demand zones: {len(problem.demand)}")
    print(f"  Candidate sites: {len(problem.candidate_sites)}")

    # Demo 1: Lagrangian Relaxation
    lagr_result, milp_solution, lagr_ls = demo_lagrangian_relaxation(problem, p=3)

    # Demo 2: Hierarchical Optimization
    budget_sol, fixed_sol = demo_hierarchical_optimization(problem, budget=20.0)

    # Demo 3: ML Optimization
    ml_ga_sol, ml_comparison = demo_ml_optimization(problem, p=3)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ADVANCED METHODS")
    print("=" * 70)

    print("\n1. Lagrangian Relaxation:")
    print(f"   - Provides optimality bounds (gap: {lagr_result.gap_percent:.2f}%)")
    print(f"   - Fast convergence ({lagr_result.iterations} iterations)")
    print(f"   - Can be combined with local search for improvement")

    print("\n2. Hierarchical Facility Location:")
    print(f"   - Models different station types with varying costs")
    print(f"   - Budget-constrained optimization")
    print(f"   - More realistic for planning diverse facility networks")

    print("\n3. ML-Enhanced Optimization:")
    print(f"   - Predicts future demand patterns")
    print(f"   - Identifies important features for site selection")
    print(f"   - Can warmstart heuristics for faster convergence")

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
