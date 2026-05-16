"""
Complete Example: Using All Three Advanced Methods Together
============================================================

This example demonstrates how to combine Lagrangian Relaxation, Hierarchical
Optimization, and ML-Enhanced methods for a comprehensive analysis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt

from istanbul_fire_opt import load_project_data, build_problem
from istanbul_fire_opt.optimization import solve_p_median_milp, solution_table
from istanbul_fire_opt.heuristics import genetic_algorithm, simulated_annealing

# Import advanced methods
from istanbul_fire_opt.lagrangian import solve_lagrangian_relaxation
from istanbul_fire_opt.hierarchical import (
    create_hierarchical_problem,
    solve_hierarchical_budgeted,
    hierarchical_results_frame,
)
from istanbul_fire_opt.ml_optimization import (
    ml_guided_genetic_algorithm,
    feature_importance_analysis,
)


def comprehensive_analysis(p=3, budget=20.0):
    """Run comprehensive analysis using all methods."""

    print("=" * 80)
    print("COMPREHENSIVE OPTIMIZATION ANALYSIS")
    print("Istanbul Fire Station Expansion - Advanced Methods Integration")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    data = load_project_data()
    problem = build_problem(data, epsilon=0.02)
    print(f"✅ Loaded {len(problem.demand)} demand zones, {len(problem.candidate_sites)} candidates")

    # Part 1: Baseline Methods
    print("\n" + "=" * 80)
    print("[2/6] BASELINE METHODS (MILP, GA, SA)")
    print("=" * 80)

    baseline_solutions = []

    print(f"\nSolving standard p-median (p={p})...")
    milp_sol = solve_p_median_milp(problem, p)
    baseline_solutions.append(milp_sol)
    print(f"  MILP: {milp_sol.metrics['weighted_average_minutes']:.2f} min ({milp_sol.runtime_seconds:.2f}s)")

    ga_sol = genetic_algorithm(problem, p, generations=100)
    baseline_solutions.append(ga_sol)
    print(f"  GA: {ga_sol.metrics['weighted_average_minutes']:.2f} min ({ga_sol.runtime_seconds:.2f}s)")

    sa_sol = simulated_annealing(problem, p, iterations=2000)
    baseline_solutions.append(sa_sol)
    print(f"  SA: {sa_sol.metrics['weighted_average_minutes']:.2f} min ({sa_sol.runtime_seconds:.2f}s)")

    baseline_table = solution_table(problem, baseline_solutions)
    print("\nBaseline Comparison:")
    print(baseline_table[["method", "weighted_avg_min", "max_min", "coverage_8", "runtime_s"]].to_string(index=False))

    # Part 2: Lagrangian Relaxation
    print("\n" + "=" * 80)
    print("[3/6] LAGRANGIAN RELAXATION")
    print("=" * 80)

    print(f"\nSolving with Lagrangian Relaxation (p={p})...")
    lagr_result = solve_lagrangian_relaxation(problem, p, max_iterations=200)

    print(f"\n📊 Lagrangian Results:")
    print(f"  Lower Bound: {lagr_result.lower_bound:.2f} minutes")
    print(f"  Upper Bound: {lagr_result.upper_bound:.2f} minutes")
    print(f"  Optimality Gap: {lagr_result.gap_percent:.2f}%")
    print(f"  Iterations: {lagr_result.iterations}")
    print(f"  Runtime: {lagr_result.runtime_seconds:.2f}s")

    if lagr_result.gap_percent < 5.0:
        print(f"  ✅ Excellent quality: within {lagr_result.gap_percent:.2f}% of optimal!")
    elif lagr_result.gap_percent < 10.0:
        print(f"  ✅ Good quality: within {lagr_result.gap_percent:.2f}% of optimal")
    else:
        print(f"  ⚠️  Gap is {lagr_result.gap_percent:.2f}% - may need more iterations")

    # Compare with MILP
    milp_value = milp_sol.metrics["objective_total_weighted_minutes"]
    lagr_value = lagr_result.upper_bound
    difference = lagr_value - milp_value
    print(f"\n🔍 Comparison with MILP:")
    print(f"  MILP objective: {milp_value:.2f}")
    print(f"  Lagrangian feasible: {lagr_value:.2f}")
    print(f"  Difference: {difference:.2f} ({100*difference/milp_value:.2f}%)")

    # Part 3: Hierarchical Optimization
    print("\n" + "=" * 80)
    print("[4/6] HIERARCHICAL FACILITY LOCATION")
    print("=" * 80)

    hp = create_hierarchical_problem(
        problem,
        major_cost=10.0,
        minor_cost=5.0,
        volunteer_cost=1.0,
    )

    print(f"\nSolving with budget constraint ({budget} units)...")
    hier_sol = solve_hierarchical_budgeted(hp, budget=budget)

    print(f"\n🏢 Hierarchical Solution:")
    print(f"  Major stations: {hier_sol.p_major} (cost: {hier_sol.p_major * 10:.1f})")
    print(f"  Minor stations: {hier_sol.p_minor} (cost: {hier_sol.p_minor * 5:.1f})")
    print(f"  Volunteer stations: {hier_sol.p_volunteer} (cost: {hier_sol.p_volunteer * 1:.1f})")
    print(f"  Total cost: {hier_sol.total_cost:.1f} / {budget:.1f}")
    print(f"  Weighted average: {hier_sol.base_solution.metrics['weighted_average_minutes']:.2f} min")
    print(f"  Runtime: {hier_sol.runtime_seconds:.2f}s")

    # Show what stations were selected
    print(f"\n📍 Selected Stations:")
    for site_idx, station_type in list(hier_sol.selected_sites.items())[:5]:
        site_name = problem.all_sites.iloc[site_idx]["site_name"]
        print(f"  {site_name}: {station_type.value}")
    if len(hier_sol.selected_sites) > 5:
        print(f"  ... and {len(hier_sol.selected_sites) - 5} more")

    # Try different budgets
    print(f"\n💰 Budget Sensitivity:")
    budget_solutions = []
    for b in [10, 15, 20, 25, 30]:
        sol = solve_hierarchical_budgeted(hp, budget=b)
        budget_solutions.append(sol)
        print(f"  Budget {b:2.0f}: {sol.p_major}M + {sol.p_minor}m + {sol.p_volunteer}v = "
              f"{sol.base_solution.metrics['weighted_average_minutes']:.2f} min")

    # Part 4: ML-Enhanced Optimization
    print("\n" + "=" * 80)
    print("[5/6] MACHINE LEARNING ENHANCED OPTIMIZATION")
    print("=" * 80)

    try:
        print("\n🤖 Feature Importance Analysis...")
        importance_df = feature_importance_analysis(problem)
        print("\nTop Features:")
        print(importance_df.head().to_string(index=False))

        print(f"\n🧠 ML-Guided Genetic Algorithm (p={p})...")
        ml_sol = ml_guided_genetic_algorithm(problem, p, ml_method="random_forest", generations=100)

        print(f"\n📊 ML-Guided Results:")
        print(f"  ML Training: {ml_sol.warmstart_time:.2f}s")
        print(f"  Optimization: {ml_sol.optimization_time:.2f}s")
        print(f"  Total: {ml_sol.warmstart_time + ml_sol.optimization_time:.2f}s")
        print(f"  Weighted average: {ml_sol.solution.metrics['weighted_average_minutes']:.2f} min")
        print(f"  Model score (R²): {ml_sol.ml_prediction.model_score:.3f}")

        # Compare with standard GA
        ga_value = ga_sol.metrics["weighted_average_minutes"]
        ml_value = ml_sol.solution.metrics["weighted_average_minutes"]
        improvement = ga_value - ml_value
        print(f"\n🔍 Comparison with Standard GA:")
        print(f"  Standard GA: {ga_value:.2f} min")
        print(f"  ML-guided GA: {ml_value:.2f} min")
        if improvement > 0:
            print(f"  ✅ Improvement: {improvement:.2f} min ({100*improvement/ga_value:.2f}%)")
        else:
            print(f"  Similar performance (difference: {abs(improvement):.2f} min)")

    except ImportError:
        print("\n⚠️  scikit-learn not available - skipping ML features")
        print("   Install with: pip install scikit-learn")
        ml_sol = None

    # Part 5: Comprehensive Comparison
    print("\n" + "=" * 80)
    print("[6/6] COMPREHENSIVE COMPARISON")
    print("=" * 80)

    comparison_data = [
        {
            "method": "MILP (baseline)",
            "weighted_avg": milp_sol.metrics["weighted_average_minutes"],
            "max_time": milp_sol.metrics["max_minutes"],
            "runtime": milp_sol.runtime_seconds,
            "special": f"Optimal for p={p}",
        },
        {
            "method": "Lagrangian",
            "weighted_avg": lagr_result.best_solution.metrics["weighted_average_minutes"],
            "max_time": lagr_result.best_solution.metrics["max_minutes"],
            "runtime": lagr_result.runtime_seconds,
            "special": f"Gap: {lagr_result.gap_percent:.2f}%",
        },
        {
            "method": "Hierarchical",
            "weighted_avg": hier_sol.base_solution.metrics["weighted_average_minutes"],
            "max_time": hier_sol.base_solution.metrics["max_minutes"],
            "runtime": hier_sol.runtime_seconds,
            "special": f"{hier_sol.p_major}M+{hier_sol.p_minor}m+{hier_sol.p_volunteer}v",
        },
    ]

    if ml_sol:
        comparison_data.append({
            "method": "ML-guided GA",
            "weighted_avg": ml_sol.solution.metrics["weighted_average_minutes"],
            "max_time": ml_sol.solution.metrics["max_minutes"],
            "runtime": ml_sol.warmstart_time + ml_sol.optimization_time,
            "special": f"R²: {ml_sol.ml_prediction.model_score:.3f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 Final Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Summary and Recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    print("\n✅ All Methods Successfully Applied:")
    print(f"  • Baseline MILP: {milp_sol.metrics['weighted_average_minutes']:.2f} min (reference)")
    print(f"  • Lagrangian: {lagr_result.gap_percent:.2f}% gap - excellent quality guarantee")
    print(f"  • Hierarchical: {hier_sol.total_cost:.1f}/{budget} budget used - practical constraints")
    if ml_sol:
        print(f"  • ML-Enhanced: R²={ml_sol.ml_prediction.model_score:.3f} - data-driven insights")

    print("\n💡 Key Insights:")
    best_method = min(comparison_data, key=lambda x: x["weighted_avg"])
    print(f"  1. Best solution: {best_method['method']} ({best_method['weighted_avg']:.2f} min)")
    print(f"  2. Lagrangian provides {lagr_result.gap_percent:.2f}% optimality guarantee")
    print(f"  3. Hierarchical enables budget planning: {hier_sol.p_major}M+{hier_sol.p_minor}m+{hier_sol.p_volunteer}v mix")
    if ml_sol and importance_df is not None:
        top_feature = importance_df.iloc[0]
        print(f"  4. ML shows {top_feature['feature']} is most important ({top_feature['importance_percent']:.1f}%)")

    print("\n🎯 Recommendations:")
    print("  • Use Lagrangian for quality guarantees in reports")
    print("  • Use Hierarchical for budget-constrained planning")
    print("  • Use ML-Enhanced for future demand scenarios")
    print("  • Combine all three for comprehensive decision support")

    print("\n" + "=" * 80)
    print("Analysis Complete! ✨")
    print("=" * 80)

    return {
        "baseline": baseline_solutions,
        "lagrangian": lagr_result,
        "hierarchical": hier_sol,
        "ml": ml_sol,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    results = comprehensive_analysis(p=3, budget=20.0)
    print("\n✅ All advanced methods demonstrated successfully!")
