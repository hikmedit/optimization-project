"""Create the formatted project notebook."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "istanbul_fire_station_expansion.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(
            """
            # Optimizing Fire Station Expansion in Istanbul

            **YZV202E Optimization for Data Science**

            This notebook implements the project proposal using a reproducible location-allocation pipeline. The primary model is a fixed-existing-station weighted p-median formulation. Following the instructor feedback, the notebook also evaluates district-level service gaps and uses one consistent travel-time metric in minutes throughout the discrete and continuous stages.
            """
        ),
        md(
            """
            ## 1. Setup

            The project code is organized under `src/istanbul_fire_opt`. Optional packages such as `PuLP`, `OSMnx`, `folium`, and `ipywidgets` improve the experience, but the core notebook runs with the fallback exact enumerator and calibrated minute proxy.
            """
        ),
        code(
            """
            from pathlib import Path
            import sys

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            import pandas as pd
            import matplotlib.pyplot as plt

            from istanbul_fire_opt.workflow import (
                build_problem,
                baseline_solution,
                run_budget_experiment,
                assignment_frame,
                selected_sites_frame,
            )
            from istanbul_fire_opt.optimization import (
                bisection_min_budget,
                equity_sensitivity,
                solution_table,
                solve_equity_refinement,
                solve_p_median_milp,
                validate_solution,
            )
            from istanbul_fire_opt.heuristics import genetic_algorithm, simulated_annealing
            from istanbul_fire_opt.continuous import continuous_results_frame, refine_site_location
            from istanbul_fire_opt.visualization import plot_coverage, plot_method_comparison, plot_solution_map

            pd.set_option("display.max_colwidth", 140)
            """
        ),
        md(
            """
            ## 2. Data Preparation

            The data layer downloads and caches official IBB files:

            - Fire station coordinates.
            - District population from the IBB/TUIK population table.
            - Average fire-event arrival time for calibration.
            - First-degree emergency transportation roads.
            - Mukhtar office locations, aggregated into district centroids.

            The project uses population as the primary demand weight. The official fire-count sheets found during implementation are city-level rather than district-level, so they are used as contextual evidence instead of district weights.
            """
        ),
        code(
            """
            data, problem, travel_model = build_problem(prefer_road_network=False)
            print(data.metadata)
            print("Travel-time model:", travel_model.mode)
            print(travel_model.note)
            display(problem.demand.head())
            display(problem.all_sites.head())
            """
        ),
        md(
            """
            ## 3. Consistent Travel-Time Metric

            The objective is measured in minutes. If road-network routing is available, the same `TravelTimeModel` interface can return shortest-path road times. In this local environment, the notebook uses a calibrated geometric proxy: district-to-site distances are converted to minutes and scaled so the current network's weighted average response matches IBB's published average arrival time. The continuous refinement stage uses this same minute-scale interface, not raw Euclidean distance.
            """
        ),
        code(
            """
            current = baseline_solution(problem)
            display(solution_table(problem, [current]))
            plot_solution_map(problem, current, title="Current fire station network baseline")
            plt.show()
            """
        ),
        md(
            """
            ## 4. P-Median and Equity Refinement

            The first model minimizes total weighted response time. The second model follows the instructor feedback by reducing district-level gaps: it minimizes the worst assigned district response time while keeping total weighted response time within `epsilon = 2%` of the p-median optimum.
            """
        ),
        code(
            """
            p = 3
            epsilon = 0.02

            p_median = solve_p_median_milp(problem, p)
            equity = solve_equity_refinement(problem, p, p_median_solution=p_median, epsilon=epsilon)
            validate_solution(problem, p_median)
            validate_solution(problem, equity)

            comparison = solution_table(problem, [current, p_median, equity])
            display(comparison)
            plot_solution_map(problem, equity, title=f"Equity-refined p-median recommendation, p={p}")
            plt.show()
            display(assignment_frame(problem, equity).sort_values("response_minutes", ascending=False).head(10))
            """
        ),
        md(
            """
            ## 5. Interactive Scenario Controls

            Use the controls to change the budget, method, equity tolerance, and coverage threshold. If `ipywidgets` is not installed, the same function can still be called manually.
            """
        ),
        code(
            """
            def run_scenario(p=3, method="Equity-refined p-median", epsilon=0.02, threshold_minutes=8.0):
                base = solve_p_median_milp(problem, int(p))
                if method == "Pure p-median":
                    solution = base
                elif method == "GA equity-aware":
                    bound = base.metrics["weighted_average_minutes"] * (1 + epsilon)
                    solution = genetic_algorithm(problem, int(p), equity_bound_minutes=bound, epsilon=epsilon, generations=80)
                elif method == "SA equity-aware":
                    bound = base.metrics["weighted_average_minutes"] * (1 + epsilon)
                    solution = simulated_annealing(problem, int(p), equity_bound_minutes=bound, epsilon=epsilon, iterations=1600)
                else:
                    solution = solve_equity_refinement(problem, int(p), p_median_solution=base, epsilon=epsilon)

                table = solution_table(problem, [solution])
                coverage_key = f"coverage_{str(float(threshold_minutes)).replace('.', '_')}_minutes"
                if coverage_key in solution.metrics:
                    print(f"Weighted coverage at {threshold_minutes:g} min:", f"{solution.metrics[coverage_key]:.1%}")
                display(table)
                plot_solution_map(problem, solution, title=f"{method}, p={p}")
                plt.show()
                return solution

            try:
                import ipywidgets as widgets
                from IPython.display import display as ipy_display

                controls = widgets.interactive(
                    run_scenario,
                    p=widgets.IntSlider(value=3, min=1, max=5, step=1, description="p"),
                    method=widgets.Dropdown(
                        options=["Equity-refined p-median", "Pure p-median", "GA equity-aware", "SA equity-aware"],
                        value="Equity-refined p-median",
                        description="method",
                    ),
                    epsilon=widgets.FloatSlider(value=0.02, min=0.0, max=0.20, step=0.01, readout_format=".0%", description="epsilon"),
                    threshold_minutes=widgets.Dropdown(options=[5.0, 8.0], value=8.0, description="Tmax"),
                )
                ipy_display(controls)
            except Exception as exc:
                print("ipywidgets is unavailable; running a default scenario instead.")
                print(type(exc).__name__, exc)
                _ = run_scenario()
            """
        ),
        md(
            """
            ## 6. Extended Analysis: Exact, GA, and SA Comparison

            This experiment compares exact p-median, equity-refined p-median, Genetic Algorithm, and Simulated Annealing for the proposal budgets `p in {1, 3, 5}`. The key evaluation columns are weighted average response time, worst district response time, P90 response time, and weighted demand coverage under 5 and 8 minutes.
            """
        ),
        code(
            """
            results, solutions = run_budget_experiment(problem, budgets=(1, 3, 5), epsilon=0.02, include_heuristics=True)
            display(results)
            plot_method_comparison(results)
            plt.show()
            plot_coverage(results)
            plt.show()
            """
        ),
        md(
            """
            ## 7. Equity Sensitivity

            A strict `epsilon = 2%` average-time tolerance may not fully reduce the worst district if the worst-served area has relatively low population. This table makes that tradeoff explicit instead of hiding it inside the average objective.
            """
        ),
        code(
            """
            sensitivity = equity_sensitivity(problem, p=5, epsilons=(0.0, 0.02, 0.05, 0.10, 0.20))
            display(sensitivity)
            """
        ),
        md(
            """
            ## 8. Continuous Local Refinement

            After the discrete model chooses candidate districts, we refine one opened station locally. The fitted surrogate is a differentiable approximation of the same calibrated response-time objective in minutes. The final coordinate is snapped to an official neighborhood point in the same district and re-evaluated with the same travel-time metric.
            """
        ),
        code(
            """
            final_solution = next(sol for sol in solutions if sol.method == "Equity-refined p-median" and sol.p == 5)
            selected = selected_sites_frame(problem, final_solution)
            display(selected)

            continuous_results = refine_site_location(
                problem,
                final_solution,
                final_solution.selected_site_indices[0],
                travel_model,
                data.neighborhoods,
            )
            display(continuous_results_frame(continuous_results))
            """
        ),
        md(
            """
            ## 9. Budget Search

            Bisection over `p` estimates the minimum budget needed to reach a target coverage level. This is the budget-planning extension described in the proposal.
            """
        ),
        code(
            """
            feasible_p, budget_trace = bisection_min_budget(problem, max_p=5, target_coverage=0.70, threshold_minutes=8.0)
            print("Smallest p reaching 70% weighted coverage within 8 minutes:", feasible_p)
            display(budget_trace)
            """
        ),
        md(
            """
            ## 10. Conclusions

            The final recommendation should be based on the equity-refined p-median solution rather than the average-only p-median. The average-only model is useful as a benchmark, but the equity layer directly addresses the risk that total-response-time minimization can leave large district-level gaps. The continuous stage is reported as a local refinement using the same minute-scale travel-time model, so it does not introduce the road-time versus Euclidean-distance inconsistency noted in the proposal feedback.
            """
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print("Wrote", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
