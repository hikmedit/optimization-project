"""Generate tables and figures used by the IEEE report."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from istanbul_fire_opt.continuous import continuous_results_frame, refine_site_location
from istanbul_fire_opt.optimization import bisection_min_budget, equity_sensitivity
from istanbul_fire_opt.visualization import plot_coverage, plot_method_comparison, plot_solution_map
from istanbul_fire_opt.workflow import build_problem, run_budget_experiment


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed"
    figure_dir = PROJECT_ROOT / "report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, problem, travel_model = build_problem()
    results, solutions = run_budget_experiment(problem, budgets=(1, 3, 5), include_heuristics=True)
    results.to_csv(output_dir / "experiment_results.csv", index=False)
    _write_latex_table(results, PROJECT_ROOT / "report" / "results_table.tex")

    final_solution = next(solution for solution in solutions if solution.method == "Equity-refined p-median" and solution.p == 5)
    plot_solution_map(problem, final_solution, title="Equity-refined recommendation, p=5", save_path=figure_dir / "final_solution_map.png")
    plot_method_comparison(results, save_path=figure_dir / "method_comparison.png")
    plot_coverage(results, save_path=figure_dir / "coverage_comparison.png")

    if final_solution.selected_site_indices:
        continuous_results = refine_site_location(
            problem,
            final_solution,
            final_solution.selected_site_indices[0],
            travel_model,
            data.neighborhoods,
        )
        continuous_results_frame(continuous_results).to_csv(output_dir / "continuous_refinement.csv", index=False)

    sensitivity = equity_sensitivity(problem, p=5)
    sensitivity.to_csv(output_dir / "equity_sensitivity.csv", index=False)
    _write_sensitivity_table(sensitivity, PROJECT_ROOT / "report" / "sensitivity_table.tex")
    feasible_p, bisection = bisection_min_budget(problem, max_p=5, target_coverage=0.70, threshold_minutes=8.0)
    bisection["feasible_p_for_70pct_8min"] = feasible_p if feasible_p is not None else ""
    bisection.to_csv(output_dir / "budget_bisection.csv", index=False)
    print("Wrote report assets to", output_dir, "and", figure_dir)


def _write_latex_table(results, path: Path) -> None:
    subset = results[
        (results["method"].isin(["Current network baseline", "MILP exact fallback (enumeration)", "Equity-refined p-median"]))
        & (results["p"].isin([0, 1, 3, 5]))
    ].copy()
    subset["method"] = subset["method"].replace({"MILP exact fallback (enumeration)": "P-median"})
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Method & $p$ & Avg & P90 & Max & Cov. $\\leq 8$ min\\\\",
        "\\hline",
    ]
    for row in subset.itertuples():
        lines.append(
            f"{_latex_escape(row.method)} & {row.p} & {row.weighted_avg_min:.2f} & "
            f"{row.p90_min:.2f} & {row.max_min:.2f} & {100 * row.coverage_8:.1f}\\%\\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sensitivity_table(sensitivity, path: Path) -> None:
    subset = sensitivity[sensitivity["epsilon"].isin([0.02, 0.05, 0.10])].copy()
    lines = [
        "\\begin{tabular}{rrrr}",
        "\\hline",
        "$\\epsilon$ & Avg & P90 & Max\\\\",
        "\\hline",
    ]
    for row in subset.itertuples():
        lines.append(f"{100 * row.epsilon:.0f}\\% & {row.weighted_average_min:.2f} & {row.p90_min:.2f} & {row.max_min:.2f}\\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    return str(value).replace("&", "\\&")


if __name__ == "__main__":
    main()
