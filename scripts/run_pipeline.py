"""Run the main optimization pipeline from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from istanbul_fire_opt.workflow import build_problem, run_budget_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Istanbul fire station expansion experiments.")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 3, 5], help="New-station budgets to evaluate.")
    parser.add_argument("--no-heuristics", action="store_true", help="Skip GA and SA experiments.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Average-time tolerance for equity refinement.")
    args = parser.parse_args()

    data, problem, travel_model = build_problem()
    print("Loaded data:", data.metadata)
    print("Travel-time model:", travel_model.mode, travel_model.note)
    results, _ = run_budget_experiment(
        problem,
        budgets=tuple(args.budgets),
        epsilon=args.epsilon,
        include_heuristics=not args.no_heuristics,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

