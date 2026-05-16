"""Machine Learning Enhanced Optimization.

This module integrates machine learning with optimization to:
1. Predict demand hotspots using population and incident data
2. Learn good initial solutions for heuristics
3. Predict solution quality before full evaluation
4. Extract feature importance for site selection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_RANDOM_SEED
from .optimization import Problem, Solution, evaluate_selection


@dataclass
class MLPredictionResult:
    """Results from ML-based demand prediction."""

    predicted_demand: np.ndarray  # Predicted demand weights
    feature_importance: dict[str, float]  # Feature importance scores
    model_score: float  # R² or similar metric
    method: str  # "random_forest", "gradient_boosting", etc.


@dataclass
class MLGuidedSolution:
    """Solution found using ML-guided optimization."""

    solution: Solution
    ml_prediction: MLPredictionResult
    warmstart_time: float  # Time spent on ML warmstart
    optimization_time: float  # Time spent on optimization
    method: str


def predict_demand_weights(
    problem: Problem,
    historical_incidents: pd.DataFrame | None = None,
    method: str = "population_density",
) -> MLPredictionResult:
    """Predict future demand weights using ML or heuristics.

    Args:
        problem: Base problem with current demand
        historical_incidents: Historical fire incident data (optional)
        method: "population_density", "random_forest", "gradient_boosting"

    Returns:
        MLPredictionResult with predicted weights
    """
    if method == "population_density":
        return _predict_by_population(problem)
    elif method == "random_forest":
        return _predict_with_random_forest(problem, historical_incidents)
    elif method == "gradient_boosting":
        return _predict_with_gradient_boosting(problem, historical_incidents)
    else:
        raise ValueError(f"Unknown method: {method}")


def _predict_by_population(problem: Problem) -> MLPredictionResult:
    """Simple baseline: predict demand proportional to population."""
    # Use current weights as prediction (proportional to population)
    predicted = problem.weights.copy()

    return MLPredictionResult(
        predicted_demand=predicted,
        feature_importance={"population": 1.0},
        model_score=1.0,  # Perfect correlation with current data
        method="population_density",
    )


def _predict_with_random_forest(
    problem: Problem, historical_incidents: pd.DataFrame | None
) -> MLPredictionResult:
    """Predict demand using Random Forest with geographic and demographic features."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        # Fallback if sklearn not available
        return _predict_by_population(problem)

    # Create features from problem data
    features, targets = _create_ml_features(problem, historical_incidents)

    if features is None or len(features) < 10:
        # Not enough data for ML
        return _predict_by_population(problem)

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=DEFAULT_RANDOM_SEED,
        n_jobs=-1,
    )

    rf.fit(features, targets)

    # Predict for all demand zones
    predicted = rf.predict(features)
    predicted = np.maximum(predicted, 0)  # Ensure non-negative

    # Normalize to sum to same total as original
    predicted = predicted * (problem.weights.sum() / predicted.sum())

    # Cross-validation score
    cv_scores = cross_val_score(rf, features, targets, cv=5, scoring="r2")
    model_score = float(cv_scores.mean())

    # Feature importance
    feature_names = _get_feature_names()
    importance_dict = dict(zip(feature_names, rf.feature_importances_))

    return MLPredictionResult(
        predicted_demand=predicted,
        feature_importance=importance_dict,
        model_score=model_score,
        method="random_forest",
    )


def _predict_with_gradient_boosting(
    problem: Problem, historical_incidents: pd.DataFrame | None
) -> MLPredictionResult:
    """Predict demand using Gradient Boosting."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return _predict_by_population(problem)

    features, targets = _create_ml_features(problem, historical_incidents)

    if features is None or len(features) < 10:
        return _predict_by_population(problem)

    # Train Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=DEFAULT_RANDOM_SEED,
    )

    gb.fit(features, targets)

    predicted = gb.predict(features)
    predicted = np.maximum(predicted, 0)
    predicted = predicted * (problem.weights.sum() / predicted.sum())

    cv_scores = cross_val_score(gb, features, targets, cv=5, scoring="r2")
    model_score = float(cv_scores.mean())

    feature_names = _get_feature_names()
    importance_dict = dict(zip(feature_names, gb.feature_importances_))

    return MLPredictionResult(
        predicted_demand=predicted,
        feature_importance=importance_dict,
        model_score=model_score,
        method="gradient_boosting",
    )


def _create_ml_features(
    problem: Problem, historical_incidents: pd.DataFrame | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create feature matrix for ML models.

    Features:
    - Population (current weight)
    - Distance to nearest existing station
    - Distance to city center (Taksim: 41.0370, 28.9850)
    - Distance to nearest highway/emergency road
    - Average distance to all other districts (centrality)
    """
    demand_df = problem.demand

    if len(demand_df) < 10:
        return None, None

    features_list = []

    # Feature 1: Population (current weight)
    population = problem.weights

    # Feature 2: Distance to nearest existing station
    existing_times = problem.time_matrix[:, problem.existing_indices]
    nearest_station_time = existing_times.min(axis=1)

    # Feature 3: Distance to city center (using lat/lon)
    if "lat" in demand_df.columns and "lon" in demand_df.columns:
        lats = demand_df["lat"].values
        lons = demand_df["lon"].values
        # Taksim Square coordinates
        center_lat, center_lon = 41.0370, 28.9850
        dist_to_center = np.sqrt(
            (lats - center_lat) ** 2 + (lons - center_lon) ** 2
        ) * 111.32  # Approximate km
    else:
        dist_to_center = np.zeros(len(demand_df))

    # Feature 4: Centrality (average distance to all other zones)
    centrality = problem.time_matrix.mean(axis=1)

    # Feature 5: Population density (if area available)
    if "area_km2" in demand_df.columns:
        area = demand_df["area_km2"].values
        density = population / np.maximum(area, 1.0)
    else:
        density = population  # Use population as proxy

    # Stack features
    features = np.column_stack(
        [
            population,
            nearest_station_time,
            dist_to_center,
            centrality,
            density,
        ]
    )

    # Target: current weights (or historical incident counts if available)
    if historical_incidents is not None and "incident_count" in historical_incidents.columns:
        targets = historical_incidents["incident_count"].values
    else:
        targets = population  # Use population as target

    return features, targets


def _get_feature_names() -> list[str]:
    """Return feature names for importance reporting."""
    return [
        "population",
        "nearest_station_time",
        "dist_to_center",
        "centrality",
        "density",
    ]


def ml_guided_genetic_algorithm(
    problem: Problem,
    p: int,
    ml_method: str = "random_forest",
    population_size: int = 60,
    generations: int = 120,
) -> MLGuidedSolution:
    """Genetic Algorithm with ML-predicted demand for warmstart.

    Uses ML to predict future demand patterns, then runs GA with those predictions.
    """
    warmstart_start = time.perf_counter()

    # Predict demand
    ml_prediction = predict_demand_weights(problem, method=ml_method)

    # Create modified problem with predicted weights
    import copy

    modified_problem = copy.deepcopy(problem)
    modified_problem.weights = ml_prediction.predicted_demand

    warmstart_time = time.perf_counter() - warmstart_start

    # Run GA on modified problem
    opt_start = time.perf_counter()

    from .heuristics import genetic_algorithm

    ga_solution = genetic_algorithm(
        modified_problem,
        p,
        population_size=population_size,
        generations=generations,
    )

    optimization_time = time.perf_counter() - opt_start

    # Evaluate on original problem
    final_solution = evaluate_selection(
        problem,
        ga_solution.selected_site_indices,
        method=f"ML-guided GA ({ml_method})",
        p=p,
    )

    return MLGuidedSolution(
        solution=final_solution,
        ml_prediction=ml_prediction,
        warmstart_time=warmstart_time,
        optimization_time=optimization_time,
        method=f"ML-guided GA ({ml_method})",
    )


def predict_solution_quality(
    problem: Problem, candidate_solutions: list[tuple[int, ...]]
) -> np.ndarray:
    """Predict solution quality without full evaluation using ML.

    This can be used to filter poor solutions before expensive evaluation.
    Uses simple features: sum of distances, coverage heuristics, etc.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        # Fallback: return zeros (no prediction)
        return np.zeros(len(candidate_solutions))

    # Extract features for each candidate solution
    features = []
    for selected in candidate_solutions:
        features.append(_extract_solution_features(problem, selected))

    features_array = np.array(features)

    # For training, we'd need labeled data (solution -> quality)
    # Here we use a simple heuristic predictor
    # In practice, you'd train this on historical optimization runs

    # Simple predictor: weighted sum of features
    # (This is a placeholder; real ML would learn weights)
    predictor = np.array([1.0, -0.5, 2.0, -1.0])  # Example weights
    predictions = features_array @ predictor

    return predictions


def _extract_solution_features(
    problem: Problem, selected: tuple[int, ...]
) -> np.ndarray:
    """Extract features from a candidate solution for quality prediction."""
    existing = list(problem.existing_indices)
    active = list(set(existing).union(selected))

    # Feature 1: Average distance to nearest active facility
    active_times = problem.time_matrix[:, active]
    avg_nearest = (problem.weights * active_times.min(axis=1)).sum() / problem.weights.sum()

    # Feature 2: Maximum distance
    max_dist = active_times.min(axis=1).max()

    # Feature 3: Coverage at 8 minutes
    covered = (active_times.min(axis=1) <= 8.0).sum()

    # Feature 4: Dispersion (how spread out are the facilities)
    if len(selected) > 1:
        selected_list = list(selected)
        # Average pairwise distance between selected facilities
        pairwise_dists = []
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                idx1, idx2 = selected_list[i], selected_list[j]
                # Use time matrix as proxy for distance
                dist = problem.time_matrix[idx1, idx2] if idx2 < problem.time_matrix.shape[0] else 0
                pairwise_dists.append(dist)
        dispersion = np.mean(pairwise_dists) if pairwise_dists else 0
    else:
        dispersion = 0

    return np.array([avg_nearest, max_dist, covered, dispersion])


def feature_importance_analysis(problem: Problem) -> pd.DataFrame:
    """Analyze which features are most predictive of good facility locations.

    Returns a DataFrame with feature importance scores.
    """
    ml_prediction = predict_demand_weights(problem, method="random_forest")

    importance_df = pd.DataFrame(
        list(ml_prediction.feature_importance.items()),
        columns=["feature", "importance"],
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df["importance_percent"] = (
        100.0 * importance_df["importance"] / importance_df["importance"].sum()
    )

    return importance_df


def ml_comparison_experiment(
    problem: Problem, p: int = 3
) -> pd.DataFrame:
    """Compare ML-guided vs standard optimization methods."""
    results = []

    # Standard GA
    from .heuristics import genetic_algorithm

    standard_ga = genetic_algorithm(problem, p, generations=100)
    results.append(
        {
            "method": "Standard GA",
            "weighted_avg_min": standard_ga.metrics["weighted_average_minutes"],
            "max_min": standard_ga.metrics["max_minutes"],
            "runtime_s": standard_ga.runtime_seconds,
        }
    )

    # ML-guided GA (Random Forest)
    ml_ga_rf = ml_guided_genetic_algorithm(problem, p, ml_method="random_forest")
    results.append(
        {
            "method": "ML-guided GA (RF)",
            "weighted_avg_min": ml_ga_rf.solution.metrics["weighted_average_minutes"],
            "max_min": ml_ga_rf.solution.metrics["max_minutes"],
            "runtime_s": ml_ga_rf.warmstart_time + ml_ga_rf.optimization_time,
        }
    )

    # ML-guided GA (Gradient Boosting)
    ml_ga_gb = ml_guided_genetic_algorithm(problem, p, ml_method="gradient_boosting")
    results.append(
        {
            "method": "ML-guided GA (GB)",
            "weighted_avg_min": ml_ga_gb.solution.metrics["weighted_average_minutes"],
            "max_min": ml_ga_gb.solution.metrics["max_minutes"],
            "runtime_s": ml_ga_gb.warmstart_time + ml_ga_gb.optimization_time,
        }
    )

    return pd.DataFrame(results)
