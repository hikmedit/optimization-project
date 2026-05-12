"""Continuous local refinement using a calibrated time surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .optimization import Problem, Solution
from .travel_time import TravelTimeModel


@dataclass
class ContinuousRefinementResult:
    """Output of local coordinate refinement."""

    site_name: str
    district: str
    method: str
    initial_lon: float
    initial_lat: float
    optimized_lon: float
    optimized_lat: float
    snapped_lon: float
    snapped_lat: float
    initial_weighted_average_minutes: float
    snapped_weighted_average_minutes: float
    hessian_eigenvalues: tuple[float, float]
    locally_convex_surrogate: bool
    note: str


def refine_site_location(
    problem: Problem,
    solution: Solution,
    selected_site_index: int,
    travel_model: TravelTimeModel,
    neighborhoods: pd.DataFrame,
    radius_km: float = 3.0,
) -> list[ContinuousRefinementResult]:
    """Refine one opened station with descent on a minute-scale surrogate."""

    site = problem.all_sites.iloc[selected_site_index]
    district_key = str(site.get("district_key", ""))
    district = str(site.get("district", ""))
    initial_lon = float(site["lon"])
    initial_lat = float(site["lat"])
    fixed_active = [idx for idx in solution.active_site_indices if idx != selected_site_index]
    fixed_response = problem.time_matrix[:, fixed_active].min(axis=1)
    initial_candidate = travel_model.candidate_time_from_point(initial_lon, initial_lat, problem.demand)
    initial_response = np.minimum(fixed_response, initial_candidate)
    initial_objective = float(np.average(initial_response, weights=problem.weights))
    surrogate = _fit_quadratic_surrogate(
        initial_lon,
        initial_lat,
        problem,
        travel_model,
        fixed_response,
        radius_km=radius_km,
    )
    results = []
    for method in ("steepest_descent", "newton"):
        offset = _optimize_surrogate(surrogate, method=method, radius_km=radius_km)
        opt_lon, opt_lat = _offset_to_lon_lat(initial_lon, initial_lat, offset[0], offset[1])
        snapped_lon, snapped_lat = _snap_to_feasible(opt_lon, opt_lat, district_key, neighborhoods)
        snapped_candidate = travel_model.candidate_time_from_point(snapped_lon, snapped_lat, problem.demand)
        snapped_response = np.minimum(fixed_response, snapped_candidate)
        snapped_objective = float(np.average(snapped_response, weights=problem.weights))
        note = (
            "The surrogate objective is in calibrated response-time minutes. "
            "The final point is snapped to an official neighborhood point in "
            "the same district and re-evaluated with the same travel-time model."
        )
        if snapped_objective > initial_objective:
            snapped_lon, snapped_lat = initial_lon, initial_lat
            snapped_objective = initial_objective
            note = (
                "The nearest snapped neighborhood point did not improve the "
                "minute-scale objective, so the original candidate centroid is "
                "kept as the feasible recommendation."
            )
        eig = np.linalg.eigvalsh(surrogate["hessian"])
        results.append(
            ContinuousRefinementResult(
                site_name=str(site["site_name"]),
                district=district,
                method=method,
                initial_lon=initial_lon,
                initial_lat=initial_lat,
                optimized_lon=float(opt_lon),
                optimized_lat=float(opt_lat),
                snapped_lon=float(snapped_lon),
                snapped_lat=float(snapped_lat),
                initial_weighted_average_minutes=initial_objective,
                snapped_weighted_average_minutes=snapped_objective,
                hessian_eigenvalues=(float(eig[0]), float(eig[1])),
                locally_convex_surrogate=bool(np.all(eig >= -1e-8)),
                note=note,
            )
        )
    return results


def continuous_results_frame(results: list[ContinuousRefinementResult]) -> pd.DataFrame:
    """Convert continuous refinement results to a table."""

    return pd.DataFrame([result.__dict__ for result in results])


def _fit_quadratic_surrogate(
    center_lon: float,
    center_lat: float,
    problem: Problem,
    travel_model: TravelTimeModel,
    fixed_response: np.ndarray,
    radius_km: float,
) -> dict[str, np.ndarray]:
    offsets = []
    grid = np.linspace(-radius_km, radius_km, 5)
    for dx in grid:
        for dy in grid:
            if math.hypot(dx, dy) <= radius_km * 1.05:
                offsets.append((dx, dy))
    design = []
    values = []
    for dx, dy in offsets:
        lon, lat = _offset_to_lon_lat(center_lon, center_lat, dx, dy)
        candidate = travel_model.candidate_time_from_point(lon, lat, problem.demand)
        response = np.minimum(fixed_response, candidate)
        objective = float(np.average(response, weights=problem.weights))
        design.append([1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy])
        values.append(objective)
    coef, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(values), rcond=None)
    hessian = np.array([[coef[3], coef[4]], [coef[4], coef[5]]], dtype=float)
    return {"coef": coef, "hessian": hessian}


def _optimize_surrogate(surrogate: dict[str, np.ndarray], method: str, radius_km: float) -> np.ndarray:
    x = np.zeros(2, dtype=float)
    hessian = surrogate["hessian"]
    for _ in range(30):
        grad = _gradient(surrogate["coef"], x)
        if np.linalg.norm(grad) < 1e-6:
            break
        if method == "newton":
            eig_min = float(np.linalg.eigvalsh(hessian).min())
            regularized = hessian + np.eye(2) * max(0.0, -eig_min + 1e-3)
            direction = -np.linalg.solve(regularized, grad)
        else:
            direction = -grad
        norm = np.linalg.norm(direction)
        if norm == 0:
            break
        direction = direction / norm
        alpha_max = _max_step_within_radius(x, direction, radius_km)
        alpha = _golden_section(lambda a: _surrogate_value(surrogate["coef"], x + a * direction), 0.0, alpha_max)
        x = x + alpha * direction
    return x


def _surrogate_value(coef: np.ndarray, x: np.ndarray) -> float:
    dx, dy = float(x[0]), float(x[1])
    return float(coef @ np.array([1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy]))


def _gradient(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx, dy = float(x[0]), float(x[1])
    return np.array([coef[1] + coef[3] * dx + coef[4] * dy, coef[2] + coef[4] * dx + coef[5] * dy])


def _golden_section(func, low: float, high: float, iterations: int = 35) -> float:
    phi = (1 + 5**0.5) / 2
    inv_phi = 1 / phi
    c = high - (high - low) * inv_phi
    d = low + (high - low) * inv_phi
    for _ in range(iterations):
        if func(c) < func(d):
            high = d
        else:
            low = c
        c = high - (high - low) * inv_phi
        d = low + (high - low) * inv_phi
    return (low + high) / 2


def _max_step_within_radius(x: np.ndarray, direction: np.ndarray, radius_km: float) -> float:
    b = 2 * float(x @ direction)
    c = float(x @ x) - radius_km**2
    disc = max(0.0, b * b - 4 * c)
    roots = [(-b + math.sqrt(disc)) / 2, (-b - math.sqrt(disc)) / 2]
    positive = [root for root in roots if root >= 0]
    return min(positive) if positive else radius_km


def _offset_to_lon_lat(lon: float, lat: float, dx_km: float, dy_km: float) -> tuple[float, float]:
    new_lat = lat + dy_km / 111.32
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    new_lon = lon + dx_km / (111.32 * cos_lat)
    return new_lon, new_lat


def _snap_to_feasible(lon: float, lat: float, district_key: str, neighborhoods: pd.DataFrame) -> tuple[float, float]:
    candidates = neighborhoods
    if district_key:
        same_district = neighborhoods[neighborhoods["district_key"] == district_key]
        if not same_district.empty:
            candidates = same_district
    dx = (candidates["lon"].to_numpy(dtype=float) - lon) * 111.32 * math.cos(math.radians(lat))
    dy = (candidates["lat"].to_numpy(dtype=float) - lat) * 111.32
    idx = int(np.argmin(dx * dx + dy * dy))
    row = candidates.iloc[idx]
    return float(row["lon"]), float(row["lat"])
