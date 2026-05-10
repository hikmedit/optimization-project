"""Travel-time estimation in minutes.

The project uses one consistent metric throughout. If road-network routing is
available, the model can be extended at this layer. The default implementation
uses a calibrated geometric time proxy in minutes, so the discrete and
continuous optimization stages do not mix road time with raw distance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


@dataclass
class TravelTimeModel:
    """Consistent station-to-demand service-cost model."""

    mode: Literal["calibrated_geometric", "road_network"] = "calibrated_geometric"
    base_speed_kmh: float = 42.0
    calibration_factor: float = 1.0
    calibration_target_minutes: float | None = None
    note: str = (
        "Calibrated geometric proxy in minutes. This fallback is used because "
        "full road-routing dependencies/data are optional; no raw Euclidean "
        "distance objective is mixed into the optimization."
    )

    def distance_matrix_km(
        self,
        origins: pd.DataFrame,
        destinations: pd.DataFrame,
        origin_lon: str = "lon",
        origin_lat: str = "lat",
        dest_lon: str = "lon",
        dest_lat: str = "lat",
    ) -> np.ndarray:
        """Great-circle distance matrix from origins to destinations."""

        o_lat = np.deg2rad(origins[origin_lat].to_numpy(dtype=float))[:, None]
        o_lon = np.deg2rad(origins[origin_lon].to_numpy(dtype=float))[:, None]
        d_lat = np.deg2rad(destinations[dest_lat].to_numpy(dtype=float))[None, :]
        d_lon = np.deg2rad(destinations[dest_lon].to_numpy(dtype=float))[None, :]
        delta_lat = d_lat - o_lat
        delta_lon = d_lon - o_lon
        a = np.sin(delta_lat / 2.0) ** 2 + np.cos(o_lat) * np.cos(d_lat) * np.sin(delta_lon / 2.0) ** 2
        return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def time_matrix_minutes(self, origins: pd.DataFrame, destinations: pd.DataFrame) -> np.ndarray:
        """Return travel-time matrix in minutes."""

        distance_km = self.distance_matrix_km(origins, destinations)
        return self.minutes_from_distance_km(distance_km)

    def minutes_from_distance_km(self, distance_km: np.ndarray | float) -> np.ndarray | float:
        """Convert distance to calibrated response-time minutes."""

        minutes = np.asarray(distance_km, dtype=float) / self.base_speed_kmh * 60.0
        minutes *= self.calibration_factor
        if np.isscalar(distance_km):
            return float(minutes)
        return minutes

    def fit_to_baseline(
        self,
        demand: pd.DataFrame,
        existing_sites: pd.DataFrame,
        weights: np.ndarray,
        target_minutes: float,
    ) -> "TravelTimeModel":
        """Scale the proxy so current baseline matches IBB average arrival time."""

        uncalibrated = TravelTimeModel(
            mode=self.mode,
            base_speed_kmh=self.base_speed_kmh,
            calibration_factor=1.0,
            calibration_target_minutes=target_minutes,
            note=self.note,
        )
        base_time = uncalibrated.time_matrix_minutes(demand, existing_sites)
        baseline = base_time.min(axis=1)
        weighted_average = float(np.average(baseline, weights=weights))
        if not math.isfinite(weighted_average) or weighted_average <= 0:
            factor = 1.0
        else:
            factor = target_minutes / weighted_average
        self.calibration_factor = factor
        self.calibration_target_minutes = target_minutes
        return self

    def candidate_time_from_point(
        self,
        point_lon: float,
        point_lat: float,
        demand: pd.DataFrame,
    ) -> np.ndarray:
        """Travel-time vector from a trial point to demand zones."""

        point = pd.DataFrame({"lon": [point_lon], "lat": [point_lat]})
        return self.time_matrix_minutes(demand, point)[:, 0]


def try_make_road_network_model() -> TravelTimeModel:
    """Return a road-network model if optional routing dependencies exist.

    The current implementation intentionally does not silently download a large
    OSM road graph. It exposes the dependency check and keeps the calibrated
    proxy as the reproducible default.
    """

    try:
        import networkx as nx  # noqa: F401
        import osmnx as ox
    except Exception:
        return TravelTimeModel()
    try:
        graph = ox.graph_from_place("Istanbul, Turkey", network_type="drive", simplify=True)
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        return RoadNetworkTravelTimeModel(graph=graph, osmnx_module=ox)
    except Exception:
        return TravelTimeModel(
            note=(
                "OSMnx was importable, but the road graph could not be retrieved. "
                "The project therefore uses the calibrated geometric proxy in minutes."
            )
        )


@dataclass
class RoadNetworkTravelTimeModel(TravelTimeModel):
    """Optional OSMnx/NetworkX shortest-path travel-time model."""

    graph: Any | None = None
    osmnx_module: Any | None = None

    def __post_init__(self) -> None:
        self.mode = "road_network"
        self.note = (
            "OSMnx road-network shortest-path travel time in minutes, calibrated "
            "against the IBB average arrival-time record."
        )

    def time_matrix_minutes(self, origins: pd.DataFrame, destinations: pd.DataFrame) -> np.ndarray:
        import networkx as nx

        if self.graph is None or self.osmnx_module is None:
            return super().time_matrix_minutes(origins, destinations)
        ox = self.osmnx_module
        demand_nodes = ox.distance.nearest_nodes(
            self.graph,
            X=origins["lon"].to_numpy(dtype=float),
            Y=origins["lat"].to_numpy(dtype=float),
        )
        site_nodes = ox.distance.nearest_nodes(
            self.graph,
            X=destinations["lon"].to_numpy(dtype=float),
            Y=destinations["lat"].to_numpy(dtype=float),
        )
        matrix = np.empty((len(origins), len(destinations)), dtype=float)
        for j, site_node in enumerate(site_nodes):
            lengths = nx.single_source_dijkstra_path_length(self.graph, site_node, weight="travel_time")
            for i, demand_node in enumerate(demand_nodes):
                seconds = lengths.get(demand_node)
                if seconds is None:
                    fallback_km = self.distance_matrix_km(origins.iloc[[i]], destinations.iloc[[j]])[0, 0]
                    matrix[i, j] = self.minutes_from_distance_km(fallback_km)
                else:
                    matrix[i, j] = (seconds / 60.0) * self.calibration_factor
        return matrix

    def fit_to_baseline(
        self,
        demand: pd.DataFrame,
        existing_sites: pd.DataFrame,
        weights: np.ndarray,
        target_minutes: float,
    ) -> "TravelTimeModel":
        self.calibration_factor = 1.0
        base_time = self.time_matrix_minutes(demand, existing_sites)
        baseline = base_time.min(axis=1)
        weighted_average = float(np.average(baseline, weights=weights))
        self.calibration_factor = 1.0 if weighted_average <= 0 else target_minutes / weighted_average
        self.calibration_target_minutes = target_minutes
        return self
