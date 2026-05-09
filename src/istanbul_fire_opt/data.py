"""Download and prepare official IBB project data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from .config import IBB_DATASETS, RAW_DATA_DIR
from .utils import as_float, normalize_key, title_case_turkish
from .xlsx import read_xlsx


@dataclass(frozen=True)
class ProjectData:
    """Clean tables used by the optimization workflow."""

    districts: pd.DataFrame
    stations: pd.DataFrame
    neighborhoods: pd.DataFrame
    arrival_minutes: float
    emergency_roads_path: Path
    metadata: dict[str, object]


def download_official_data(force: bool = False) -> dict[str, Path]:
    """Download source files to data/raw and return local paths."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, info in IBB_DATASETS.items():
        path = RAW_DATA_DIR / info["filename"]
        if force or not path.exists():
            _download(info["url"], path)
        paths[key] = path
    return paths


def load_project_data(force_download: bool = False) -> ProjectData:
    """Load clean station, population, centroid, and calibration data."""

    paths = download_official_data(force=force_download)
    stations = load_fire_stations(paths["fire_stations_2025"])
    population = load_population(paths["population"])
    neighborhoods = load_neighborhood_points(paths["muhtarlik_locations"])
    districts = build_district_table(population, neighborhoods)
    arrival_minutes = load_arrival_minutes(paths["arrival_time"])
    metadata = {
        "station_count": int(len(stations)),
        "district_count": int(len(districts)),
        "neighborhood_count": int(len(neighborhoods)),
        "arrival_minutes_target": arrival_minutes,
        "source_titles": {key: value["title"] for key, value in IBB_DATASETS.items()},
    }
    return ProjectData(
        districts=districts,
        stations=stations,
        neighborhoods=neighborhoods,
        arrival_minutes=arrival_minutes,
        emergency_roads_path=paths["emergency_roads"],
        metadata=metadata,
    )


def load_fire_stations(path: str | Path) -> pd.DataFrame:
    """Load 2025 fire station coordinates."""

    raw = read_xlsx(path)
    required = ["Birim", "İlçe", "Koordinat(Y)", "Koordinat(X)"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"fire station file missing columns: {missing}")
    status = raw["Statü"] if "Statü" in raw.columns else pd.Series([""] * len(raw))
    address = raw["İstasyon Adresi"] if "İstasyon Adresi" in raw.columns else pd.Series([""] * len(raw))
    df = pd.DataFrame(
        {
            "site_name": raw["Birim"].astype(str).str.strip(),
            "district": raw["İlçe"].map(title_case_turkish),
            "district_key": raw["İlçe"].map(normalize_key),
            "status": status.astype(str).str.strip(),
            "address": address.astype(str).str.strip(),
            "lon": raw["Koordinat(Y)"].map(as_float),
            "lat": raw["Koordinat(X)"].map(as_float),
        }
    )
    df = df.dropna(subset=["lon", "lat"])
    df = df[df["site_name"].str.len() > 0].reset_index(drop=True)
    df["site_id"] = [f"existing_{idx:03d}" for idx in range(len(df))]
    df["is_existing"] = True
    return df


def load_population(path: str | Path) -> pd.DataFrame:
    """Load latest district population totals from IBB/TUIK age-sex table."""

    raw = read_xlsx(path)
    if "Yıl" not in raw.columns or "İlçe" not in raw.columns:
        raise ValueError("population file must contain 'Yıl' and 'İlçe'")
    raw["Yıl"] = pd.to_numeric(raw["Yıl"], errors="coerce")
    latest_year = int(raw["Yıl"].max())
    latest = raw[raw["Yıl"] == latest_year].copy()
    excluded = {"Yıl", "İlçe", "ilce_kodu"}
    numeric_cols = [col for col in latest.columns if col not in excluded]
    for col in numeric_cols:
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0)
    latest["population"] = latest[numeric_cols].sum(axis=1)
    return (
        latest.assign(
            district=latest["İlçe"].map(title_case_turkish),
            district_key=latest["İlçe"].map(normalize_key),
            population_year=latest_year,
        )[["district", "district_key", "population_year", "population"]]
        .drop_duplicates("district_key")
        .reset_index(drop=True)
    )


def load_neighborhood_points(path: str | Path) -> pd.DataFrame:
    """Load mukhtar office points and keep district/neighborhood coordinates."""

    with Path(path).open(encoding="utf-8") as file:
        geojson = json.load(file)
    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates") or [None, None]
        district_raw = props.get("İlçe Adı") or props.get("Ilce Adi")
        neighborhood_raw = props.get("Mahalle Adı") or props.get("Mahalle Adi")
        rows.append(
            {
                "district": title_case_turkish(district_raw),
                "district_key": normalize_key(district_raw),
                "neighborhood": title_case_turkish(neighborhood_raw),
                "lon": as_float(props.get("Longtitude", coords[0])),
                "lat": as_float(props.get("Latitude", coords[1])),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    return df[df["district_key"].str.len() > 0].reset_index(drop=True)


def build_district_table(population: pd.DataFrame, neighborhoods: pd.DataFrame) -> pd.DataFrame:
    """Join latest population with official neighborhood-derived centroids."""

    centroids = (
        neighborhoods.groupby("district_key")
        .agg(
            district_from_geo=("district", "first"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            neighborhood_count=("neighborhood", "count"),
        )
        .reset_index()
    )
    districts = population.merge(centroids, on="district_key", how="inner")
    districts["demand_id"] = [f"demand_{idx:02d}" for idx in range(len(districts))]
    districts["weight"] = districts["population"].astype(float)
    columns = [
        "demand_id",
        "district",
        "district_key",
        "population_year",
        "population",
        "weight",
        "lon",
        "lat",
        "neighborhood_count",
    ]
    return districts[columns].sort_values("district").reset_index(drop=True)


def load_arrival_minutes(path: str | Path) -> float:
    """Return latest all-event arrival-time calibration target in minutes.

    The IBB spreadsheet stores time-like values below one. Multiplying by 24
    recovers the published minute-scale values, e.g. 0.276388... -> 6.633 min.
    """

    raw = read_xlsx(path)
    if "Yil" not in raw.columns and "Yıl" in raw.columns:
        raw = raw.rename(columns={"Yıl": "Yil"})
    if "Yil" not in raw.columns:
        raise ValueError("arrival-time file must contain a year column")
    raw["Yil"] = pd.to_numeric(raw["Yil"], errors="coerce")
    row = raw.loc[raw["Yil"].idxmax()]
    preferred = ["Tüm İtfai Olaylar", "Yanginlar", "Yangınlar"]
    for column in preferred:
        if column in raw.columns:
            value = as_float(row[column])
            return value * 24.0 if 0 < value < 1 else value
    numeric = pd.to_numeric(row.drop(labels=["Yil"], errors="ignore"), errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("arrival-time file has no numeric calibration value")
    value = float(numeric.iloc[0])
    return value * 24.0 if 0 < value < 1 else value


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "optimization-final-project/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    except URLError as exc:
        raise RuntimeError(f"could not download {url}") from exc
