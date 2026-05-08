"""Project configuration and official data sources."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_EPSILON = 0.02
DEFAULT_COVERAGE_THRESHOLDS = (5.0, 8.0)
DEFAULT_BUDGETS = (1, 3, 5)
DEFAULT_RANDOM_SEED = 2026

IBB_DATASETS = {
    "fire_stations_2025": {
        "title": "IBB Fire Station Location Data 2025",
        "url": "https://data.ibb.gov.tr/dataset/75cd7b09-dafb-41fa-90c9-9c7396a58700/resource/d3efb4ce-90b4-46b6-a96e-bbbe5b1e0234/download/itfaiye-istasyon-konumlar.2025.xlsx",
        "filename": "itfaiye-istasyon-konumlar-2025.xlsx",
    },
    "population": {
        "title": "IBB/TUIK Population Information",
        "url": "https://data.ibb.gov.tr/dataset/6646992d-e1cf-4d5b-bb79-3f92dcc3fc38/resource/da5a16eb-50a2-4264-88d1-92e4a859a8f7/download/nufus-bilgileri.xlsx",
        "filename": "nufus-bilgileri.xlsx",
    },
    "arrival_time": {
        "title": "IBB Fire Event Average Arrival Time",
        "url": "https://data.ibb.gov.tr/dataset/93b52900-8b72-4eed-8ee1-6bfd284a390c/resource/ad9926fd-b1d5-4832-bc42-c3b237b93d59/download/itfaiye-ort.-var-suresi-2024.xlsx",
        "filename": "itfaiye-ortalama-varis-suresi-2024.xlsx",
    },
    "emergency_roads": {
        "title": "IBB First-degree Emergency Transportation Roads",
        "url": "https://data.ibb.gov.tr/dataset/5122aba9-da03-4be7-8951-defb7329d113/resource/04a631ea-e5b7-4843-af98-203c1dbc6bb7/download/acil_ulasim_yollari_verisi.geojson",
        "filename": "acil-ulasim-yollari.geojson",
    },
    "muhtarlik_locations": {
        "title": "IBB Mukhtar Office Address Locations",
        "url": "https://data.ibb.gov.tr/dataset/c310cde9-92b1-4c51-9575-d71b1dc7ac43/resource/71f75529-7fae-4a85-b05f-664c62eda422/download/muhtarlik_lokasyon.geojson",
        "filename": "muhtarlik-lokasyon.geojson",
    },
    "fire_counts": {
        "title": "IBB Fire Counts",
        "url": "https://data.ibb.gov.tr/dataset/01ff7818-53cd-4eef-b327-47a31afe70ad/resource/0431f40c-5926-4bfd-b87c-2bf2a4cf7b36/download/yangn-says_2022_2023.xlsx",
        "filename": "yangin-sayilari-2022-2025.xlsx",
    },
}

