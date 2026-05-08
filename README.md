# Istanbul Fire Station Expansion Optimization

This project implements the YZV202E proposal **"Optimizing Fire Station Expansion in Istanbul to Reduce Emergency Response Time"** as a notebook-first optimization study.

The implementation follows the instructor feedback:

- All service costs are reported in **minutes**.
- The discrete p-median model and continuous refinement use the same travel-time interface.
- If road routing is unavailable, the project uses one calibrated geometric time proxy throughout, not a mix of road time and raw Euclidean distance.
- The recommended model includes an equity refinement that minimizes the worst district response time within a 2% weighted-average response-time tolerance.

## Structure

- `notebooks/istanbul_fire_station_expansion.ipynb` - main formatted Jupyter Notebook.
- `src/istanbul_fire_opt/` - reusable data, travel-time, optimization, heuristic, continuous-refinement, and visualization code.
- `scripts/run_pipeline.py` - command-line experiment runner.
- `scripts/generate_report_assets.py` - generates report CSV tables and figures.
- `report/main.tex` - IEEE conference report source.
- `tests/test_core.py` - core optimization unit tests.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_pipeline.py
jupyter notebook notebooks/istanbul_fire_station_expansion.ipynb
```

The code can run without optional packages such as `pulp`, `osmnx`, and `ipywidgets`. In that case, it uses exact enumeration for district-scale p-median instances and a calibrated geometric travel-time proxy.

## Data Sources

The data pipeline downloads official IBB open-data sources into `data/raw/`:

- Fire station locations, 2025.
- TUIK/IBB district population table.
- IBB fire-event average arrival-time table.
- First-degree emergency transportation roads GeoJSON.
- Mukhtar office location GeoJSON, used to derive official district centroids.

## Report

Generate figures and tables, then compile:

```bash
PYTHONPATH=src python scripts/generate_report_assets.py
cd report
pdflatex main.tex
pdflatex main.tex
```

