"""Matplotlib visualizations for the notebook and report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimization import Problem, Solution
from .workflow import assignment_frame, selected_sites_frame


def plot_solution_map(
    problem: Problem,
    solution: Solution,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot demand response times, existing stations, and opened candidates."""

    assignments = assignment_frame(problem, solution)
    selected = selected_sites_frame(problem, solution)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    scatter = ax.scatter(
        assignments["demand_lon"],
        assignments["demand_lat"],
        c=assignments["response_minutes"],
        s=np.sqrt(assignments["population"]) / 8,
        cmap="viridis_r",
        alpha=0.88,
        edgecolor="black",
        linewidth=0.35,
        label="District demand",
    )
    ax.scatter(
        problem.existing_sites["lon"],
        problem.existing_sites["lat"],
        marker="^",
        s=18,
        color="#444444",
        alpha=0.55,
        label="Existing stations",
    )
    if not selected.empty:
        ax.scatter(
            selected["lon"],
            selected["lat"],
            marker="*",
            s=220,
            color="#d62728",
            edgecolor="black",
            linewidth=0.8,
            label="Opened candidate sites",
        )
        for _, row in selected.iterrows():
            ax.annotate(str(row["district"]), (row["lon"], row["lat"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title or f"{solution.method}, p={solution.p}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Response time (minutes)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_method_comparison(results: pd.DataFrame, save_path: str | Path | None = None):
    """Compare average, P90, and worst response times by method."""

    df = results[results["p"] > 0].copy()
    labels = [f"{row.method}\np={row.p}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 4.7))
    ax.bar(x - width, df["weighted_avg_min"], width, label="Weighted avg")
    ax.bar(x, df["p90_min"], width, label="P90")
    ax.bar(x + width, df["max_min"], width, label="Max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Minutes")
    ax.set_title("Method comparison: average vs district service gaps")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_coverage(results: pd.DataFrame, save_path: str | Path | None = None):
    """Plot weighted coverage at five and eight minutes."""

    df = results[results["p"] > 0].copy()
    labels = [f"{row.method}\np={row.p}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(x - width / 2, df["coverage_5"], width, label="<= 5 min")
    ax.bar(x + width / 2, df["coverage_8"], width, label="<= 8 min")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted demand share")
    ax.set_title("Coverage under response-time thresholds")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig, ax

