#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_macro_alignment():
    macro = pd.read_csv(OUTPUT_DIR / "rq1_rq2_macro_alignment.csv")
    macro = macro.sort_values("predicted_gap", ascending=False)

    x = range(len(macro))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], macro["avg_demand"], width=width, color="#1f77b4", label="Demand score")
    ax.bar(x, macro["avg_actual_share"], width=width, color="#2ca02c", label="Actual fast share")
    ax.bar(
        [i + width for i in x],
        macro["avg_predicted_share"],
        width=width,
        color="#ff7f0e",
        label="Predicted fast share",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(macro["macro_region"], rotation=30, ha="right")
    ax.set_ylabel("Share / score")
    ax.set_ylim(0, 1.05)
    ax.set_title("RQ1 – Macro Alignment of EV Demand and Fast-Charging Supply")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq1_macro_alignment.png", dpi=300)
    plt.close(fig)


def plot_country_alignment_scatter():
    country = pd.read_csv(OUTPUT_DIR / "rq1_rq2_country_alignment.csv")
    country = country[country["confidence_level"] == "standard"]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        country["demand_score_filled"],
        country["fast_actual_share"],
        s=country["station_count"] / 20,
        c=country["predicted_gap"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Demand score")
    ax.set_ylabel("Actual fast share")
    ax.set_title("RQ1 – Country-Level Demand vs. Actual Fast Charging")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted gap (demand − predicted fast share)")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq1_country_alignment_scatter.png", dpi=300)
    plt.close(fig)


def plot_country_alignment_annotated():
    country = pd.read_csv(OUTPUT_DIR / "rq1_rq2_country_alignment.csv")
    country = country[country["confidence_level"] == "standard"]

    top_gap = country.sort_values("predicted_gap", ascending=False).head(12)
    top_size = country.sort_values("station_count", ascending=False).head(8)
    labels = pd.concat([top_gap, top_size]).drop_duplicates(subset=["country_code"])

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        country["demand_score_filled"],
        country["fast_actual_share"],
        s=country["station_count"] / 20,
        c=country["predicted_gap"],
        cmap="viridis",
        alpha=0.7,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Demand score")
    ax.set_ylabel("Actual fast share")
    ax.set_title("RQ1 – Demand vs. Actual Fast Charging (Annotated)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted gap (demand − predicted fast share)")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)

    for _, row in labels.iterrows():
        ax.annotate(
            row["country_code"],
            (row["demand_score_filled"], row["fast_actual_share"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color="black",
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq1_country_alignment_scatter_annotated.png", dpi=300)
    plt.close(fig)


def plot_top_risk_countries():
    country = pd.read_csv(OUTPUT_DIR / "rq1_rq2_country_alignment.csv")
    focus = (
        country[country["station_count"] >= 1000]
        .sort_values("predicted_gap", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(focus["country_code"], focus["predicted_gap"], color="#d62728")
    ax.set_ylabel("Predicted gap (demand − predicted fast share)")
    ax.set_xlabel("Country code (≥ 1,000 stations)")
    ax.set_ylim(0, focus["predicted_gap"].max() * 1.1)
    ax.set_title("RQ2 – Top High-Risk Countries by Predicted Gap")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq2_top_risk_countries.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_macro_alignment()
    plot_country_alignment_scatter()
    plot_country_alignment_annotated()
    plot_top_risk_countries()
    print("Saved RQ1/RQ2 visualisations to outputs.")


if __name__ == "__main__":
    main()
