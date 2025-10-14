#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_ev_model_timeline():
    timeline = pd.read_csv(OUTPUT_DIR / "rq3_ev_model_timeline_global.csv")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(timeline["first_year"], timeline["models_released"], marker="o", color="#1f77b4")
    ax1.set_xlabel("Launch year")
    ax1.set_ylabel("Models released", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(timeline["first_year"], timeline["cumulative_models"], marker="s", linestyle="--", color="#ff7f0e")
    ax2.set_ylabel("Cumulative models", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax1.set_title("Global EV Model Launch Timeline")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq3_ev_model_timeline.png", dpi=300)
    plt.close(fig)


def plot_macro_model_share():
    macro = pd.read_csv(OUTPUT_DIR / "rq3_macro_timeline_summary.csv")
    macro = macro.sort_values("median_launch", ascending=False)

    indices = range(len(macro))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in indices], macro["share_since_2018"], width=width, color="#1f77b4", label="Since 2018")
    ax.bar(indices, macro["share_since_recent"], width=width, color="#2ca02c", label="Since 2020")
    ax.bar([i + width for i in indices], macro["share_since_near_term"], width=width, color="#ff7f0e", label="Since 2023")

    ax.set_xticks(list(indices))
    ax.set_xticklabels(macro["macro_region"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of EV models")
    ax.set_title("Share of EV Models by Launch Period and Region")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq3_macro_model_share.png", dpi=300)
    plt.close(fig)


def plot_future_gap_macro():
    macro = pd.read_csv(OUTPUT_DIR / "rq4_future_gap_macro.csv")
    macro = macro.sort_values("avg_future_gap", ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(macro["macro_region"], macro["avg_future_gap"], color="#d62728", alpha=0.8)
    ax1.set_ylabel("Average future gap (demand - predicted fast)", color="#d62728")
    ax1.set_xlabel("Macro region")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0, 0.7)

    ax2 = ax1.twinx()
    ax2.plot(macro["macro_region"], macro["stations_to_upgrade"], marker="o", color="#1f77b4")
    ax2.set_ylabel("Stations to upgrade", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title("Future Fast-Charging Gaps by Region")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq4_future_gap_macro.png", dpi=300)
    plt.close(fig)


def plot_future_gap_country():
    country = pd.read_csv(OUTPUT_DIR / "rq4_future_gap_country.csv")
    focus = country[country["station_count"] >= 1000].sort_values("future_gap_share", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(focus["country_code"], focus["future_gap_share"], color="#ff7f0e")
    ax.set_ylabel("Future gap share")
    ax.set_xlabel("Country code (≥ 1,000 stations)")
    ax.set_ylim(0, 0.9)
    ax.set_title("Top Future Fast-Charging Gaps by Country")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rq4_future_gap_country.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_ev_model_timeline()
    plot_macro_model_share()
    plot_future_gap_macro()
    plot_future_gap_country()
    print("Saved RQ3/RQ4 visualisations to outputs.")


if __name__ == "__main__":
    main()
