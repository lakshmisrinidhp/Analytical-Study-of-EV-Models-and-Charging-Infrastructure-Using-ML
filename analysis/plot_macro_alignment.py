#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("outputs/rq1_rq2_macro_alignment.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "macro_alignment_gap.png"


def main() -> None:
    macro = pd.read_csv(INPUT_PATH)
    macro = macro.sort_values("predicted_gap", ascending=False)

    x = range(len(macro))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        [i - width for i in x],
        macro["avg_demand"],
        width=width,
        color="#1f77b4",
        label="Demand Score",
    )
    ax.bar(
        x,
        macro["avg_actual_share"],
        width=width,
        color="#2ca02c",
        label="Actual Fast Share",
    )
    ax.bar(
        [i + width for i in x],
        macro["avg_predicted_share"],
        width=width,
        color="#ff7f0e",
        label="Predicted Fast Share",
    )

    ax.set_ylabel("Share / Score (0-1)")
    ax.set_xlabel("Macro Region")
    ax.set_title("Macro-Level Alignment of Fast Charging Supply vs EV Demand")
    ax.set_xticks(list(x))
    ax.set_xticklabels(macro["macro_region"], rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved macro alignment chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
