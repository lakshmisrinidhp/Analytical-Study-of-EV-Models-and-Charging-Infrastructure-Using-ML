# Global EV Charging Alignment – II2202 Project

This repository contains all scripts, datasets, and generated outputs for the II2202 Fall 2025 project **“Global Electric Vehicle Landscape: An Analytical Study of EV Models and Charging Infrastructure.”**

## Highlights

- **Research questions:**  
1. To what extent can machine learning based alignment metrics explain the relationship between existing charging station characteristics (such as power level, number of ports, and geographic distribution) and the demand patterns of electric vehicle models across different regions?
2. Which countries or regions, as identified through modeled “predicted versus actual fast-charger
shares,” exhibit the highest potential risk of mismatch between electric vehicle charging demand
and available infrastructure?

- **Datasets:** curated CSVs under `datasets/`, including global charging station inventories, EV model lists, and regional summaries.

- **Analysis scripts:** Python code in `analysis/`  
  - `solve_rq1_rq2.py` builds demand scores, trains a weighted logistic regression classifier, and produces alignment tables.  
  - `solve_rq3_rq4.py` creates EV timeline summaries and future gap projections.  
  - `plot_rq1_rq2.py` and `plot_rq3_rq4.py` generate all figures used in the report.

- **Outputs:** Ready-to-use tables and figures in `outputs/`, e.g.,  
  - `rq1_rq2_country_alignment.csv` – country-level demand vs. fast-share metrics.  
  - `rq3_ev_model_timeline.png` – surge in BEV launches from 2018–2025.  
  - `rq4_future_gap_macro.png` – fast-charger upgrades needed to stay future-ready.

## Reproducing the Analysis

```bash
# RQ1 & RQ2 alignment model
python3 analysis/solve_rq1_rq2.py

# RQ3 & RQ4 timelines / future gaps
python3 analysis/solve_rq3_rq4.py

# Generate plots
python3 analysis/plot_rq1_rq2.py
python3 analysis/plot_rq3_rq4.py
```

All scripts assume Python 3.9+ with `pandas`, `numpy`, `scikit-learn`, and `matplotlib` installed.

## Project Authors

- Lakshmi Srinidh Pachabotla  
- Muhammad Ishfaq

For questions or contributions, feel free to open an issue or fork the project.
