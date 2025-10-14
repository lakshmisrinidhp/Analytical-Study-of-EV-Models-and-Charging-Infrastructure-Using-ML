#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


DATA_DIR = Path("datasets")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


EU_COUNTRIES: Set[str] = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}

MIDDLE_EAST_COUNTRIES: Set[str] = {
    "AE",
    "BH",
    "IL",
    "IQ",
    "IR",
    "JO",
    "KW",
    "LB",
    "OM",
    "PS",
    "QA",
    "SA",
    "SY",
    "TR",
    "YE",
}

LATAM_COUNTRIES: Set[str] = {
    "AR",
    "BO",
    "BR",
    "CL",
    "CO",
    "CR",
    "CU",
    "DO",
    "EC",
    "GT",
    "HN",
    "MX",
    "NI",
    "PA",
    "PE",
    "PY",
    "SV",
    "UY",
    "VE",
}


def _extract_market_tokens(region_text: str) -> List[str]:
    tokens = re.findall(r"[A-Z]{2,}", region_text.replace("&", "/"))
    return tokens


def _expand_to_countries(tokens: Iterable[str]) -> Set[str]:
    countries: Set[str] = set()
    for token in tokens:
        if token == "EU":
            countries.update(EU_COUNTRIES)
        elif token == "UK":
            countries.add("GB")
        elif token == "US":
            countries.add("US")
        elif token == "CA":
            countries.add("CA")
        elif token == "CN":
            countries.add("CN")
        elif token == "JP":
            countries.add("JP")
        elif token == "IN":
            countries.add("IN")
        elif token == "VN":
            countries.add("VN")
        elif token == "ME":
            countries.update(MIDDLE_EAST_COUNTRIES)
        elif token == "LA":
            countries.update(LATAM_COUNTRIES)
    return countries


def map_macro_region(country_code: str) -> str:
    if country_code == "US":
        return "United_States"
    if country_code == "GB":
        return "United_Kingdom"
    if country_code == "CA":
        return "Canada"
    if country_code == "CN":
        return "China"
    if country_code == "JP":
        return "Japan"
    if country_code == "IN":
        return "India"
    if country_code in EU_COUNTRIES:
        return "European_Union"
    if country_code in MIDDLE_EAST_COUNTRIES:
        return "Middle_East"
    if country_code in LATAM_COUNTRIES:
        return "Latin_America"
    return "Other"


def build_model_country_links(models: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in models.iterrows():
        market_regions = row.get("market_regions", "")
        if isinstance(market_regions, float) and math.isnan(market_regions):
            continue
        tokens = _extract_market_tokens(str(market_regions))
        country_codes = _expand_to_countries(tokens)
        if not country_codes:
            continue
        for code in country_codes:
            records.append(
                {
                    "model_id": idx,
                    "make": row["make"],
                    "model": row["model"],
                    "first_year": int(row["first_year"]),
                    "country_code": code,
                    "macro_region": map_macro_region(code),
                }
            )
    return pd.DataFrame(records)


def timeline_analysis(models: pd.DataFrame, links: pd.DataFrame) -> None:
    timeline = (
        models.groupby("first_year")
        .size()
        .rename("models_released")
        .reset_index()
        .sort_values("first_year")
    )
    timeline["cumulative_models"] = timeline["models_released"].cumsum()
    timeline.to_csv(OUTPUT_DIR / "rq3_ev_model_timeline_global.csv", index=False)

    recent_cut = 2020
    near_term_cut = 2023

    region_metrics = (
        links.groupby("macro_region")
        .agg(
            model_entries=("model_id", "nunique"),
            median_launch=("first_year", "median"),
            share_since_2018=("first_year", lambda x: (x >= 2018).mean()),
            share_since_recent=("first_year", lambda x: (x >= recent_cut).mean()),
            share_since_near_term=("first_year", lambda x: (x >= near_term_cut).mean()),
        )
        .reset_index()
    )
    region_metrics = region_metrics.sort_values("median_launch", ascending=False)
    region_metrics.to_csv(OUTPUT_DIR / "rq3_macro_timeline_summary.csv", index=False)


def future_gap_analysis(
    country_alignment: pd.DataFrame,
    ev_models: pd.DataFrame,
    links: pd.DataFrame,
) -> None:
    recent_mask = links["first_year"] >= 2020
    near_term_mask = links["first_year"] >= 2023

    recent_share = (
        links.groupby("country_code")["first_year"]
        .apply(lambda x: (x >= 2020).mean())
        .rename("share_models_since_2020")
    )
    near_term_share = (
        links.groupby("country_code")["first_year"]
        .apply(lambda x: (x >= 2023).mean())
        .rename("share_models_since_2023")
    )

    future = country_alignment.merge(recent_share, on="country_code", how="left")
    future = future.merge(near_term_share, on="country_code", how="left")

    future["share_models_since_2020"] = future["share_models_since_2020"].fillna(0.0)
    future["share_models_since_2023"] = future["share_models_since_2023"].fillna(0.0)

    # Scenario: fast-charging demand rises with the share of models launched since 2020.
    future["future_demand_share"] = np.clip(
        future["demand_score_filled"] + 0.2 * future["share_models_since_2020"] + 0.1 * future["share_models_since_2023"],
        0,
        0.97,
    )
    future["future_gap_share"] = future["future_demand_share"] - future["fast_predicted_share"]
    future["future_gap_share"] = future["future_gap_share"].clip(lower=0)
    future["stations_to_upgrade"] = future["future_gap_share"] * future["station_count"]
    future["macro_region"] = future["country_code"].apply(map_macro_region)

    future = future[
        [
            "country_code",
            "macro_region",
            "station_count",
            "fast_actual_share",
            "fast_predicted_share",
            "demand_score_filled",
            "future_demand_share",
            "predicted_gap",
            "future_gap_share",
            "stations_to_upgrade",
            "risk_level",
            "confidence_level",
            "share_models_since_2020",
            "share_models_since_2023",
        ]
    ]

    future.sort_values("future_gap_share", ascending=False).to_csv(
        OUTPUT_DIR / "rq4_future_gap_country.csv", index=False
    )

    macro_future = (
        future.groupby("macro_region")
        .agg(
            total_stations=("station_count", "sum"),
            avg_fast_predicted_share=("fast_predicted_share", "mean"),
            avg_demand=("demand_score_filled", "mean"),
            avg_future_demand=("future_demand_share", "mean"),
            avg_future_gap=("future_gap_share", "mean"),
            stations_to_upgrade=("stations_to_upgrade", "sum"),
            share_models_since_2020=("share_models_since_2020", "mean"),
            share_models_since_2023=("share_models_since_2023", "mean"),
        )
        .reset_index()
    )
    macro_future.to_csv(OUTPUT_DIR / "rq4_future_gap_macro.csv", index=False)


def main() -> None:
    models = pd.read_csv(DATA_DIR / "ev_models_2025.csv")
    models = models.dropna(subset=["first_year"])
    models["first_year"] = models["first_year"].astype(int)

    links = build_model_country_links(models)

    timeline_analysis(models, links)

    country_alignment = pd.read_csv(OUTPUT_DIR / "rq1_rq2_country_alignment.csv")
    future_gap_analysis(country_alignment, models, links)

    print("RQ3 and RQ4 analyses complete. Outputs saved in the outputs directory.")


if __name__ == "__main__":
    main()
