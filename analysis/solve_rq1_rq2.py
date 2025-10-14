#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("datasets")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Source: EU member states list (as of 2025)
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


@dataclass
class DemandInputs:
    models: pd.DataFrame
    stations: pd.DataFrame
    country_summary: pd.DataFrame
    world_summary: pd.DataFrame


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
        elif token == "VN":
            countries.add("VN")
        elif token == "ME":
            countries.update(MIDDLE_EAST_COUNTRIES)
        elif token == "LA":
            countries.update(LATAM_COUNTRIES)
    return countries


def build_ev_demand(inputs: DemandInputs) -> pd.DataFrame:
    models = inputs.models.copy()
    models = models.dropna(subset=["first_year"])
    models["first_year"] = models["first_year"].astype(int)

    # KMeans clusters capture temporal generations of BEVs.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    models["generation_cluster"] = kmeans.fit_predict(models[["first_year"]])
    centers = kmeans.cluster_centers_.flatten()
    center_rank = np.argsort(centers)
    normalized_centers = (centers - centers.min()) / (centers.max() - centers.min() + 1e-6)
    # Map cluster id -> score in [0.6, 0.95]
    score_lookup: Dict[int, float] = {}
    for order, cluster_id in enumerate(center_rank):
        norm_value = normalized_centers[cluster_id]
        score_lookup[cluster_id] = 0.6 + 0.35 * norm_value
    models["fast_req_score"] = models["generation_cluster"].map(score_lookup)

    country_rows: List[Dict[str, object]] = []
    for _, row in models.iterrows():
        market_regions = row.get("market_regions", "")
        if isinstance(market_regions, float) and math.isnan(market_regions):
            continue
        tokens = _extract_market_tokens(str(market_regions))
        country_codes = _expand_to_countries(tokens)
        if not country_codes:
            continue
        for code in country_codes:
            country_rows.append(
                {
                    "country_code": code,
                    "fast_req_score": float(row["fast_req_score"]),
                    "first_year": int(row["first_year"]),
                }
            )

    if not country_rows:
        raise ValueError("No country-level demand rows could be inferred from EV models.")

    demand_df = pd.DataFrame(country_rows)
    demand_by_country = (
        demand_df.groupby("country_code")
        .agg(
            ev_models=("fast_req_score", "count"),
            demand_score=("fast_req_score", "mean"),
            median_launch=("first_year", "median"),
        )
        .reset_index()
    )
    global_mean = demand_by_country["demand_score"].mean()
    demand_by_country["demand_score_filled"] = demand_by_country["demand_score"]
    demand_by_country["demand_score_filled"] = demand_by_country["demand_score_filled"].fillna(global_mean)

    demand_by_country.to_csv(OUTPUT_DIR / "ev_demand_by_country.csv", index=False)
    return demand_by_country


def map_macro_region(country_code: str) -> str:
    if country_code == "US":
        return "US"
    if country_code == "GB":
        return "UK"
    if country_code == "CA":
        return "Canada"
    if country_code == "CN":
        return "China"
    if country_code == "JP":
        return "Japan"
    if country_code in EU_COUNTRIES:
        return "European_Union"
    if country_code in MIDDLE_EAST_COUNTRIES:
        return "Middle_East"
    if country_code in LATAM_COUNTRIES:
        return "Latin_America"
    return "Other"


def prepare_station_features(
    inputs: DemandInputs, demand_by_country: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    stations = inputs.stations.copy()
    country_summary = inputs.country_summary.rename(columns={"stations": "total_stations"})
    world_summary = inputs.world_summary.rename(
        columns={"count": "stations_world_count", "max_power_kw_max": "max_power_kw"}
    )

    stations = stations.merge(country_summary, on="country_code", how="left")
    stations = stations.merge(world_summary[["country_code", "country", "max_power_kw"]], on="country_code", how="left")
    stations = stations.merge(
        demand_by_country[["country_code", "demand_score_filled", "ev_models"]],
        on="country_code",
        how="left",
    )

    global_demand = demand_by_country["demand_score"].mean()
    stations["demand_score_filled"] = stations["demand_score_filled"].fillna(global_demand)
    stations["ev_models"] = stations["ev_models"].fillna(0)

    stations["macro_region"] = stations["country_code"].apply(map_macro_region)

    numeric_cols = ["latitude", "longitude", "ports", "total_stations", "max_power_kw", "demand_score_filled", "ev_models"]
    for col in numeric_cols:
        stations[col] = stations[col].fillna(stations[col].median())

    X = stations[
        [
            "latitude",
            "longitude",
            "ports",
            "total_stations",
            "max_power_kw",
            "demand_score_filled",
            "ev_models",
            "country_code",
            "macro_region",
        ]
    ]
    y = stations["is_fast_dc"].astype(int)

    return X, y


def train_fast_dc_model(X: pd.DataFrame, y: pd.Series) -> dict:
    numeric_features = ["latitude", "longitude", "ports", "total_stations", "max_power_kw", "demand_score_filled", "ev_models"]
    categorical_features = ["country_code", "macro_region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=3,
        scoring=["accuracy", "roc_auc", "balanced_accuracy"],
        n_jobs=None,
    )

    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=None)
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = sorted(
        [
            {"feature": feature_names[idx], "importance": float(importance)}
            for idx, importance in enumerate(perm.importances_mean)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    evaluation = {
        "test_classification_report": report,
        "test_roc_auc": roc_auc,
        "test_precision_fast": float(precision),
        "test_recall_fast": float(recall),
        "test_f1_fast": float(f1),
        "confusion_matrix": conf_matrix,
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_results["test_accuracy"].std()),
        "cv_roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cv_results["test_roc_auc"].std()),
        "cv_balanced_accuracy_mean": float(cv_results["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(cv_results["test_balanced_accuracy"].std()),
        "permutation_importance_top10": importances[:10],
    }

    # Fit on full data for downstream analysis
    pipeline.fit(X, y)
    evaluation["fitted_pipeline"] = pipeline
    return evaluation


def build_alignment_outputs(
    X: pd.DataFrame,
    y: pd.Series,
    demand_by_country: pd.DataFrame,
    pipeline: Pipeline,
) -> dict:
    prob_full = pipeline.predict_proba(X)[:, 1]
    stations_extended = X.copy()
    stations_extended = stations_extended.assign(
        actual_fast=y,
        predicted_fast_prob=prob_full,
    )

    country_actual = stations_extended.groupby("country_code").agg(
        fast_actual_share=("actual_fast", "mean"),
        fast_predicted_share=("predicted_fast_prob", "mean"),
        station_count=("actual_fast", "count"),
    )

    country_alignment = country_actual.merge(
        demand_by_country[["country_code", "demand_score_filled", "ev_models"]],
        left_index=True,
        right_on="country_code",
        how="left",
    )

    country_alignment["demand_score_filled"] = country_alignment["demand_score_filled"].fillna(
        demand_by_country["demand_score"].mean()
    )
    country_alignment["actual_gap"] = country_alignment["demand_score_filled"] - country_alignment["fast_actual_share"]
    country_alignment["predicted_gap"] = country_alignment["demand_score_filled"] - country_alignment["fast_predicted_share"]
    country_alignment["data_quality_flag"] = np.where(
        country_alignment["station_count"] < 50, "low_station_count", "ok"
    )
    country_alignment["confidence_level"] = np.where(
        country_alignment["station_count"] < 50, "low", "standard"
    )

    # Cluster countries into risk tiers using KMeans on gap metrics.
    clustering_features = country_alignment[["actual_gap", "predicted_gap"]].fillna(0.0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_features)
    country_alignment["risk_cluster"] = cluster_labels

    cluster_gap_means = (
        country_alignment.groupby("risk_cluster")[["actual_gap", "predicted_gap"]].mean().mean(axis=1).sort_values()
    )
    cluster_rank = {cluster_id: rank for rank, cluster_id in enumerate(cluster_gap_means.index)}
    risk_label_lookup = {0: "Low", 1: "Medium", 2: "High"}
    inverse_rank = {cluster_id: risk_label_lookup[rank] for cluster_id, rank in cluster_rank.items()}
    country_alignment["risk_level"] = country_alignment["risk_cluster"].map(inverse_rank)

    country_alignment = country_alignment.sort_values("predicted_gap", ascending=False)

    macro_alignment = (
        country_alignment.groupby("country_code")
        .first()
        .reset_index()
        .assign(macro_region=lambda df: df["country_code"].apply(map_macro_region))
        .groupby("macro_region")
        .agg(
            avg_demand=("demand_score_filled", "mean"),
            avg_actual_share=("fast_actual_share", "mean"),
            avg_predicted_share=("fast_predicted_share", "mean"),
            total_stations=("station_count", "sum"),
        )
        .reset_index()
    )
    macro_alignment["actual_gap"] = macro_alignment["avg_demand"] - macro_alignment["avg_actual_share"]
    macro_alignment["predicted_gap"] = macro_alignment["avg_demand"] - macro_alignment["avg_predicted_share"]

    highlight_markets = (
        country_alignment[country_alignment["station_count"] >= 1000]
        .head(10)[
            [
                "country_code",
                "fast_actual_share",
                "fast_predicted_share",
                "demand_score_filled",
                "actual_gap",
                "predicted_gap",
                "station_count",
                "risk_level",
            ]
        ]
    )

    country_alignment.to_csv(OUTPUT_DIR / "rq1_rq2_country_alignment.csv", index=False)
    macro_alignment.to_csv(OUTPUT_DIR / "rq1_rq2_macro_alignment.csv", index=False)
    highlight_markets.to_csv(OUTPUT_DIR / "rq1_rq2_highlight_markets.csv", index=False)

    top_under_served = country_alignment.head(15)[
        [
            "country_code",
            "fast_actual_share",
            "fast_predicted_share",
            "demand_score_filled",
            "actual_gap",
            "predicted_gap",
            "station_count",
            "risk_level",
            "confidence_level",
        ]
    ]

    return {
        "country_alignment": country_alignment,
        "macro_alignment": macro_alignment,
        "highlight_markets": highlight_markets,
        "top_under_served": top_under_served,
    }


def main() -> None:
    models = pd.read_csv(DATA_DIR / "ev_models_2025.csv")
    stations_ml = pd.read_csv(DATA_DIR / "charging_stations_2025_ml.csv")
    country_summary = pd.read_csv(DATA_DIR / "country_summary_2025.csv")
    world_summary = pd.read_csv(DATA_DIR / "world_summary_2025.csv")

    inputs = DemandInputs(
        models=models,
        stations=stations_ml,
        country_summary=country_summary,
        world_summary=world_summary,
    )

    demand_by_country = build_ev_demand(inputs)
    X, y = prepare_station_features(inputs, demand_by_country)
    evaluation = train_fast_dc_model(X, y)

    alignment = build_alignment_outputs(X, y, demand_by_country, evaluation["fitted_pipeline"])

    evaluation_output = evaluation.copy()
    evaluation_output.pop("fitted_pipeline")

    alignment_no_low = alignment["country_alignment"][
        alignment["country_alignment"]["confidence_level"] == "standard"
    ]
    corr_actual = alignment_no_low["demand_score_filled"].corr(alignment_no_low["fast_actual_share"])
    corr_predicted = alignment_no_low["demand_score_filled"].corr(alignment_no_low["fast_predicted_share"])
    global_actual_share = alignment_no_low["fast_actual_share"].mean()
    global_predicted_share = alignment_no_low["fast_predicted_share"].mean()
    global_demand = alignment_no_low["demand_score_filled"].mean()

    summary_payload = {
        "model_evaluation": evaluation_output,
        "top_under_served_countries": alignment["top_under_served"].to_dict(orient="records"),
        "validation": {
            "correlation_demand_vs_actual_fast": float(corr_actual),
            "correlation_demand_vs_pred_fast": float(corr_predicted),
            "global_average_demand": float(global_demand),
            "global_average_actual_fast": float(global_actual_share),
            "global_average_predicted_fast": float(global_predicted_share),
        },
    }

    with open(OUTPUT_DIR / "rq1_rq2_model_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print("Analysis complete. Key artifacts saved to the outputs directory.")


if __name__ == "__main__":
    main()
