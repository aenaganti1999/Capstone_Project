import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocess import preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_PATH = BASE_DIR / "baseline_stats.json"
PREDICTIONS_PATH = BASE_DIR / "predictions.json"

LOW_SHIFT_THRESHOLD = 10
MEDIUM_SHIFT_THRESHOLD = 25


def load_predictions():
    with open(PREDICTIONS_PATH, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def extract_production_features(df: pd.DataFrame) -> dict:
    """
    Extract all production feature values after preprocessing.
    """

    production_features = {}

    for row in df["input_data"]:

        if not isinstance(row, dict):
            continue

        processed = preprocess_input(row)
        processed_dict = processed.iloc[0].to_dict()

        for feature, value in processed_dict.items():

            production_features.setdefault(feature, []).append(float(value))

    return production_features


def generate_monitoring_report():

    df = load_predictions()
    df = df[df["input_data"].notna()]

    total_predictions = len(df)

    production_features = extract_production_features(df)

    feature_stats = {}

    for feature, values in production_features.items():

        feature_stats[feature] = {
            "count": len(values),
            "mean": round(float(np.mean(values)), 2),
            "std": round(float(np.std(values)), 2),
            "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2),
        }

    positive_rate = (df["prediction"] == 1).mean()
    negative_rate = (df["prediction"] == 0).mean()

    avg_probability = df["probability"].mean()

    avg_latency = df["latency_seconds"].mean()
    p95_latency = df["latency_seconds"].quantile(0.95)
    max_latency = df["latency_seconds"].max()

    prediction_counts = (
        df["prediction"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {

        "total_predictions": total_predictions,

        "positive_rate": float(positive_rate),

        "negative_rate": float(negative_rate),

        "average_probability": float(avg_probability),

        "average_latency_seconds": float(avg_latency),

        "p95_latency_seconds": float(p95_latency),

        "max_latency_seconds": float(max_latency),

        "prediction_distribution": prediction_counts,

        "feature_stats": feature_stats,
    }


def generate_drift_report():

    df = load_predictions()
    df = df[df["input_data"].notna()]

    with BASELINE_PATH.open("r") as f:
        baseline_stats = json.load(f)

    production_features = extract_production_features(df)

    feature_report = {}

    low_shift = 0
    medium_shift = 0
    high_shift = 0

    for feature, train_stats in baseline_stats.items():

        if feature not in production_features:
            continue

        production_values = production_features[feature]

        train_mean = train_stats["mean"]
        train_std = train_stats["std"]
        train_min = train_stats["min"]
        train_max = train_stats["max"]

        prod_mean = float(np.mean(production_values))
        prod_std = float(np.std(production_values))
        prod_min = float(np.min(production_values))
        prod_max = float(np.max(production_values))

        mean_shift = prod_mean - train_mean

        mean_shift_percent = (
            abs(mean_shift)
            / max(abs(train_mean), 1e-6)
        ) * 100

        if mean_shift_percent < LOW_SHIFT_THRESHOLD:

            status = "LOW"
            low_shift += 1

        elif mean_shift_percent < MEDIUM_SHIFT_THRESHOLD:

            status = "MEDIUM"
            medium_shift += 1

        else:

            status = "HIGH"
            high_shift += 1

        feature_report[feature] = {

            "training_mean": round(train_mean, 2),
            "production_mean": round(prod_mean, 2),

            "training_std": round(train_std, 2),
            "production_std": round(prod_std, 2),

            "training_min": round(train_min, 2),
            "production_min": round(prod_min, 2),

            "training_max": round(train_max, 2),
            "production_max": round(prod_max, 2),

            "mean_shift": round(mean_shift, 2),

            "mean_shift_percent": round(mean_shift_percent, 2),

            "status": status,
        }

    return {

        "summary": {

            "total_features": len(feature_report),

            "high_shift": high_shift,

            "medium_shift": medium_shift,

            "low_shift": low_shift,
        },

        "features": feature_report,
    }