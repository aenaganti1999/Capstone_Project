import json
import pandas as pd
import numpy as np
from pathlib import Path
from app.preprocess import preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent
baseline_path = BASE_DIR / "baseline_stats.json"


def load_predictions():
    with open(BASE_DIR / "predictions.json", "r") as f:
        data = json.load(f)

    print(data[0])

    return pd.DataFrame(data)


def generate_monitoring_report():

    df = load_predictions()

    df = df[df["input_data"].notna()]

    total_predictions = len(df)

    feature_stats = {}

    features = [
        "RIDAGEYR",
        "RIAGENDR",
        "BMXBMI",
        "PAQ605",
        "PAQ620",
        "SLD012",
        "INDFMMPI",
        "BPQ020",
        "DR1TKCAL",
        "DR1TSUGR",
        "DR1TTFAT",
        "DR1TPROT",
        "DR1TSODI",
        "DBD895",
        "DBD900",
    ]

    for feature in features:

        values = []

        for row in df["input_data"]:

            if not isinstance(row, dict):

                continue

            value = row.get(feature)

            if value is not None:

                values.append(float(value))

        if values:

            feature_stats[feature] = {
                "count": int(len(values)),
                "mean": float(round(np.mean(values), 2)),
                "std": float(round(np.std(values), 2)),
                "min": float(round(np.min(values), 2)),
                "max": float(round(np.max(values), 2)),
            }

    positive_rate = (df["prediction"] == 1).mean()

    negative_rate = (df["prediction"] == 0).mean()

    avg_probability = df["probability"].mean()

    avg_latency = df["latency_seconds"].mean()

    p95_latency = df["latency_seconds"].quantile(0.95)

    max_latency = df["latency_seconds"].max()

    prediction_counts = df["prediction"].astype(str).value_counts().to_dict()

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
    print(Path("baseline_stats.json").resolve())

    with baseline_path.open("r") as f:

        content = f.read()

        print("content:")
        print(repr(content))

        baseline_stats = json.loads(content)

    production_features = {}

    for row in df["input_data"]:

        if not isinstance(row, dict):
            continue

        processed = preprocess_input(row)

        processed_dict = processed.iloc[0].to_dict()

        for feature, value in processed_dict.items():

            production_features.setdefault(feature, []).append(float(value))

    drift_report = {}

    high_drift = 0
    medium_drift = 0
    low_drift = 0

    for feature in baseline_stats:

        if feature not in production_features:
            continue

        train_mean = baseline_stats[feature]["mean"]

        prod_mean = np.mean(production_features[feature])

        drift_pct = (abs(prod_mean - train_mean) / max(abs(train_mean), 0.0001)) * 100

        if drift_pct < 10:
            status = "LOW"

        elif drift_pct < 25:
            status = "MEDIUM"

        else:
            status = "HIGH"

        if status == "HIGH":
            high_drift += 1

        elif status == "MEDIUM":
            medium_drift += 1

        else:
            low_drift += 1

        drift_report[feature] = {
            "training_mean": round(float(train_mean), 2),
            "production_mean": round(float(prod_mean), 2),
            "drift_percent": round(float(drift_pct), 2),
            "status": status,
        }

    return {
        "summary": {
            "total_features": len(drift_report),
            "high_drift": high_drift,
            "medium_drift": medium_drift,
            "low_drift": low_drift,
        },
        "features": drift_report,
    }
