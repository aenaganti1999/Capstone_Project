import json
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

X_TRAIN_PATH = BASE_DIR / "artifacts" / "data" / "X_train.csv"
BASELINE_PATH = BASE_DIR / "baseline_stats.json"

FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "PAQ620",
    "SLD012",
    "INDFMMPI",
    "BPQ020",
    "DR1TSUGR",
    "DR1TTFAT",
    "DR1TPROT",
    "DR1TSODI",
    "DBD895",
    "DBD900",
]

# def create_baseline_stats():

#     X_train = pd.read_csv(X_TRAIN_PATH)

#     FEATURES = X_train.columns.tolist()

#     baseline_stats = {}

#     for feature in FEATURES:

#         values = X_train[feature].dropna()

#         baseline_stats[feature] = {
#             "count": int(values.count()),
#             "mean": float(round(values.mean(), 2)),
#             "std": float(round(values.std(), 2)),
#             "min": float(round(values.min(), 2)),
#             "max": float(round(values.max(), 2)),
#         }

#     with open(BASELINE_PATH, "w") as f:
#         json.dump(baseline_stats, f, indent=2)

#     print(f"Baseline stats saved to {BASELINE_PATH}")


def create_baseline_stats():
    X_train = pd.read_csv(X_TRAIN_PATH)

    FEATURES = X_train.columns.tolist()

    print("Loaded training data")
    print(f"Rows: {len(X_train)}")
    print(f"Features: {len(FEATURES)}")

    baseline_stats = {}

    for feature in FEATURES:

        values = X_train[feature].dropna()

        histogram, bin_edges = np.histogram(
    values,
    bins=10
)

    baseline_stats[feature] = {

    "count": int(values.count()),
    "mean": float(round(values.mean(), 2)),
    "median": float(round(values.median(),2)),
    "std": float(round(values.std(), 2)),
    "min": float(round(values.min(), 2)),
    "max": float(round(values.max(), 2)),
    "histogram": histogram.tolist(),
    "values": values.tolist(),
    "bin_edges": [round(float(x), 4) for x in bin_edges]
    }

    print("Created baseline stats")
    print(f"Number of features saved: {len(baseline_stats)}")

    print("Writing file to:")
    print(BASELINE_PATH)

    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline_stats, f, indent=2)

    print("Finished writing file")


if __name__ == "__main__":
    create_baseline_stats()
