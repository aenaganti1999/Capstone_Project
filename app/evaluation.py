import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import app.model_loader as ml

BASE_DIR = Path(__file__).resolve().parent.parent

X_TEST_PATH = BASE_DIR / "artifacts" / "data" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "artifacts" / "data" / "y_test.csv"
EVALUATION_PATH = BASE_DIR / "evaluation_report.json"


def generate_evaluation_report():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    model = ml.model
    threshold = ml.threshold

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    report = {
        "threshold": float(threshold),
        "total_samples": int(len(y_test)),
        "accuracy": float(round(accuracy_score(y_test, predictions), 4)),
        "precision": float(
            round(precision_score(y_test, predictions, zero_division=0), 4)
        ),
        "recall": float(round(recall_score(y_test, predictions, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, predictions, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, probabilities), 4)),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    with open(EVALUATION_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = generate_evaluation_report()
    print(json.dumps(report, indent=2))
