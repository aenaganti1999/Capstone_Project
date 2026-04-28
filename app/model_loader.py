import joblib
from pathlib import Path

# globals (initially empty)
model = None
imputer = None
train_columns = None
threshold = None


def load_artifacts():
    global model, imputer, train_columns, threshold

    if model is None or not hasattr(model, "predict"):
        base_path = Path("artifacts")

        model = joblib.load(base_path / "model" / "model.joblib")
        imputer = joblib.load(base_path / "model" / "imputer.joblib")
        train_columns = joblib.load(base_path / "model" / "train_columns.joblib")
        threshold = joblib.load(base_path / "model" / "threshold.joblib")
