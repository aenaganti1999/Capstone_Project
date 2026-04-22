import joblib
from pathlib import Path

artifact_path = Path("artifacts/model")

model = joblib.load(artifact_path / "model.joblib")
train_columns = joblib.load(artifact_path / "train_columns.joblib")
imputer = joblib.load(artifact_path / "imputer.joblib")
threshold = joblib.load(artifact_path / "threshold.joblib")
