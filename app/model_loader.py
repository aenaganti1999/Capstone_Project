import joblib
from pathlib import Path

artifact_path = Path("artifacts/model")

model = joblib.load(artifact_path / "model.joblib")
#scaler = joblib.load(artifact_path / "scaler.joblib")
train_columns = joblib.load(artifact_path / "train_columns.joblib")
imputer = joblib.load(artifact_path / "imputer.joblib")
