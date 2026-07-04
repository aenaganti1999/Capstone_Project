import joblib
from pathlib import Path
import shap
from app.services.artifact_service import download_required_artifacts

# globals (initially empty)
model = None
imputer = None
train_columns = None
threshold = None
explainer = None


def load_artifacts():
    global model, imputer, train_columns, threshold, explainer

    if model is None or not hasattr(model, "predict"):

        download_required_artifacts()

        base_path = Path("artifacts")

        model = joblib.load(base_path / "model" / "model.joblib")
        imputer = joblib.load(base_path / "model" / "imputer.joblib")
        train_columns = joblib.load(base_path / "model" / "train_columns.joblib")
        threshold = joblib.load(base_path / "model" / "threshold.joblib")
        explainer = shap.TreeExplainer(model)
