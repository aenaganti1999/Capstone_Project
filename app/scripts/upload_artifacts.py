from pathlib import Path

from app.services.s3_services import S3Service

s3 = S3Service()

ARTIFACTS = {
    Path("artifacts/model/model.joblib"): "2026-07-01/models/model.joblib",
    Path("artifacts/model/imputer.joblib"): "2026-07-01/models/imputer.joblib",
    Path(
        "artifacts/model/train_columns.joblib"
    ): "2026-07-01/models/train_columns.joblib",
    Path("artifacts/model/threshold.joblib"): "2026-07-01/models/threshold.joblib",
    Path("artifacts/data/X_train.csv"): "2026-07-01/explainability/X_train.csv",
    Path("baseline_stats.json"): "2026-07-01/monitoring/baseline_stats.json",
    Path("evaluation_report.json"): "2026-07-01/evaluation/evaluation_report.json",
    Path("predictions.json"): "2026-07-01/monitoring/predictions.json",
}

for local_path, s3_key in ARTIFACTS.items():
    s3.upload_file(local_path, s3_key)
