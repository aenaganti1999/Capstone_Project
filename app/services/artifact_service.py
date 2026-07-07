from pathlib import Path
import logging

from app.services.s3_services import S3Service

logger = logging.getLogger(__name__)

ARTIFACTS = {
    "2026-07-01/models/model.joblib": Path("artifacts/model/model.joblib"),
    "2026-07-01/models/imputer.joblib": Path("artifacts/model/imputer.joblib"),
    "2026-07-01/models/train_columns.joblib": Path(
        "artifacts/model/train_columns.joblib"
    ),
    "2026-07-01/models/threshold.joblib": Path("artifacts/model/threshold.joblib"),
    "2026-07-01/explainability/X_train.csv": Path("artifacts/data/X_train.csv"),
    "2026-07-01/monitoring/baseline_stats.json": Path("baseline_stats.json"),
    "2026-07-01/evaluation/evaluation_report.json": Path("evaluation_report.json"),
    "2026-07-01/monitoring/predictions.json": Path("predictions.json"),
}


def download_required_artifacts() -> None:
    """
    Downloads all required model artifacts from S3
    if they are not already present locally.
    """

    s3 = S3Service()

    for s3_key, local_path in ARTIFACTS.items():

        if local_path.exists():

            logger.info(
                "Artifact already exists: %s",
                local_path,
            )

            continue

        logger.info(
            "Downloading %s",
            s3_key,
        )

        s3.download_file(
            s3_key,
            local_path,
        )

    logger.info("All required artifacts are available.")
