from pathlib import Path
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import (
    AWS_REGION,
    S3_BUCKET_NAME,
)

logger = logging.getLogger(__name__)


class S3Service:

    def __init__(self):

        self.bucket = S3_BUCKET_NAME

        self.client = boto3.client(
            "s3",
            region_name=AWS_REGION,
        )

    def upload_file(
        self,
        local_path: Path,
        s3_key: str,
    ) -> None:

        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        try:

            self.client.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
            )

            logger.info(
                "Uploaded %s to %s",
                local_path,
                s3_key,
            )

        except NoCredentialsError:
            logger.exception("AWS credentials not configured.")

            raise

        except ClientError:
            logger.exception(
                "Failed to upload %s",
                local_path,
            )

            raise

    def download_file(
        self,
        s3_key: str,
        destination: Path,
    ) -> None:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            self.client.download_file(
                self.bucket,
                s3_key,
                str(destination),
            )

            logger.info(
                "Downloaded %s",
                s3_key,
            )

        except NoCredentialsError:
            logger.exception("AWS credentials not configured.")

            raise

        except ClientError:
            logger.exception(
                "Unable to download %s",
                s3_key,
            )

            raise
