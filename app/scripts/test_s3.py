from pathlib import Path

from app.services.s3_services import S3Service

s3 = S3Service()

s3.upload_file(
    Path("baseline_stats.json"),
    "monitoring/baseline_stats.json",
)

print("Upload successful!")
