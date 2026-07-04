import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1",
)

S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME",
    "obesity-artifacts",
)
