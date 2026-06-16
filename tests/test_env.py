from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL
)

print("Key Found:", OPENAI_API_KEY is not None)
print("Model:", OPENAI_MODEL)