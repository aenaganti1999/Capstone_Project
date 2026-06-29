# test_openai.py

from openai import OpenAI
from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4.1-mini", messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message.content)
