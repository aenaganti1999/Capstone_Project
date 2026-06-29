from app.explanation_toolcall import build_prompt
from openai import OpenAI

from app.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_explanation(prediction, probability, top_factors, input_data):
    prompt = build_prompt(
        prediction=prediction,
        probability=probability,
        top_factors=top_factors,
        input_data=input_data,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
