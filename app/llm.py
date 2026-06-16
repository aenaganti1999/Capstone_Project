from app.explanation_toolcall import build_prompt
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(
    api_key=OPENAI_API_KEY

)

def generate_explanation(
    prediction,
    probability,
    top_factors
):

    prompt = build_prompt(
        prediction,
        probability,
        top_factors
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
    )