from app.llm import build_prompt

top_factors = [
    {"feature": "Age", "value": 45, "impact": 0.24},
    {"feature": "High Blood Pressure", "value": 1, "impact": 0.57},
]

prompt = build_prompt(
    prediction=1, probability=0.61, top_factors=top_factors, input_data={}
)

print(prompt)
