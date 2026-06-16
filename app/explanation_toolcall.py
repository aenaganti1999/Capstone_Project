def build_prompt(
    prediction,
    probability,
    top_factors
):

    factors = ""

    for item in top_factors:

        factors += (
            f"- {item.feature}: "
            f"value={item.value}, "
            f"impact={item.impact}\n"
        )

    return f"""
You are helping explain an obesity prediction model.

Prediction: {prediction}

Probability: {probability:.2%}

Top factors:

{factors}

Explain:
1. Why the prediction was made.
2. Which factors increased risk.
3. Which factors decreased risk.
4. Use plain English.
5. Do not provide medical advice.
6. Maximum 150 words.
"""