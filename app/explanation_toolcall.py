def build_prompt(prediction, probability, top_factors, input_data):
    factor_lines = []

    for factor in top_factors:
        if isinstance(factor, dict):
            feature = factor["feature"]
            value = factor["value"]
            impact = factor["impact"]
        else:
            feature = factor.feature
            value = factor.value
            impact = factor.impact

        factor_lines.append(f"- {feature}: value={value}, impact={impact:.4f}")

    factors = "\n".join(factor_lines)

    return f"""

    You are explaining an obesity prediction model.

    Prediction: {prediction}
    Probability: {probability:.2%}

    Top factors from SHAP:
    {factors}

    Input features:
    {input_data}

    Rules:
    - Positive impact values increased the model's obesity prediction.
    - Negative impact values decreased the model's obesity prediction.
    - Use only the feature names shown in Top factors.
    - Do not mention raw column names like
        - RIDAGEYR,
        - DBD895,
        - DR1TKCAL,
        - BPQ020, or
        -sodium_density.
    - Do not say a factor is protective. Say it decreased the model's prediction.
    - Do not give medical advice.
    - Do not recommend treatments, diets, or lifestyle changes.
    - Do not directly use features like
        - DBD895 or
        - DR1TKCAL in the explanation.
        - Instead, say "Meals Not Prepared at Home" or "Daily Calories".
    - Use plain English.

    - Maximum 120 words.

    Explain:
    1. Why the model made this prediction.
    2. Which factors increased the prediction.
    3. Which factors decreased the prediction.
    4. Mention the top 3 increasing factors and
    5. top 2 decreasing factors by name.
    - Do not replace feature names with vague phrases such as
    - "body composition measurement" or "other factors".
   """
