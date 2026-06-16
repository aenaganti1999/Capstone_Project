import pandas as pd


FEATURE_MAP = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "PAQ620": "Physical Activity",
    "SLD012": "Sleep Duration",
    "INDFMMPI": "Family Income",
    "BPQ020": "High Blood Pressure",
    "DBD895": "Daily Calories",
    "DBD900": "Daily Sodium Intake",
    "protein_ratio": "Protein Ratio",
    "sugar_ratio": "Sugar Ratio",
    "fast_food_ratio": "Fast Food Consumption",
    "fat_calorie_ratio": "Calories From Fat"
}


def get_top_factors(
    processed_df,
    explainer,
    top_k=5
):

    shap_values = explainer.shap_values(
        processed_df
    )

    contributions = []

    for feature, impact in zip(
        processed_df.columns,
        shap_values[0]
    ):

        contributions.append(
            {
                "feature": FEATURE_MAP.get(
                    feature,
                    feature
                ),
                "value": float(
                    processed_df.iloc[0][feature]
                ),
                "impact": float(impact)
            }
        )

    contributions.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return contributions[:top_k]