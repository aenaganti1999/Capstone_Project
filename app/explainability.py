FEATURE_MAP = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "PAQ620": "Physical Activity",
    "SLD012": "Sleep Duration",
    "INDFMMPI": "Family Income",
    "BPQ020": "High Blood Pressure",
    "DR1TKCAL": "Daily Calories",
    "DR1TSODI": "Daily Sodium Intake",
    "DR1TSUGR": "Daily Sugar Intake",
    "DR1TPROT": "Daily Protein Intake",
    "DR1TTFAT": "Daily Fat Intake",
    "DBD895": "Meals Not Prepared at Home",
    "protein_ratio": "Protein Ratio",
    "sugar_ratio": "Sugar Ratio",
    "fast_food_ratio": "Fast Food Consumption",
    "fat_calorie_ratio": "Calories From Fat",
    "sodium_density": "Sodium Density",
}


def get_top_factors(processed_df, explainer, top_k=5):

    shap_values = explainer.shap_values(processed_df)

    contributions = []

    for feature, impact in zip(processed_df.columns, shap_values[0]):

        contributions.append(
            {
                "feature": FEATURE_MAP.get(feature, feature.replace("_", " ").title()),
                "value": float(processed_df.iloc[0][feature]),
                "impact": float(impact),
            }
        )

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return contributions[:top_k]
