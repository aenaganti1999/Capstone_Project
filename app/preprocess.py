import pandas as pd
import numpy as np
import app.model_loader as ml


def preprocess_input(data: dict):
    df = pd.DataFrame([data])#overhead of creating a dataframe for each record, instead we can create a dataframe for the entire batch and preprocess it together. This would be more efficient.
    df = df.apply(pd.to_numeric, errors="coerce")
    required_cols = [
        "DR1TKCAL",
        "DR1TSUGR",
        "DR1TTFAT",
        "DR1TPROT",
        "DR1TSODI",
        "DBD900",
        "DBD895",
        "SLD012",
        "INDFMMPI",
        "PAQ605",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Replace weird values
    replace_val = ml.imputer.get("replace_value")

    if replace_val is not None:
        df = df.replace(replace_val, np.nan)

    # Missing value handling
    for col in ["DR1TKCAL", "DR1TSUGR", "DR1TTFAT", "DR1TPROT", "DR1TSODI"]:
        df[col + "_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(ml.imputer[col])

    df["DBD900_missing"] = df["DBD900"].isna().astype(int)
    df["DBD900"] = df["DBD900"].fillna(ml.imputer["DBD900"])
    df["DBD895"] = df["DBD895"].fillna(0)
    df["SLD012"] = df["SLD012"].fillna(ml.imputer["SLD012"])
    df["INDFMMPI"] = df["INDFMMPI"].fillna(ml.imputer["INDFMMPI"])

    # Transformations
    df["RIAGENDR"] = df["RIAGENDR"].map({1: 0, 2: 1}).fillna(0)

    # Feature engineering
    eps = ml.imputer.get("epsilon", 1e-6)

    calories = df["DR1TKCAL"] + eps

    df["protein_ratio"] = df["DR1TPROT"] / calories
    df["sugar_ratio"] = df["DR1TSUGR"] / calories
    df["sodium_density"] = df["DR1TSODI"] / calories
    df["fast_food_ratio"] = df["DBD900"] / (df["DBD895"] + 1)
    df["calorie_activity"] = df["DR1TKCAL"] * df["PAQ605"]
    df["fat_calorie_ratio"] = df["DR1TTFAT"] / calories
    df["diet_quality"] = df["protein_ratio"] - df["sugar_ratio"]
    df["log_calories"] = np.log1p(df["DR1TKCAL"].fillna(0))
    df["log_sodium"] = np.log1p(df["DR1TSODI"].fillna(0))

    # Align columns
    df = df.reindex(columns=ml.train_columns, fill_value=0)

    return df
