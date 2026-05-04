import pandas as pd
import numpy as np
from typing import Union, List
import app.model_loader as ml


def preprocess_input(data: Union[dict, List[dict]]) -> pd.DataFrame:
    """
    Preprocess input data for model inference.

    Args:
        data: Single dict or list of dicts

    Returns:
        Preprocessed DataFrame ready for model inference
    """
    # Handle both single and batch inputs
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)

    # Selective type conversion (only required columns)
    numeric_cols = [
        "RIDAGEYR",
        "RIAGENDR",
        "BMXBMI",
        "PAQ605",
        "PAQ620",
        "SLD012",
        "INDFMMPI",
        "BPQ020",
        "DR1TKCAL",
        "DR1TSUGR",
        "DR1TTFAT",
        "DR1TPROT",
        "DR1TSODI",
        "DBD895",
        "DBD900",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Extract imputer values once
    imputer = ml.imputer
    replace_val = imputer.get("replace_value")
    epsilon = imputer.get("epsilon", 1e-6)

    # Add missing required columns
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

    # Replace special missing value markers
    if replace_val is not None:
        df = df.replace(replace_val, np.nan)

    # Efficient missing value handling with VECTORIZED operations
    impute_cols = ["DR1TKCAL", "DR1TSUGR", "DR1TTFAT", "DR1TPROT", "DR1TSODI"]
    for col in impute_cols:
        df[col + "_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(imputer[col])

    df["DBD900_missing"] = df["DBD900"].isna().astype(int)
    df["DBD900"] = df["DBD900"].fillna(imputer["DBD900"])
    df["DBD895"] = df["DBD895"].fillna(0)
    df["SLD012"] = df["SLD012"].fillna(imputer["SLD012"])
    df["INDFMMPI"] = df["INDFMMPI"].fillna(imputer["INDFMMPI"])

    # Vectorized gender mapping (faster than .map())
    df["RIAGENDR"] = df["RIAGENDR"].replace({1: 0, 2: 1}).fillna(0)

    # Feature engineering (vectorized operations work on all rows)
    calories = df["DR1TKCAL"] + epsilon

    df["protein_ratio"] = df["DR1TPROT"] / calories
    df["sugar_ratio"] = df["DR1TSUGR"] / calories
    df["sodium_density"] = df["DR1TSODI"] / calories
    df["fast_food_ratio"] = df["DBD900"] / (df["DBD895"] + 1)
    df["calorie_activity"] = df["DR1TKCAL"] * df["PAQ605"]
    df["fat_calorie_ratio"] = df["DR1TTFAT"] / calories
    df["diet_quality"] = df["protein_ratio"] - df["sugar_ratio"]
    df["log_calories"] = np.log1p(df["DR1TKCAL"].fillna(0))
    df["log_sodium"] = np.log1p(df["DR1TSODI"].fillna(0))

    # Align columns to match training schema
    df = df.reindex(columns=ml.train_columns, fill_value=0)

    return df
