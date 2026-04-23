import pandas as pd
import numpy as np
from .model_loader import train_columns, imputer


def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # Replace weird values
    df = df.replace(imputer["replace_value"], np.nan)

    # Missing value handling
    for col in ['DR1TKCAL','DR1TSUGR','DR1TTFAT','DR1TPROT','DR1TSODI']:
        df[col + '_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(imputer[col])   

    df['DBD900_missing'] = df['DBD900'].isna().astype(int)
    df['DBD900'] = df['DBD900'].fillna(imputer["DBD900"])
    df["SLD012"] = df["SLD012"].fillna(imputer["SLD012"])
    df["INDFMMPI"] = df["INDFMMPI"].fillna(imputer["INDFMMPI"])

    # Fix dtype issues (VERY IMPORTANT)
    numeric_cols = ['DR1TKCAL','DR1TSUGR','DR1TTFAT','DR1TPROT','DR1TSODI',
    'SLD012','INDFMMPI','DBD900']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Transformations
    df["RIAGENDR"] = df["RIAGENDR"].map({1: 0, 2: 1})

    # Feature engineering
    eps = imputer.get("epsilon", 1e-6)

    calories = df['DR1TKCAL'] + eps

    df['protein_ratio'] = df['DR1TPROT'] / calories
    df['sugar_ratio'] = df['DR1TSUGR'] / calories
    df['sodium_density'] = df['DR1TSODI'] / calories
    df['fast_food_ratio'] = df['DBD900'] / (df['DBD895'] + 1)
    df['calorie_activity'] = df['DR1TKCAL'] * df['PAQ605']
    df['fat_calorie_ratio'] = df['DR1TTFAT'] / calories
    df['diet_quality'] = df['protein_ratio'] - df['sugar_ratio']
    df['log_calories'] = np.log1p(df['DR1TKCAL'])
    df['log_sodium'] = np.log1p(df['DR1TSODI'])

    # Align columns
    df = df.reindex(columns=train_columns, fill_value=0)
    
    return df