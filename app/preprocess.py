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

    df["ALQ111"] = df["ALQ111"].fillna(imputer["ALQ111_fill"])
    df["SLD012"] = df["SLD012"].fillna(imputer["SLD012"])
    df["INDFMMPI"] = df["INDFMMPI"].fillna(imputer["INDFMMPI"])

    
    # Transformations
    df["RIAGENDR"] = df["RIAGENDR"].map({1: 0, 2: 1})
    df["ALQ111"] = (df["ALQ111"] == 1).astype(int)


    # Feature engineering (safe division)
    eps = 1e-6

    df['protein_ratio'] = df['DR1TPROT'] / (df['DR1TKCAL'] + eps)
    df['sugar_ratio'] = df['DR1TSUGR'] / (df['DR1TKCAL'] + eps)
    df['sodium_density'] = df['DR1TSODI'] / (df['DR1TKCAL'] + eps)
    df['fast_food_ratio'] = df['DBD900'] / (df['DBD895'] + 1)
    df['calorie_activity'] = df['DR1TKCAL'] * df['PAQ605']
    df['fat_calorie_ratio'] = df['DR1TTFAT'] / (df['DR1TKCAL'] + eps)
    df['diet_quality'] = df['protein_ratio'] - df['sugar_ratio']
    df['log_calories'] = np.log1p(df['DR1TKCAL'])
    df['log_sodium'] = np.log1p(df['DR1TSODI'])

    
    # Align columns
    df = df.reindex(columns=train_columns, fill_value=0)

    
    # Scaling (if used)
    # if 'scaler' in globals():
    #     df = scaler.transform(df)

    return df