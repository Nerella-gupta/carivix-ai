from __future__ import annotations

import pandas as pd


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight dtype downcasting to reduce memory usage."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    optimized = df.copy()
    for column in optimized.columns:
        series = optimized[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_integer_dtype(series):
            if series.isna().any():
                continue
            downcast = pd.to_numeric(series, downcast="integer")
            optimized[column] = downcast
        elif pd.api.types.is_float_dtype(series):
            if series.isna().any():
                continue
            optimized[column] = pd.to_numeric(series, downcast="float")
    return optimized
