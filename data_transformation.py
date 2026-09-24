from __future__ import annotations

import os
import re
from typing import Iterable, Sequence

import pandas as pd


class DataCleaner:
    """Normalize field names and basic row-level cleaning."""

    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = [
            re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower()).strip("_")
            for col in cleaned.columns
        ]
        return cleaned

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)


class DataTransformer:
    """Apply lightweight feature transformation helpers."""

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: Sequence[str] | None = None, method: str = "onehot") -> pd.DataFrame:
        if columns is None:
            columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                continue
            normalized = result[column].astype(str).str.strip().str.lower()
            result[column] = normalized
            dummies = pd.get_dummies(result[column], prefix=column, dtype=int)
            result = pd.concat([result.drop(columns=[column]), dummies], axis=1)
        return result

    @staticmethod
    def normalize_zscore(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                continue
            series = pd.to_numeric(result[column], errors="coerce")
            mean = series.mean()
            std = series.std(ddof=0)
            if pd.isna(std) or std == 0:
                normalized = 0.0
            else:
                normalized = (series - mean) / std
            result[column] = normalized.fillna(0.0)
        return result


class DataProcessingEngine:
    """Simple end-to-end processing pipeline used by project tests."""

    def __init__(self, missing_strategy: str = "median"):
        self.missing_strategy = missing_strategy

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        result = df.copy()
        for column in columns or []:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame, columns: Sequence[str] | None = None, strategy: str = "median") -> pd.DataFrame:
        result = df.copy()
        for column in columns or []:
            if column not in result.columns:
                continue
            series = result[column]
            if pd.api.types.is_numeric_dtype(series):
                if strategy == "median":
                    fill_value = series.median()
                elif strategy == "mean":
                    fill_value = series.mean()
                else:
                    fill_value = series.mode().iloc[0] if not series.mode().empty else 0
                result[column] = series.fillna(fill_value)
            else:
                fill_value = series.mode().iloc[0] if not series.mode().empty else "unknown"
                result[column] = series.fillna(fill_value).astype(str).str.strip().str.lower()
        return result

    def run(
        self,
        df: pd.DataFrame,
        *,
        dedup_subset: Sequence[str] | None = None,
        normalize_cols: Sequence[str] | None = None,
        date_col: str | None = None,
        categorical_cols: Sequence[str] | None = None,
        schema: dict | None = None,
    ) -> pd.DataFrame:
        if df is None:
            raise ValueError("df cannot be None")
        result = df.copy()

        result = DataCleaner.standardize_columns(result)
        normalized_dedup = None
        if dedup_subset is not None:
            normalized_dedup = [str(col).strip().lower().replace(" ", "_") for col in dedup_subset]
            normalized_dedup = [col for col in normalized_dedup if col in result.columns]
            if normalized_dedup:
                result = result.drop_duplicates(subset=normalized_dedup).reset_index(drop=True)
            else:
                result = result.drop_duplicates().reset_index(drop=True)
        else:
            result = result.drop_duplicates().reset_index(drop=True)

        if date_col is not None:
            normalized_date = str(date_col).strip().lower().replace(" ", "_")
            if normalized_date in result.columns:
                result[normalized_date] = pd.to_datetime(result[normalized_date], errors="coerce").dt.strftime("%Y-%m-%d")

        # Fill missing values in target numeric columns.
        if normalize_cols is not None:
            normalized_cols = [str(col).strip().lower() for col in normalize_cols]
            result = self._coerce_numeric(result, normalized_cols)
            result = self._fill_missing_values(result, normalized_cols, strategy=self.missing_strategy)
            result = DataTransformer.normalize_zscore(result, normalized_cols)

        if categorical_cols is not None:
            normalized_cats = [str(col).strip().lower() for col in categorical_cols]
            result = self._fill_missing_values(result, normalized_cats, strategy=self.missing_strategy)
            result = DataTransformer.encode_categorical(result, columns=normalized_cats, method="onehot")

        if schema:
            # Apply an optional schema to ensure required columns exist.
            for column, dtype in schema.items():
                if column in result.columns and hasattr(result[column], "astype"):
                    try:
                        result[column] = result[column].astype(dtype)
                    except Exception:
                        pass

        return result.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: str, fmt: str = "parquet") -> None:
    """Persist processed data to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt.lower() == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
