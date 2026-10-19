"""Data loading module for TSV challenge datasets."""

import os
from pathlib import Path
from typing import Generator, List, Optional, Union
import pandas as pd

from ..config import ColumnMapping, DEFAULT_CONFIG


class TSVDataLoader:
    """Loads and validates TSV datasets for Business Entity Resolution."""

    def __init__(self, column_mapping: Optional[ColumnMapping] = None):
        self.columns = column_mapping or DEFAULT_CONFIG.columns

    def load_tsv(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        chunk_size: Optional[int] = None,
        sep: str = "\t",
    ) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Loads a TSV file with error handling and missing-data standardization.

        Args:
            file_path: Path to the TSV file.
            encoding: Text encoding (defaults to utf-8, falls back to latin-1 if needed).
            chunk_size: If specified, yields DataFrames in chunks.
            sep: Delimiter character (defaults to '\\t').

        Returns:
            pandas DataFrame or chunk generator.

        Raises:
            FileNotFoundError: If the specified file cannot be found.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found at: {path.resolve()}\n"
                f"Please place your challenge TSV file in 'data/raw/' and update the path."
            )

        try:
            if chunk_size:
                return pd.read_csv(
                    path,
                    sep=sep,
                    encoding=encoding,
                    dtype=str,
                    chunksize=chunk_size,
                    on_bad_lines="skip",
                    low_memory=False,
                )
            else:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=encoding,
                    dtype=str,
                    on_bad_lines="skip",
                    low_memory=False,
                )
                return self._standardize_missing_values(df)
        except UnicodeDecodeError:
            # Fallback for datasets with legacy or non-UTF8 characters
            if chunk_size:
                return pd.read_csv(
                    path,
                    sep=sep,
                    encoding="latin-1",
                    dtype=str,
                    chunksize=chunk_size,
                    on_bad_lines="skip",
                    low_memory=False,
                )
            else:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding="latin-1",
                    dtype=str,
                    on_bad_lines="skip",
                    low_memory=False,
                )
                return self._standardize_missing_values(df)

    def _standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing text representations with empty strings or NaN consistently."""
        df = df.fillna("")
        # Clean potential string representations of nulls
        null_markers = {"nan", "null", "none", "n/a", "na", "<null>", "nil"}
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: "" if x.lower() in null_markers else x)
        return df

    def validate_columns(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        allow_missing: bool = True,
    ) -> List[str]:
        """
        Validates the presence of expected columns in the loaded dataset.

        Args:
            df: The loaded dataframe.
            expected_columns: List of column names that should ideally be present.
            allow_missing: If True, logs a warning and returns missing columns;
                           if False, raises ValueError.

        Returns:
            List of missing columns.
        """
        missing = [c for c in expected_columns if c not in df.columns]
        if missing and not allow_missing:
            raise ValueError(
                f"Missing required columns in dataset: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )
        return missing

    def inspect_dataset(self, df: pd.DataFrame) -> dict:
        """Returns diagnostic metadata about the loaded dataset without leaking records."""
        return {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": list(df.columns),
            "empty_value_ratios": {
                col: round((df[col] == "").mean(), 4) for col in df.columns
            },
        }
