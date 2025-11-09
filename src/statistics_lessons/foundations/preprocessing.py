"""Data cleaning and preprocessing utilities."""

from typing import Iterable, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


def drop_missing(
    data: pd.DataFrame, columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Remove rows with missing values.

    Args:
        data (pd.DataFrame): DataFrame to clean.
        columns (Optional[Iterable[str]], optional): Specific columns to inspect for
            missing values. If ``None``, all columns are considered.

    Returns:
        pd.DataFrame: Cleaned DataFrame without rows containing ``NaN``.

    Raises:
        ValueError: If any specified column is not present in ``data``.
    """
    if columns is not None:
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    # When columns is None, check all columns
    subset = list(columns) if columns is not None else None
    # ``dropna`` removes rows with any missing values in the subset
    return data.dropna(subset=subset)


def standardize(series: pd.Series) -> pd.Series:
    """Standardize a numeric series to mean 0 and standard deviation 1.

    Args:
        series (pd.Series): Numeric column to standardize.

    Returns:
        pd.Series: Standardized values with mean 0 and standard deviation 1.

    Raises:
        ValueError: If ``series`` is empty.
        TypeError: If ``series`` is not numeric.
    """
    if series.empty:
        raise ValueError("Series must not be empty.")
    if not is_numeric_dtype(series):
        raise TypeError("Series must contain numeric values.")

    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        # Avoid division by zero when the series is constant
        return series - mean_val
    return (series - mean_val) / std_val


if __name__ == "__main__":
    # Basic demonstration
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [5, 6, 7, 8]})
    cleaned = drop_missing(df)
    print(cleaned)
    standardized = standardize(cleaned["a"])
    print(standardized)
