"""Cleaning steps for the raw King County house sales data.

Refactored from the data preparation section of King-County.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bedroom count of the implausible record in the dataset (33 bedrooms, 2 bathrooms).
IMPLAUSIBLE_BEDROOMS = 33


def remove_implausible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records that are almost certainly data-entry errors.

    Apply this to training data only. Do not call it at inference time.
    """
    return df[df["bedrooms"] != IMPLAUSIBLE_BEDROOMS]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw columns without changing the input.

    - ``sqft_basement`` is recomputed as ``sqft_living - sqft_above``, because
      the raw column contains a ``?`` entry.
    - Missing ``waterfront`` and ``view`` values are filled with 0.
    - ``last_known_change`` is the renovation year, or the build year when
      ``yr_renovated`` is missing or 0. ``yr_built`` and ``yr_renovated``
      are then dropped.
    """
    df = df.copy()

    df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]
    df["waterfront"] = df["waterfront"].fillna(0)
    df["view"] = df["view"].fillna(0)

    renovated = df["yr_renovated"].fillna(0)
    df["last_known_change"] = np.where(renovated == 0, df["yr_built"], renovated).astype(int)

    return df.drop(columns=["yr_renovated", "yr_built"])