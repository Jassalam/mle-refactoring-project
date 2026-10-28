import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kc_cleaning import clean_raw, remove_implausible_rows  # noqa: E402


def make_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bedrooms": [3, 4, 2],
            "sqft_living": [2000, 1500, 1200],
            "sqft_above": [1500, 1500, 1000],
            "waterfront": [1.0, np.nan, 0.0],
            "view": [0.0, np.nan, 2.0],
            "yr_built": [1950, 1980, 2001],
            "yr_renovated": [0.0, np.nan, 2010.0],
        }
    )


def test_clean_raw_fills_missing_flags():
    out = clean_raw(make_raw())
    assert out["waterfront"].isna().sum() == 0
    assert out["view"].isna().sum() == 0


def test_clean_raw_last_known_change_uses_renovation_or_build_year():
    raw = make_raw()
    out = clean_raw(raw)
    assert out.loc[0, "last_known_change"] == 1950  # renovation 0 -> build year
    assert out.loc[1, "last_known_change"] == 1980  # renovation missing -> build year
    assert out.loc[2, "last_known_change"] == 2010  # renovation kept


def test_clean_raw_recomputes_basement_and_drops_year_columns():
    raw = make_raw()
    out = clean_raw(raw)
    assert (out["sqft_basement"] == raw["sqft_living"] - raw["sqft_above"]).all()
    assert "yr_built" not in out.columns
    assert "yr_renovated" not in out.columns


def test_clean_raw_does_not_mutate_input():
    raw = make_raw()
    before = raw.copy()
    clean_raw(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_remove_implausible_rows_drops_33_bedrooms():
    df = pd.DataFrame({"bedrooms": [3, 33, 4]})
    assert list(remove_implausible_rows(df)["bedrooms"]) == [3, 4]