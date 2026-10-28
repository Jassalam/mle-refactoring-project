"""Feature engineering for the King County house sales data.

Refactored from the feature engineering section of King-County.ipynb.
"""

#from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.kc_cleaning import clean_raw

# Reference point used as a rough "centre of wealth" (notebook: Medina, WA).
CENTER_LAT = 47.62774
CENTER_LONG = -122.24194

# Latitude used for the cos() longitude correction (kept from the notebook).
LAT_CORRECTION = 47.6219

# Approximate km per degree of latitude on a sphere of radius 6378 km.
KM_PER_DEGREE = 2 * np.pi * 6378 / 360


def _km_distance(delta_lat, delta_long):
    """Equirectangular distance in km between points given their coordinate deltas."""
    delta_long_corr = delta_long * np.cos(np.radians(LAT_CORRECTION))
    return np.sqrt(delta_long_corr**2 + delta_lat**2) * KM_PER_DEGREE


def _project(lat_long: np.ndarray) -> np.ndarray:
    """Map (lat, long) to a plane where Euclidean distance matches the km formula."""
    lat = lat_long[:, 0]
    long = lat_long[:, 1]
    return np.column_stack([long * np.cos(np.radians(LAT_CORRECTION)), lat])


class KingCountyFeatures(BaseEstimator, TransformerMixin):
    """Clean raw King County data (via ``clean_raw``) and add location features.

    Adds two columns:
    - ``center_distance``: km from the reference point (CENTER_LAT, CENTER_LONG)
    - ``water_distance``: km to the nearest waterfront house seen during fit

    Differences from the notebook:
    - The waterfront reference set is learned in ``fit`` (training data only).
    - Nearest-neighbour search uses a KD-tree instead of a nested loop. The
      notebook recomputes the cos() correction for each reference house;
      this version uses the fixed ``LAT_CORRECTION``. The effect is small.
    - ``sqft_price`` is not computed here because it depends on ``price``,
      which is not available at inference time.
    """

    def fit(self, X: pd.DataFrame, y=None):
        cleaned = clean_raw(X)
        water = cleaned.loc[cleaned["waterfront"] == 1, ["lat", "long"]].to_numpy()
        if len(water) == 0:
            raise ValueError("At least one waterfront house is needed to fit water_distance.")
        # Store plain arrays, not a cKDTree: skops cannot serialize the tree.
        self.water_points_ = _project(water)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "water_points_")
        df = clean_raw(X)

        df["center_distance"] = _km_distance(
            np.abs(CENTER_LAT - df["lat"]),
            np.abs(CENTER_LONG - df["long"]),
        )

        tree = cKDTree(self.water_points_)
        dists, _ = tree.query(_project(df[["lat", "long"]].to_numpy()))
        df["water_distance"] = dists * KM_PER_DEGREE
        return df