import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kc_features import CENTER_LAT, CENTER_LONG, KingCountyFeatures  # noqa: E402


def make_raw(n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "bedrooms": rng.integers(1, 6, n),
            "sqft_living": rng.integers(800, 4000, n),
            "sqft_above": rng.integers(500, 2000, n),
            "waterfront": [1, 0, np.nan, 0, 0, 1][:n],
            "view": [0, 2, np.nan, 0, 1, 0][:n],
            "yr_built": rng.integers(1900, 2015, n),
            "yr_renovated": [0, np.nan, 2005, 0, 0, np.nan][:n],
            "lat": 47.5 + rng.random(n) * 0.3,
            "long": -122.4 + rng.random(n) * 0.3,
        }
    )


def test_transform_adds_distance_features():
    raw = make_raw()
    out = KingCountyFeatures().fit(raw).transform(raw)
    assert "center_distance" in out.columns
    assert "water_distance" in out.columns
    assert (out["center_distance"] >= 0).all()
    assert (out["water_distance"] >= 0).all()


def test_transform_does_not_mutate_input():
    raw = make_raw()
    before = raw.copy()
    KingCountyFeatures().fit(raw).transform(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_water_distance_zero_for_waterfront_houses():
    raw = make_raw()
    out = KingCountyFeatures().fit(raw).transform(raw)
    water_rows = out[raw["waterfront"] == 1]
    assert np.allclose(water_rows["water_distance"], 0.0, atol=1e-9)


def test_water_reference_learned_only_from_fit_data():
    train = make_raw(seed=1)
    test = make_raw(seed=2)
    test["waterfront"] = 0.0  # no waterfront houses in the test frame

    model = KingCountyFeatures().fit(train)
    out = model.transform(test)
    assert out["water_distance"].notna().all()
    assert len(model.water_points_) == int((train["waterfront"] == 1).sum())


def test_fit_requires_a_waterfront_house():
    raw = make_raw()
    raw["waterfront"] = 0.0
    with pytest.raises(ValueError):
        KingCountyFeatures().fit(raw)


def test_center_distance_is_zero_at_center():
    raw = make_raw()
    raw.loc[0, ["lat", "long"]] = [CENTER_LAT, CENTER_LONG]
    out = KingCountyFeatures().fit(raw).transform(raw)
    assert out.loc[0, "center_distance"] == pytest.approx(0.0, abs=1e-9)


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        KingCountyFeatures().transform(make_raw())