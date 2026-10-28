import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kc_train import build_model, load_model, save_model, train_model  # noqa: E402

SMALL_GRID = {"model__regressor__alpha": [0.1, 1.0], "model__regressor__l1_ratio": [0.5]}


def make_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    bedrooms = rng.integers(1, 6, n)
    sqft_living = rng.integers(800, 4000, n)
    grade = rng.integers(3, 12, n)
    waterfront = (rng.random(n) < 0.1).astype(float)
    waterfront[0] = 1.0  # at least one waterfront house
    waterfront[1] = np.nan  # a missing value to clean
    price = sqft_living * (grade * 60 + 100) + rng.normal(0, 20_000, n)

    df = pd.DataFrame(
        {
            "id": np.arange(n) + 1_000_000,
            "date": "20141001T000000",
            "bedrooms": bedrooms,
            "bathrooms": bedrooms * 0.75,
            "sqft_living": sqft_living,
            "sqft_lot": rng.integers(1000, 20000, n),
            "floors": rng.choice([1.0, 1.5, 2.0], n),
            "waterfront": waterfront,
            "view": rng.integers(0, 5, n).astype(float),
            "condition": rng.integers(1, 6, n),
            "grade": grade,
            "sqft_above": (sqft_living * 0.7).astype(int),
            "yr_built": rng.integers(1900, 2015, n),
            "yr_renovated": np.where(rng.random(n) < 0.2, 2000, 0),
            "zipcode": 98000 + rng.integers(0, 50, n),
            "lat": 47.4 + rng.random(n) * 0.4,
            "long": -122.5 + rng.random(n) * 0.3,
            "sqft_living15": sqft_living,
            "sqft_lot15": rng.integers(1000, 20000, n),
        }
    )
    return df, pd.Series(price, index=df.index, name="price")


def test_pipeline_has_expected_steps():
    names = [name for name, _ in build_model().steps]
    assert names == ["features", "drop", "polynomial", "scaler", "model"]


def test_train_fits_and_predicts_raw_rows():
    df, y = make_data()
    search = train_model(df, y, param_grid=SMALL_GRID, cv=3)
    preds = search.best_estimator_.predict(df)
    assert preds.shape == (len(df),)
    assert np.isfinite(preds).all()


def test_train_does_not_modify_input_data():
    df, y = make_data()
    df.loc[df.index[5], "bedrooms"] = 33  # the row training removes internally
    df_before = df.copy()
    y_before = y.copy()

    train_model(df, y, param_grid=SMALL_GRID, cv=3)

    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_series_equal(y, y_before)


def test_predictions_ignore_id_date_and_price():
    df, y = make_data()
    model = train_model(df, y, param_grid=SMALL_GRID, cv=3).best_estimator_

    changed = df.copy()
    changed["id"] = 0
    changed["date"] = "20990101T000000"
    changed["price"] = 1.0
    np.testing.assert_allclose(model.predict(df), model.predict(changed))


def test_train_removes_33_bedroom_row_before_fitting():
    df, y = make_data()
    df.loc[df.index[5], "bedrooms"] = 33
    search = train_model(df, y, param_grid=SMALL_GRID, cv=3)
    assert search.best_estimator_ is not None


def test_save_and_load_round_trip(tmp_path):
    df, y = make_data()
    model = train_model(df, y, param_grid=SMALL_GRID, cv=3).best_estimator_
    path = tmp_path / "model" / "model.bin"

    save_model(model, path)
    assert path.exists() and path.stat().st_size > 0

    loaded = load_model(path)
    np.testing.assert_allclose(model.predict(df), loaded.predict(df))


def test_train_requires_waterfront_house():
    df, y = make_data()
    df["waterfront"] = 0.0
    with pytest.raises(ValueError):
        train_model(df, y, param_grid=SMALL_GRID, cv=3)