"""Model training for the King County price model.

Refactored from the modeling section of King-County.ipynb. The whole model,
including feature engineering, is one scikit-learn Pipeline, so it can take
raw house records at inference time.
"""

#from __future__ import annotations

from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.kc_cleaning import remove_implausible_rows
from src.kc_features import KingCountyFeatures

# Columns that are identifiers, raw dates, or the target. They are never features.
NON_FEATURE_COLUMNS = ["id", "date", "price"]

DEFAULT_PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop the named columns if they are present."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in (self.columns or []) if c in X.columns]
        return X.drop(columns=to_drop)


def build_model() -> Pipeline:
    """Raw house records -> cleaned features -> polynomial terms -> scaled -> ElasticNet.

    Steps are named so that GridSearchCV can address them, e.g.
    ``model__regressor__alpha``.
    """
    return Pipeline(
        [
            ("features", KingCountyFeatures()),
            ("drop", DropColumns(NON_FEATURE_COLUMNS)),
            ("polynomial", PolynomialFeatures(2, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def train_model(
    raw: pd.DataFrame,
    y: pd.Series,
    param_grid: dict | None = None,
    cv: int = 5,
) -> GridSearchCV:
    """Tune and fit the model on training data only.

    Implausible rows (the 33-bedroom record) are removed here, before fitting.
    They are not removed inside the pipeline, so prediction never drops rows.
    Returns the fitted GridSearchCV; ``best_estimator_`` is the refitted pipeline.
    """
    # Work on copies so the caller's DataFrame and Series are never modified.
    raw = raw.copy()
    y = y.copy()

    cleaned = remove_implausible_rows(raw)
    y_clean = y.loc[cleaned.index]

    search = GridSearchCV(
        build_model(),
        param_grid or DEFAULT_PARAM_GRID,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(cleaned, y_clean)
    return search


def save_model(model: Pipeline, path: str | Path) -> None:
    """Save the fitted pipeline with skops."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        sio.dump(model, f_out)


def load_model(path: str | Path) -> Pipeline:
    """Load a pipeline saved by save_model.

    skops refuses unknown types by default. This loader trusts every type the
    file contains, so only load files you saved yourself.
    """
    path = str(path)
    trusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=trusted)