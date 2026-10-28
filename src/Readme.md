# King County House Prices: Refactored Pipeline

This project refactors the King County house price notebook into Python modules with a reusable scikit-learn pipeline, tests, and a model training step.

The original analysis is in [King-County.ipynb](King-County.ipynb). The assignment brief is in [project-description.md](project-description.md).

## Project Structure

```
mle-refactoring-project/
├── King-County.ipynb          # original notebook (EDA, cleaning, modeling)
├── project-description.md     # assignment brief
├── src/
│   ├── __init__.py
│   ├── kc_cleaning.py         # cleaning: clean_raw, remove_implausible_rows
│   ├── kc_features.py         # feature engineering: KingCountyFeatures
│   └── kc_train.py            # modeling: build_model, train_model, save_model, load_model
└── tests/
    ├── test_kc_cleaning.py
    ├── test_kc_features.py
    └── test_kc_train.py
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Then select the `.venv` interpreter as the kernel for [King-County.ipynb](King-County.ipynb).

## What Each Module Does

**`src/kc_cleaning.py`**

- `clean_raw(df)`: recomputes `sqft_basement` as `sqft_living - sqft_above` (the raw column contains a `?`), fills missing `waterfront` and `view` with 0, and builds `last_known_change` from `yr_renovated` or `yr_built`. Returns a new DataFrame.
- `remove_implausible_rows(df)`: drops the record with 33 bedrooms, which is a data-entry error. Used on training data only.

**`src/kc_features.py`**

- `KingCountyFeatures`: a scikit-learn transformer that runs `clean_raw` and adds two features:
  - `center_distance`: distance in km from a reference point in Medina, WA.
  - `water_distance`: distance in km to the nearest waterfront house. The waterfront reference points are learned in `fit()` from training data only, so test data never affects them.

**`src/kc_train.py`**

- `build_model()`: one pipeline from raw records to predictions: feature engineering, polynomial terms (degree 2), standard scaling, and ElasticNet with a standardized target.
- `train_model(raw, y)`: removes implausible rows, then runs `GridSearchCV` with 5-fold cross-validation over `alpha` and `l1_ratio`. Works on copies, so the input data is not modified.
- `save_model` and `load_model`: save and load the fitted pipeline with [skops](https://skops.readthedocs.io/).

## Using the Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.kc_cleaning import remove_implausible_rows
from src.kc_features import KingCountyFeatures
from src.kc_train import train_model, save_model

raw = pd.read_csv("data/King_County_House_prices_dataset.csv")
train_raw, test_raw = train_test_split(raw, test_size=0.3, random_state=42)

# Features only
features = KingCountyFeatures().fit(remove_implausible_rows(train_raw))
X_train = features.transform(train_raw)

# Full model: cleaning, features and tuning in one call
search = train_model(train_raw, train_raw["price"])
model = search.best_estimator_
print("Test R^2:", model.score(test_raw, test_raw["price"]))
save_model(model, "model/model.bin")
```

## Running the Tests

```bash
uv run pytest -q
```

Expected output: `19 passed`.

## Notes on Changes from the Notebook

- **Split before feature engineering.** The notebook engineered features on all rows and then split. The refactored code splits first, which avoids leaking information from the test set. Test scores may differ slightly from the notebook.
- **No `inplace` changes.** The notebook's `inplace=True` drops are replaced by functions that return new DataFrames, so the loaded data is never modified.
- **Nearest-neighbour search.** `water_distance` uses a KD-tree instead of the notebook's nested loop, which is much faster. It uses a single fixed longitude correction, so values differ slightly from the notebook.
- **`sqft_price` is not computed in the pipeline**, because it depends on `price`, which is not available at prediction time.
