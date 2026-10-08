"""Fixed temporal split and train-fitted logistic baseline pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN

SPLIT_BOUNDARIES = {
    'train': ('2005-03-31', '2005-12-31'),
    'validation': ('2006-03-31', '2006-05-31'),
    'test': ('2006-08-31', '2006-11-30'),
}
NUMERIC_COLUMNS = [c for c in PREDICTOR_COLUMNS if c not in CATEGORICAL_COLUMNS]


def temporal_split_labels(snapshot_dates: pd.Series) -> pd.Series:
    """Label the approved windows; all other dates are excluded, never training."""
    dates = pd.to_datetime(snapshot_dates, errors='raise')
    if dates.isna().any():
        raise ValueError('Snapshot dates must be nonmissing.')
    labels = pd.Series('excluded', index=dates.index, dtype='str')
    for name, (start, end) in SPLIT_BOUNDARIES.items():
        labels.loc[dates.between(start, end)] = name
    embargo = dates.dt.to_period('M').isin(pd.PeriodIndex(
        ['2006-01', '2006-02', '2006-06', '2006-07'], freq='M'))
    labels.loc[embargo] = 'embargo'
    return labels


def validate_model_panel(panel: pd.DataFrame) -> None:
    """Verify the Day 5 artifact contract without modelling held-out rows."""
    expected = set(METADATA_COLUMNS + PREDICTOR_COLUMNS + [TARGET_COLUMN])
    if panel.shape != (18472, 48) or set(panel.columns) != expected:
        raise ValueError('Expected the Day 5 panel: 18,472 rows and the 48 documented columns.')
    if not pd.api.types.is_datetime64_any_dtype(panel.snapshot_date):
        raise TypeError('Parse snapshot_date before validation.')
    if panel[METADATA_COLUMNS].isna().any().any() or panel.duplicated(METADATA_COLUMNS).any():
        raise ValueError('Missing metadata or duplicate player-snapshot keys.')
    if not panel.snapshot_date.dt.is_month_end.all():
        raise ValueError('Expected month-end snapshots.')
    if panel.UserID.nunique() != 4025:
        raise ValueError('Unexpected unique player count.')
    if panel[TARGET_COLUMN].value_counts().to_dict() != {0: 11865, 1: 6607}:
        raise ValueError('Unexpected target values/counts.')


def predictor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only dictionary predictors; identifiers, dates and target are excluded."""
    result = frame.loc[:, PREDICTOR_COLUMNS].copy()
    if not all(pd.api.types.is_numeric_dtype(result[c]) for c in NUMERIC_COLUMNS):
        raise TypeError('All documented numerical predictors must be numeric.')
    if np.isinf(result[NUMERIC_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError('Infinite predictors are invalid; undefined ratios must remain missing.')
    return result


def build_logistic_pipeline(*, class_weight: str | None = None) -> Pipeline:
    """Return a fresh, unfitted pipeline; caller must fit on TRAIN only."""
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC_COLUMNS),
        ('categorical', categorical, CATEGORICAL_COLUMNS),
    ], remainder='drop')
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', LogisticRegression(max_iter=5000, random_state=42, class_weight=class_weight)),
    ])
